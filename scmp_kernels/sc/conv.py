"""SC 2D convolution built on the SC matmul kernels.

``sc_conv2d`` is the convolution analog of :func:`sc_matmul`: it computes
``y = conv2d(x, weight, bias)`` with the multiply-accumulate done in stochastic
computing, by lowering the convolution to ``a @ b.T`` and reusing the tuned
Triton SC-matmul kernels. All SC quantization (int-quant, bitstream MAC) happens
inside those kernels — this module only reshapes. fp32 in/out.

Three dispatch paths, picked automatically from the conv geometry:

* **pointwise fast path** — 1×1, stride 1, no padding/dilation, ``groups=1``:
  a pure reshape, **no im2col**. ``x (B,Cin,H,W) -> (B·H·W, Cin)`` matmuls with
  ``weight (Cout,Cin)``. This is the MAC-dominant case in MobileNet /
  EfficientNet / most modern CNNs, and it never materializes an unfolded tensor.
* **depthwise** — ``groups == Cin == Cout``: a batched 3D matmul, one tiny
  matmul per channel (``a3 (B·C, L, kH·kW) × b3 (B·C, 1, kH·kW)``). The im2col
  here is only ``kH·kW`` columns wide, so materializing it is cheap.
* **general** — any other ``groups`` (incl. a dense kxk conv, ``groups=1``):
  ``F.unfold`` im2col per group, then a 2D matmul. Correct for arbitrary
  grouped convs; the im2col cost is the classic one (only this path pays it).

Per-row mixed precision (``mp_config``): the pointwise and dense-``groups=1``
paths — the two that lower to a single ``(rows, K)`` matmul — support the same
per-row ``stoc_len`` dispatch as ``scmp_llm``'s SCLinear: each output spatial
position (one lowered row) is classified by its row abs-max and routed to one
of the configured levels, one ``sc_matmul`` call per level, outputs scattered
back. Each spatial position is the CNN analog of the LLM's token row. The
depthwise and arbitrary-grouped paths ignore ``mp_config`` (uniform stream);
in MobileNet-class nets they are a small share of the MACs and the calibrator
excludes them from the budget for consistency.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .matmul import sc_matmul

__all__ = ["sc_conv2d", "conv2d_lowered_rows"]

_IntOrPair = Union[int, Sequence[int]]


def _pair(v: _IntOrPair) -> tuple[int, int]:
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _out_hw(H, W, kHW, sHW, pHW, dHW) -> tuple[int, int]:
    (kH, kW), (sH, sW), (pH, pW), (dH, dW) = kHW, sHW, pHW, dHW
    Hout = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    Wout = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    return Hout, Wout


def _mp_dispatch_rows(
    a: torch.Tensor,
    w: torch.Tensor,
    *,
    granularity: str,
    chunk_d: int,
    mm: dict,
    mp_config,
    default_operator: str,
    mp_operator: Optional[str] = None,
    mp_block_idx: Optional[int] = None,
    mp_total_blocks: Optional[int] = None,
    mp_logger: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """Per-row mixed-precision matmul: classify each lowered row of ``a`` by
    its abs-max, then call ``sc_matmul`` once per stoc_len level on that
    level's row subset and scatter back. Mirrors scmp_llm SCLinear's MP path
    (per_row quant groups are independent per row, so a row-subset call is
    bit-identical to the same rows inside the full-batch call).

    ``mp_operator`` / ``mp_block_idx`` / ``mp_total_blocks`` are the module's
    identity for the AdaptiveMPConfig threshold-table lookup (per-(operator,
    block) MP — the scmp_llm cross-layer granularity). When ``mp_operator`` is
    None the geometry-class ``default_operator`` ("pw" / "conv") is used, so a
    plain ``sc_conv2d(mp_config=...)`` call (fixed-fraction MPConfig or an
    AdaptiveMPConfig quantile / per-op-default) still works.

    ``mp_logger(stoc_len, n_rows)`` is invoked once per non-empty level so the
    caller can accumulate the realized (row / MAC weighted) average stream
    length without this module depending on any tracker.
    """
    from ..mp.config import (            # local import: mp never imports sc,
        AdaptiveMPConfig,                # but keep the coupling one-way anyway
        MPConfig,
        adaptive_classify_rows,
        classify_rows_by_metric,
    )

    operator = mp_operator if mp_operator is not None else default_operator
    metric = a.abs().amax(dim=-1)
    if isinstance(mp_config, AdaptiveMPConfig):
        assignment = adaptive_classify_rows(
            metric, mp_config,
            operator=operator,
            block_idx=mp_block_idx,
            total_blocks=mp_total_blocks,
        )
    elif isinstance(mp_config, MPConfig):
        assignment = classify_rows_by_metric(
            metric, mp_config.stoc_len_levels, mp_config.level_fractions)
    else:
        raise TypeError(
            f"sc_conv2d: mp_config must be an scmp_kernels.mp MPConfig or "
            f"AdaptiveMPConfig (or subclass), got {type(mp_config).__name__}")

    out = torch.zeros((a.shape[0], w.shape[0]),
                      dtype=torch.float32, device=a.device)
    mm_level = dict(mm)
    for sl, indices in assignment.level_row_indices.items():
        n = int(indices.numel())
        if n == 0:
            continue
        if mp_logger is not None:
            mp_logger(int(sl), n)
        if sl <= 0:                       # explicit skip level (prune → zero)
            continue
        mm_level["stoc_len"] = int(sl)
        a_sub = a.index_select(0, indices).contiguous()
        out[indices] = sc_matmul(
            a_sub, w, granularity=granularity, chunk_d=chunk_d, **mm_level)
    return out


@torch.no_grad()
def conv2d_lowered_rows(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    stride: _IntOrPair = 1,
    padding: _IntOrPair = 0,
    dilation: _IntOrPair = 1,
    groups: int = 1,
) -> Optional[torch.Tensor]:
    """Return the lowered ``(rows, K)`` input matrix that :func:`sc_conv2d`
    row-dispatches on (row = output spatial position, in the same order as the
    kernel's own lowering), or ``None`` for the depthwise / grouped paths
    (which do not support per-row MP).

    The MP calibrator uses this so its per-row metric is computed on exactly
    the rows the runtime classifier sees — the CNN analog of the LLM rule that
    calibration and runtime must share one metric surface.
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    B, Cin, H, W = x.shape
    kH, kW = int(weight.shape[2]), int(weight.shape[3])
    x32 = x.to(torch.float32)

    is_pointwise = (kH == 1 and kW == 1 and stride == (1, 1)
                    and padding == (0, 0) and dilation == (1, 1) and groups == 1)
    if is_pointwise:
        return x32.permute(0, 2, 3, 1).reshape(B * H * W, Cin).contiguous()
    if groups == 1:
        K = Cin * kH * kW
        unf = F.unfold(x32, (kH, kW), dilation=dilation,
                       padding=padding, stride=stride)          # (B, K, L)
        L = unf.shape[-1]
        return unf.transpose(1, 2).reshape(B * L, K).contiguous()
    return None


@torch.no_grad()
def sc_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    stride: _IntOrPair = 1,
    padding: _IntOrPair = 0,
    dilation: _IntOrPair = 1,
    groups: int = 1,
    granularity: str = "per_row",
    mode: str = "bipolar",
    sc_prec: int = 8,
    stoc_len: Optional[int] = None,
    chunk_d: int = 0,
    halve_bipolar_stoc_len: bool = False,
    mp_config=None,
    mp_operator: Optional[str] = None,
    mp_block_idx: Optional[int] = None,
    mp_total_blocks: Optional[int] = None,
    mp_logger: Optional[Callable[[int, int], None]] = None,
    **sc_kwargs,
) -> torch.Tensor:
    """Stochastic-computing 2D convolution ``conv2d(x, weight, bias)``.

    Args:
        x: input ``(B, Cin, H, W)``.
        weight: ``(Cout, Cin // groups, kH, kW)`` (torch conv layout).
        bias: optional ``(Cout,)`` added in fp32 after the SC MAC.
        stride/padding/dilation: int or (h, w) pair, as in ``nn.Conv2d``.
        groups: conv groups. ``1`` (incl. 1×1), depthwise (``groups==Cin==Cout``),
            and arbitrary grouped convs are all supported.
        granularity: SC quant scope forwarded to :func:`sc_matmul` for the 2D
            paths (``per_row`` / ``per_tensor``). The depthwise batched path
            always uses ``per_row``.
        mode / sc_prec / stoc_len / chunk_d / halve_bipolar_stoc_len: forwarded
            to :func:`sc_matmul` unchanged. ``chunk_d`` applies only to the 2D
            paths (pointwise / dense / grouped), never the depthwise 3D path.
        mp_config: optional ``scmp_kernels.mp`` MPConfig / AdaptiveMPConfig.
            Enables per-row (per output spatial position) stoc_len dispatch on
            the pointwise and dense-``groups=1`` paths; MP levels are passed as
            explicit ``stoc_len`` per level call (in halved space when
            ``halve_bipolar_stoc_len``). Ignored (uniform stream) on the
            depthwise / grouped paths.
        mp_operator / mp_block_idx / mp_total_blocks: identity for the
            AdaptiveMPConfig threshold-table lookup (analog of the LLM's
            ``_sc_op_name`` / ``_sc_block_idx``).
        mp_logger: optional ``f(stoc_len, n_rows)`` callback, called once per
            dispatched level (realized-budget tracking).
        **sc_kwargs: extra :func:`sc_matmul` kwargs (``group_a``, ``group_b``,
            ``rng_levels``, ``config``, ``smooth_scales``).

    Returns:
        ``(B, Cout, Hout, Wout)`` with dtype matching ``x``.
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    # Per-row MP requires the per_row quant path (rows independent per launch).
    if mp_config is not None and granularity != "per_row":
        raise ValueError(
            f"sc_conv2d: mp_config requires granularity='per_row' (per-row "
            f"stoc_len dispatch), got granularity={granularity!r}.")

    B, Cin, H, W = x.shape
    Cout = weight.shape[0]
    kH, kW = int(weight.shape[2]), int(weight.shape[3])
    Hout, Wout = _out_hw(H, W, (kH, kW), stride, padding, dilation)

    x32 = x.to(torch.float32)
    w32 = weight.to(torch.float32)
    mm = dict(mode=mode, sc_prec=sc_prec, stoc_len=stoc_len,
              halve_bipolar_stoc_len=halve_bipolar_stoc_len, **sc_kwargs)

    is_pointwise = (kH == 1 and kW == 1 and stride == (1, 1)
                    and padding == (0, 0) and dilation == (1, 1) and groups == 1)
    is_depthwise = (groups == Cin and groups == Cout and groups != 1)

    if is_pointwise:
        # No im2col: each spatial position is already a length-Cin row.
        a = x32.permute(0, 2, 3, 1).reshape(B * H * W, Cin).contiguous()
        w = w32.reshape(Cout, Cin).contiguous()
        if mp_config is not None:
            y = _mp_dispatch_rows(
                a, w, granularity=granularity, chunk_d=chunk_d, mm=mm,
                mp_config=mp_config, default_operator="pw",
                mp_operator=mp_operator, mp_block_idx=mp_block_idx,
                mp_total_blocks=mp_total_blocks, mp_logger=mp_logger)
        else:
            y = sc_matmul(a, w, granularity=granularity, chunk_d=chunk_d, **mm)
        out = y.reshape(B, H, W, Cout).permute(0, 3, 1, 2)

    elif is_depthwise:
        kk = kH * kW
        unf = F.unfold(x32, (kH, kW), dilation=dilation,
                       padding=padding, stride=stride)          # (B, Cin*kk, L)
        L = unf.shape[-1]
        a3 = (unf.view(B, Cin, kk, L).permute(0, 1, 3, 2)
              .reshape(B * Cin, L, kk).contiguous())
        w = w32.view(Cin, kk)
        b3 = (w.view(Cin, 1, kk).unsqueeze(0).expand(B, Cin, 1, kk)
              .reshape(B * Cin, 1, kk).contiguous())
        y3 = sc_matmul(a3, b3, granularity="per_row", **mm)     # (B*Cin, L, 1)
        out = y3.reshape(B, Cout, Hout, Wout)

    elif groups == 1:
        K = Cin * kH * kW
        unf = F.unfold(x32, (kH, kW), dilation=dilation,
                       padding=padding, stride=stride)          # (B, K, L)
        L = unf.shape[-1]
        a = unf.transpose(1, 2).reshape(B * L, K).contiguous()
        w = w32.reshape(Cout, K).contiguous()
        if mp_config is not None:
            y = _mp_dispatch_rows(
                a, w, granularity=granularity, chunk_d=chunk_d, mm=mm,
                mp_config=mp_config, default_operator="conv",
                mp_operator=mp_operator, mp_block_idx=mp_block_idx,
                mp_total_blocks=mp_total_blocks, mp_logger=mp_logger)
        else:
            y = sc_matmul(a, w, granularity=granularity, chunk_d=chunk_d, **mm)
        out = y.reshape(B, L, Cout).transpose(1, 2).reshape(B, Cout, Hout, Wout)

    else:
        # Arbitrary grouped conv: lower each group independently.
        Cin_g, Cout_g = Cin // groups, Cout // groups
        K = Cin_g * kH * kW
        parts = []
        for g in range(groups):
            xg = x32[:, g * Cin_g:(g + 1) * Cin_g]
            wg = w32[g * Cout_g:(g + 1) * Cout_g]
            unf = F.unfold(xg, (kH, kW), dilation=dilation,
                           padding=padding, stride=stride)
            L = unf.shape[-1]
            a = unf.transpose(1, 2).reshape(B * L, K).contiguous()
            w = wg.reshape(Cout_g, K).contiguous()
            y = sc_matmul(a, w, granularity=granularity, chunk_d=chunk_d, **mm)
            parts.append(y.reshape(B, L, Cout_g).transpose(1, 2)
                         .reshape(B, Cout_g, Hout, Wout))
        out = torch.cat(parts, dim=1)

    if bias is not None:
        out = out + bias.to(torch.float32).reshape(1, -1, 1, 1)
    return out.to(x.dtype)
