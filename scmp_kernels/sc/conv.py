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

Mixed precision (optional, ``mp_config=``): the CNN analog of the LLM's
per-token-row MP. The im2col rows of the two 2D paths (pointwise / dense
``groups=1``) are the exact CNN counterpart of token rows — each row is one
spatial position — so the same classifiers from ``scmp_kernels.mp``
(quantile / calibrated-table / free-boundary) apply unchanged: rows are ranked
by their abs-max, bucketed into ``stoc_len`` levels, and each bucket runs its
own ``sc_matmul`` at that stream length (level ``0`` = prune: rows stay zero,
bias only). The depthwise 3D path and arbitrary grouped convs keep the uniform
``stoc_len`` (depthwise is <2% of MobileNet MACs and usually left FP anyway).
``mp_config=None`` (default) leaves every path byte-identical to before.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .matmul import sc_matmul

__all__ = ["sc_conv2d"]

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


# ---------------------------------------------------------------------------
# Mixed precision (per-im2col-row) — CNN analog of the LLM's per-token-row MP
# ---------------------------------------------------------------------------

def _mp_classify(a: torch.Tensor, mp_config, operator: str):
    """Bucket the im2col rows of ``a`` into stoc_len levels.

    Reuses the transformer-side classifiers from ``scmp_kernels.mp`` verbatim
    (imported lazily so plain sc_conv2d never touches the mp package). The
    CNN row-importance metric is the per-row abs-max — the direct analog of
    the LLM's ``x_row_max`` — computed on the already-lowered 2D operand, so
    it is the same rows the SC matmul will quantize per-row.
    """
    from ..mp import (
        AdaptiveMPConfig,
        MPConfig,
        adaptive_classify_rows,
        classify_rows_by_metric,
        get_current_block_idx,
    )

    metric = a.abs().amax(dim=1)
    # FreeBoundaryMPConfig subclasses AdaptiveMPConfig — check Adaptive first.
    if isinstance(mp_config, AdaptiveMPConfig):
        return adaptive_classify_rows(
            metric, mp_config, operator=operator,
            block_idx=get_current_block_idx())
    if isinstance(mp_config, MPConfig):
        return classify_rows_by_metric(
            metric, mp_config.stoc_len_levels, mp_config.level_fractions)
    raise TypeError(
        f"sc_conv2d: mp_config must be an scmp_kernels.mp MPConfig or "
        f"AdaptiveMPConfig (incl. FreeBoundaryMPConfig), "
        f"got {type(mp_config).__name__}")


def _sc_matmul_mp(
    a: torch.Tensor,
    w: torch.Tensor,
    mp_config,
    operator: str,
    *,
    granularity: str,
    chunk_d: int,
    mm: dict,
) -> torch.Tensor:
    """Per-level SC matmul dispatch: ``out[rows_l] = sc_matmul(a[rows_l], w,
    stoc_len=level_l)`` for each stoc_len level, scattered into one output.

    Mirrors the LLM attention patch's dispatch loop. ``stoc_len`` is a Triton
    ``tl.constexpr``, so per-row stream lengths necessarily mean one kernel
    launch per level — rows are gathered per bucket and results scattered
    back. Per-row quantization makes each row's result independent of which
    other rows share the launch, so a single-level assignment is bit-identical
    to the uniform call. Level ``0`` rows are pruned (skipped): they stay
    zero, and only the fp32 bias (added by the caller) reaches the output.
    """
    assignment = _mp_classify(a, mp_config, operator)
    out = torch.zeros(a.shape[0], w.shape[0],
                      dtype=torch.float32, device=a.device)
    for sl, rows in assignment.level_row_indices.items():
        if rows.numel() == 0 or sl == 0:
            continue
        out[rows] = sc_matmul(
            a[rows].contiguous(), w, granularity=granularity,
            chunk_d=chunk_d, **{**mm, "stoc_len": int(sl)})
    return out


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
        mp_config: optional ``scmp_kernels.mp`` config (``MPConfig`` /
            ``AdaptiveMPConfig`` / ``FreeBoundaryMPConfig``) enabling per-row
            mixed precision on the 2D paths (pointwise + dense ``groups=1``):
            im2col rows are classified by abs-max and each stoc_len level runs
            its own :func:`sc_matmul` (level ``0`` = prune). Overrides
            ``stoc_len`` per level; requires ``granularity="per_row"``.
            Composes with ``halve_bipolar_stoc_len`` the same way as the LLM's
            SCLinear: halving shrinks only the RNG grid (``rng_levels`` →
            ``2**(sc_prec-1)``), and the MP levels are read as *effective*
            cycle counts in the halved space (e.g. ``[128, 32]`` at
            ``sc_prec=8``). Depthwise / grouped paths ignore it (uniform
            ``stoc_len``). Default ``None`` — the single-launch behavior,
            byte-identical to before.
        **sc_kwargs: extra :func:`sc_matmul` kwargs (``group_a``, ``group_b``,
            ``rng_levels``, ``config``, ``smooth_scales``).

    Returns:
        ``(B, Cout, Hout, Wout)`` with dtype matching ``x``.
    """
    if mp_config is not None and granularity != "per_row":
        raise ValueError(
            f"sc_conv2d: mp_config requires granularity='per_row' (per-row "
            f"quantization is what makes per-level dispatch equivalent to "
            f"the uniform call), got '{granularity}'.")

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

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
            y = _sc_matmul_mp(a, w, mp_config, "pw",
                              granularity=granularity, chunk_d=chunk_d, mm=mm)
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
            y = _sc_matmul_mp(a, w, mp_config, "conv",
                              granularity=granularity, chunk_d=chunk_d, mm=mm)
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
