"""Per-row mixed-precision (MP) dispatch in ``sc_conv2d``.

Two layers of checks, mirroring ``test_sc_conv.py``:

* **CPU (stubbed)** — ``sc_matmul`` is replaced with an exact fp64 matmul, so
  any level split must reproduce ``F.conv2d`` exactly. This proves the MP
  gather/scatter bookkeeping (classify -> per-level matmul -> scatter) is
  lossless, independent of SC noise. Also covers pruning (level 0 rows stay
  zero), the AdaptiveMPConfig quantile path, and the validation errors.

* **GPU (real Triton kernels)** — per-row quantization makes each row's SC
  result independent of which other rows share the launch, so an MP run must
  be *bit-identical per row* to the uniform run at that row's stoc_len:
  ``out_mp[rows_at_sl] == sc_conv2d(..., stoc_len=sl)[rows_at_sl]``.
  Checked for a single-level config (== whole baseline) and per level for a
  two-level config, on the pointwise fast path (incl. chunk_d) and dense 3x3.

Run:  python -m pytest tests/test_mp_conv.py -q     (or run the file directly)
"""
import torch
import torch.nn.functional as F

import scmp_kernels.sc.conv as conv_mod
from scmp_kernels import sc_conv2d
from scmp_kernels.mp import AdaptiveMPConfig, MPConfig, classify_rows_by_metric


def _exact(a, b, **kwargs):
    return (a.to(torch.float64) @ b.to(torch.float64).transpose(-1, -2)).to(a.dtype)


class _stub_sc_matmul:
    """Temporarily replace conv_mod.sc_matmul with the exact matmul."""

    def __enter__(self):
        self._orig = conv_mod.sc_matmul
        conv_mod.sc_matmul = _exact

    def __exit__(self, *exc):
        conv_mod.sc_matmul = self._orig


def _pw_rows(x):
    """The pointwise fast path's im2col rows: (B*H*W, Cin)."""
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(B * H * W, C)


# ---------------------------------------------------------------------------
# CPU: dispatch bookkeeping is exact (stubbed sc_matmul)
# ---------------------------------------------------------------------------

def test_mp_lowering_exact():
    torch.manual_seed(0)
    with _stub_sc_matmul():
        for name, mp in [
            ("MPConfig 2-level", MPConfig([256, 64])),
            ("MPConfig 3-level", MPConfig([256, 128, 64], [0.3, 0.3, 0.4])),
            ("Adaptive quantile", AdaptiveMPConfig(
                stoc_len_levels=[256, 64], target_fractions=[0.5, 0.5])),
        ]:
            # pointwise fast path
            x = torch.randn(2, 32, 12, 12)
            w = torch.randn(64, 32, 1, 1)
            b = torch.randn(64)
            ref = F.conv2d(x, w, b)
            got = sc_conv2d(x, w, b, mp_config=mp)
            err = (got - ref).abs().max().item()
            assert err < 1e-4, f"pointwise {name}: err {err:.3e}"
            # dense 3x3 (groups=1) path
            x = torch.randn(2, 8, 10, 10)
            w = torch.randn(16, 8, 3, 3)
            ref = F.conv2d(x, w, padding=1)
            got = sc_conv2d(x, w, padding=1, mp_config=mp)
            err = (got - ref).abs().max().item()
            assert err < 1e-4, f"dense {name}: err {err:.3e}"
            print(f"  ok  CPU exact dispatch      {name}")


def test_mp_pruning_zero_rows():
    torch.manual_seed(1)
    with _stub_sc_matmul():
        x = torch.randn(2, 16, 8, 8)
        w = torch.randn(24, 16, 1, 1)
        mp = MPConfig([256, 0], [0.5, 0.5])          # bottom half pruned
        got = sc_conv2d(x, w, mp_config=mp)           # no bias
        ref = F.conv2d(x, w)

        a = _pw_rows(x)
        assign = classify_rows_by_metric(
            a.abs().amax(dim=1), mp.stoc_len_levels, mp.level_fractions)
        kept, pruned = assign.level_row_indices[256], assign.level_row_indices[0]
        got_rows = _pw_rows(got)
        ref_rows = _pw_rows(ref)
        assert got_rows[pruned].abs().max().item() == 0.0, "pruned rows not zero"
        err = (got_rows[kept] - ref_rows[kept]).abs().max().item()
        assert err < 1e-4, f"kept rows err {err:.3e}"
        assert pruned.numel() > 0 and kept.numel() > 0
        print(f"  ok  CPU pruning              kept={kept.numel()} "
              f"pruned={pruned.numel()}")


def test_mp_depthwise_ignored():
    torch.manual_seed(2)
    with _stub_sc_matmul():
        x = torch.randn(2, 32, 10, 10)
        w = torch.randn(32, 1, 3, 3)
        mp = MPConfig([256, 64])
        ref = F.conv2d(x, w, padding=1, groups=32)
        got = sc_conv2d(x, w, padding=1, groups=32, mp_config=mp)
        err = (got - ref).abs().max().item()
        assert err < 1e-4, f"depthwise with mp_config: err {err:.3e}"
        print("  ok  CPU depthwise             mp ignored, lowering exact")


def test_mp_validation_errors():
    x = torch.randn(1, 8, 4, 4)
    w = torch.randn(8, 8, 1, 1)
    mp = MPConfig([256, 64])
    for kwargs, exc in [
        (dict(mp_config=mp, granularity="per_tensor"), ValueError),
        (dict(mp_config="not-a-config"), TypeError),
    ]:
        try:
            with _stub_sc_matmul():
                sc_conv2d(x, w, **kwargs)
        except exc:
            pass
        else:
            raise AssertionError(f"{kwargs} did not raise {exc.__name__}")
    print("  ok  CPU validation            per_tensor/bad-type raise")


# ---------------------------------------------------------------------------
# GPU: per-row independence => MP rows bit-match the uniform runs
# ---------------------------------------------------------------------------

def _gpu_cases():
    torch.manual_seed(3)
    dev = "cuda"
    return [
        # (name, x, w, conv_kwargs, extra sc kwargs)
        ("pointwise", torch.randn(2, 16, 8, 8, device=dev),
         torch.randn(32, 16, 1, 1, device=dev), {}, {}),
        ("pointwise chunk_d", torch.randn(2, 128, 6, 6, device=dev),
         torch.randn(32, 128, 1, 1, device=dev), {}, {"chunk_d": 64}),
        ("dense 3x3", torch.randn(2, 8, 8, 8, device=dev),
         torch.randn(16, 8, 3, 3, device=dev), {"padding": 1}, {}),
    ]


def _rows_view(y):
    """(B, Cout, H, W) -> (B*H*W, Cout), matching the im2col row order the
    2D paths classify (pointwise: B*H*W; dense: B*L with L=H*W row-major)."""
    B, C, H, W = y.shape
    return y.permute(0, 2, 3, 1).reshape(B * H * W, C)


def _im2col_rows(x, w, conv_kwargs):
    """Reproduce the 2D-path im2col operand to recompute the MP metric."""
    kH, kW = w.shape[2], w.shape[3]
    if kH == 1 and kW == 1 and not conv_kwargs:
        return _pw_rows(x)
    unf = F.unfold(x, (kH, kW), padding=conv_kwargs.get("padding", 0),
                   stride=conv_kwargs.get("stride", 1),
                   dilation=conv_kwargs.get("dilation", 1))
    return unf.transpose(1, 2).reshape(-1, unf.shape[1])


def test_gpu_mp_matches_uniform_per_level():
    if not torch.cuda.is_available():
        print("  skip GPU tests (no CUDA)")
        return
    levels, fracs = [256, 64], [0.5, 0.5]
    for name, x, w, ck, sk in _gpu_cases():
        uniform = {sl: sc_conv2d(x, w, stoc_len=sl, **ck, **sk)
                   for sl in levels}

        # single level == whole uniform baseline
        got1 = sc_conv2d(x, w, mp_config=MPConfig([256], [1.0]), **ck, **sk)
        d1 = (got1 - uniform[256]).abs().max().item()
        assert d1 == 0.0, f"{name}: single-level MP != uniform (max diff {d1:.3e})"

        # two-level: each bucket bit-matches its own uniform run
        mp = MPConfig(levels, fracs)
        got2 = sc_conv2d(x, w, mp_config=mp, **ck, **sk)
        a = _im2col_rows(x.to(torch.float32), w, ck)
        assign = classify_rows_by_metric(a.abs().amax(dim=1), levels, fracs)
        got_rows = _rows_view(got2)
        for sl in levels:
            rows = assign.level_row_indices[sl]
            assert rows.numel() > 0
            d = (got_rows[rows] - _rows_view(uniform[sl])[rows]).abs().max().item()
            assert d == 0.0, (f"{name}: level {sl} rows differ from uniform "
                              f"stoc_len={sl} run (max diff {d:.3e})")
        print(f"  ok  GPU per-level bit-match  {name}")


def test_gpu_mp_halve_composition():
    """MP + halve composes like the LLM's SCLinear: the grid halves
    (rng_levels -> 2**(sc_prec-1)) and MP levels are effective cycle counts
    in the halved space. A single-level [128] MP run under halve must
    bit-match the uniform halve run (stoc_len=None -> 128/128)."""
    if not torch.cuda.is_available():
        print("  skip GPU halve test (no CUDA)")
        return
    torch.manual_seed(4)
    x = torch.randn(2, 16, 8, 8, device="cuda")
    w = torch.randn(32, 16, 1, 1, device="cuda")
    uniform = sc_conv2d(x, w, stoc_len=None, halve_bipolar_stoc_len=True)
    got = sc_conv2d(x, w, mp_config=MPConfig([128], [1.0]),
                    halve_bipolar_stoc_len=True)
    d = (got - uniform).abs().max().item()
    assert d == 0.0, f"MP[128]+halve != uniform halve (max diff {d:.3e})"
    # two-level halved-space config runs and stays finite
    got2 = sc_conv2d(x, w, mp_config=MPConfig([128, 32], [0.5, 0.5]),
                     halve_bipolar_stoc_len=True)
    assert torch.isfinite(got2).all()
    print("  ok  GPU halve composition     MP[128]+halve == uniform halve")


if __name__ == "__main__":
    test_mp_lowering_exact()
    test_mp_pruning_zero_rows()
    test_mp_depthwise_ignored()
    test_mp_validation_errors()
    test_gpu_mp_matches_uniform_per_level()
    test_gpu_mp_halve_composition()
    print("\nsc_conv2d MIXED PRECISION — ALL PASSED")
