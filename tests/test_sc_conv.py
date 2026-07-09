"""Correctness of the sc_conv2d im2col/reshape lowering (CPU, no GPU needed).

The SC quantization noise is a GPU/Triton concern; the *lowering* — the pure
reshape/unfold bookkeeping that turns each dispatch path into an ``a @ b.T`` —
must be exact. We stub ``sc_matmul`` with an exact fp64 ``a @ b.T`` and check
``sc_conv2d`` reproduces ``F.conv2d`` for every geometry (pointwise fast path,
depthwise, dense kxk, and a general grouped conv).

Run:  python -m pytest tests/test_sc_conv.py -q     (or run the file directly)
"""
import torch
import torch.nn.functional as F

import scmp_kernels.sc.conv as conv_mod
from scmp_kernels import sc_conv2d


def _exact(a, b, **kwargs):
    return (a.to(torch.float64) @ b.to(torch.float64).transpose(-1, -2)).to(a.dtype)


def _check(name, x, w, bias=None, *, stride=1, padding=0, dilation=1, groups=1,
           tol=1e-4):
    ref = F.conv2d(x, w, bias, stride=stride, padding=padding,
                   dilation=dilation, groups=groups)
    got = sc_conv2d(x, w, bias, stride=stride, padding=padding,
                    dilation=dilation, groups=groups)
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    err = (got - ref).abs().max().item()
    assert err < tol, f"{name}: max abs err {err:.3e} > {tol}"
    print(f"  ok  {name:34s} err={err:.2e}")
    return err


def test_lowering_exact():
    conv_mod.sc_matmul = _exact          # stub the SC matmul with an exact one
    torch.manual_seed(0)
    B = 2
    worst = 0.0
    # pointwise fast path (1x1, stride1, pad0, groups1)
    worst = max(worst, _check("pointwise 1x1 g1",
                torch.randn(B, 32, 12, 12), torch.randn(64, 32, 1, 1),
                torch.randn(64)))
    # 1x1 with stride 2 -> falls to the general path (not the fast path)
    worst = max(worst, _check("1x1 stride2 (general)",
                torch.randn(B, 32, 12, 12), torch.randn(16, 32, 1, 1),
                stride=2))
    # dense 3x3 stem, groups1, stride2, pad1
    worst = max(worst, _check("dense 3x3 s2 p1",
                torch.randn(B, 3, 32, 32), torch.randn(16, 3, 3, 3),
                torch.randn(16), stride=2, padding=1))
    # depthwise 3x3 / 5x5
    worst = max(worst, _check("depthwise 3x3 s1 p1",
                torch.randn(B, 64, 16, 16), torch.randn(64, 1, 3, 3),
                torch.randn(64), padding=1, groups=64))
    worst = max(worst, _check("depthwise 5x5 s2 p2",
                torch.randn(B, 72, 16, 16), torch.randn(72, 1, 5, 5),
                stride=2, padding=2, groups=72))
    # general grouped conv (1 < groups < Cin)
    worst = max(worst, _check("grouped g4 3x3",
                torch.randn(B, 16, 10, 10), torch.randn(32, 4, 3, 3),
                torch.randn(32), padding=1, groups=4))
    # dilation
    worst = max(worst, _check("dense 3x3 dil2",
                torch.randn(B, 8, 16, 16), torch.randn(8, 8, 3, 3),
                padding=2, dilation=2))
    print(f"[test_lowering_exact] worst err = {worst:.2e}")


if __name__ == "__main__":
    test_lowering_exact()
    print("\nsc_conv2d LOWERING EXACT — ALL PASSED")
