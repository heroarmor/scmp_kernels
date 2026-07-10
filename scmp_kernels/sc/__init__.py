"""Stochastic-computing kernels.

Public surface:

* ``sc_matmul`` — unified granularity-dispatcher. Accepts
  ``granularity={"per_tensor","per_row","per_head"}`` plus orthogonal
  knobs (``mode``, ``sc_prec``, ``stoc_len``, ``chunk_d``,
  ``group_a``/``group_b``, ``rng_levels``, ``config``). All historical
  specialized entry points (per-tensor, per-row, per-row-MLP,
  per-row-grouped, per-head-bipolar) are reachable through this single
  function.
* ``sc_conv2d`` — 2D convolution built on ``sc_matmul``. Lowers the conv to a
  matmul (1×1 → pure reshape, no im2col; depthwise → batched; general kxk /
  grouped → im2col) and forwards the same SC knobs. The conv analog of
  ``sc_matmul``.
* ``clear_rng_cache`` — drop cached RNG sequences (call after changing
  Sobol/Owen env vars or rotating seeds).
* ``det_kernel_tuning`` — context manager opting in to det-tuned tile
  sizes on the batched grouped path.

All inputs/outputs are float32; quantization happens inside the Triton kernels.
"""

from .matmul import sc_matmul
from .conv import sc_conv2d, conv2d_lowered_rows
from .kernels import clear_rng_cache, det_kernel_tuning

__all__ = [
    "sc_matmul",
    "sc_conv2d",
    "conv2d_lowered_rows",
    "clear_rng_cache",
    "det_kernel_tuning",
]
