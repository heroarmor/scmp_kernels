"""Enable-table cache keys must include the scramble parameters.

The enable tables bake the Owen scramble in at build time
(``build_enable_tables`` -> ``_prepare_rng_prefix`` -> ``_owen_scramble``),
but the cache keys used to cover only (config, sc_prec, stoc_len,
rng_levels). Switching ``SC_OWEN_MODE`` / ``SC_SCRAMBLE_MASKS``
mid-process silently reused stale tables unless the caller remembered
``clear_rng_cache()``.

Checks (GPU — table builds launch Triton kernels):

* truncated stream (stoc_len < 2**sc_prec, the fixed-level / MP regime):
  toggling SC_OWEN_MODE between bitrev and off WITHOUT clear_rng_cache
  changes the result, and toggling back reproduces the original bits
  (both mode entries live in the cache side by side);
* SC_SCRAMBLE_MASKS (mask count) is likewise part of the key;
* full-length stream never scrambles: results identical across modes and
  the shared "|scr=none" entry means no cache growth on a mode switch.

Run:  python -m pytest tests/test_scramble_cache.py -q   (or run directly)
"""
import os

import torch

from scmp_kernels import sc_matmul
import scmp_kernels.sc.kernels as K


def _run(stoc_len):
    torch.manual_seed(0)
    a = torch.randn(8, 64, device="cuda")
    b = torch.randn(16, 64, device="cuda")
    return sc_matmul(a, b, granularity="per_row", stoc_len=stoc_len)


class _env:
    """Temporarily set/unset env vars, restoring on exit."""

    def __init__(self, **kv):
        self._kv = kv

    def __enter__(self):
        self._old = {k: os.environ.get(k) for k in self._kv}
        for k, v in self._kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_scramble_mode_in_cache_key():
    if not torch.cuda.is_available():
        print("  skip (no CUDA)")
        return
    K.clear_rng_cache()

    with _env(SC_OWEN_MODE=None, SC_SCRAMBLE_MASKS=None):
        y_bitrev = _run(64)                       # default: bitrev, M=64
        with _env(SC_OWEN_MODE="off"):            # no clear_rng_cache!
            y_off = _run(64)
        y_bitrev2 = _run(64)                      # back to default

    d_modes = (y_bitrev - y_off).abs().max().item()
    assert d_modes > 0.0, (
        "SC_OWEN_MODE=off returned the bitrev result — stale cached table "
        "(scramble mode missing from the enable-table cache key)")
    d_back = (y_bitrev - y_bitrev2).abs().max().item()
    assert d_back == 0.0, (
        f"returning to bitrev must reproduce the original bits, diff {d_back:.3e}")
    print(f"  ok  mode in key            off vs bitrev diff={d_modes:.3e}, "
          f"bitrev roundtrip exact")


def test_scramble_mask_count_in_cache_key():
    if not torch.cuda.is_available():
        print("  skip (no CUDA)")
        return
    K.clear_rng_cache()

    with _env(SC_OWEN_MODE=None, SC_SCRAMBLE_MASKS=None):
        y_m64 = _run(64)
        with _env(SC_SCRAMBLE_MASKS="16"):        # no clear_rng_cache!
            y_m16 = _run(64)
        y_m64_again = _run(64)

    d = (y_m64 - y_m16).abs().max().item()
    assert d > 0.0, (
        "SC_SCRAMBLE_MASKS=16 returned the M=64 result — stale cached table "
        "(mask count missing from the enable-table cache key)")
    d_back = (y_m64 - y_m64_again).abs().max().item()
    assert d_back == 0.0, f"M=64 roundtrip must be exact, diff {d_back:.3e}"
    print(f"  ok  mask count in key      M=16 vs M=64 diff={d:.3e}, "
          f"M=64 roundtrip exact")


def test_full_length_shares_none_tag():
    if not torch.cuda.is_available():
        print("  skip (no CUDA)")
        return
    K.clear_rng_cache()

    with _env(SC_OWEN_MODE=None, SC_SCRAMBLE_MASKS=None):
        y_a = _run(256)                           # full length: no scramble
        n_entries = len(K._enable_table_cache)
        with _env(SC_OWEN_MODE="off"):
            y_b = _run(256)
        n_after = len(K._enable_table_cache)

    d = (y_a - y_b).abs().max().item()
    assert d == 0.0, (
        f"full-length results must be scramble-independent, diff {d:.3e}")
    assert n_after == n_entries, (
        f"full-length entries must share the |scr=none tag — cache grew "
        f"{n_entries} -> {n_after} on a mode switch")
    print("  ok  full-length none tag    identical bits, no cache growth")


if __name__ == "__main__":
    test_scramble_mode_in_cache_key()
    test_scramble_mask_count_in_cache_key()
    test_full_length_shares_none_tag()
    print("\nSCRAMBLE CACHE KEY — ALL PASSED")
