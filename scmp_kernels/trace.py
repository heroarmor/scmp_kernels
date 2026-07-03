"""SC computation trace — precision/shape log for external HW simulators.

Records, for every ``sc_matmul`` call, the *effective* precision
(``stoc_len`` = SC cycle count, post-halving; ``rng_levels`` = resolved
enable-grid size) together with the matmul shape and the caller-supplied
identity (operator, decoder block, sub-unit such as a MoE expert or
attention head). An energy/latency simulator can then compute e.g.
``MACs x energy_per_MAC(stoc_len)`` per group, or replay the ordered
per-call timeline.

Design constraints (why it looks the way it does):

* **Zero overhead when off.** The only cost on the hot path is reading one
  module-level bool (``_ENABLED``). Callers guard with
  ``if trace._ENABLED:`` before doing any work.
* **No device synchronization.** Only host-side metadata is recorded
  (tensor *shapes* and plain ints) — never tensor values, ``.item()``, or
  anything that would stall the async CUDA pipeline.
* **No heavy imports.** Pure stdlib; importable everywhere the package is.
* **Bounded memory.** Trace mode spills to disk every ``_SPILL`` records,
  so week-long decode runs cannot OOM the host.
* **Thread-safe.** Context is thread-local; accumulation is locked
  (relevant for e.g. RULER's ``--threads > 1`` prediction driver).

Usage
-----
Enable via env (picked up at import time) or API::

    SC_MP_TRACE=/path/out.json            # enables; summary mode (default)
    SC_MP_TRACE_MODE=trace                # per-call JSONL instead

    from scmp_kernels import trace
    trace.enable("/path/out.json", mode="summary")

Applications tag identity before their matmuls (cheap; overwrite per call)::

    if trace._ENABLED:
        trace.set_context(op="down_proj", block=12, unit=expert_idx)

``flush()`` writes the file and resets by default. An ``atexit`` hook
flushes automatically (header carries ``"atexit": true`` so unattended
files are recognizable). Multi-config sweeps must call ``reset()`` before
and ``flush(path)`` after each config, or configs merge into one file.

Coverage and known semantics
----------------------------
* Only SC matmuls are recorded. FP16 ops — ``lm_head``, embeddings, the
  MoE router ``gate``, norms, softmax — do NOT appear; total model work is
  strictly larger than the trace's MAC sum (header carries a ``coverage``
  note).
* Rows dispatched to a drop level (``stoc_len <= 0``) are zero-filled by
  the callers without an ``sc_matmul`` call and are NOT recorded; with
  such configs (none shipped today) per-op rows no longer sum to tokens.
* Uniform (non-MP) attention runs record one 3D call with
  ``batch = B*H`` and ``unit = null``; MP attention runs record per-head
  2D calls (``unit`` = head index). Consumers must accept both flavors.
* ``d_in``/``d_out`` are *representative* (first call of the group): for
  autoregressive decode, attention's KV length grows per step, in which
  case the group is flagged ``dims_vary`` (``macs``/``rows``/``row_cycles``
  stay exact — they accumulate true per-call values).

Output schemas
--------------
summary (JSON)::

    {"schema": "scmp-trace-summary-v1",
     "header": {...static config + coverage...},
     "groups": [{"block": 12, "op": "down_proj", "unit": null,
                 "stoc_len": 128, "sc_prec": 8, "halve": true,
                 "granularity": "per_row", "mode": "bipolar",
                 "rng_levels": 128, "chunk_d": 128, "smoothed": true,
                 "d_in": 9728, "d_out": 2560, "dims_vary": false,
                 "calls": 64, "rows": 81920, "macs": 5497558138880,
                 "row_cycles": 10485760}, ...]}

trace (JSONL; header line first, then one object per call in execution
order — ``seq``)::

    {"seq": 18211, "block": 12, "op": "qk", "unit": 7, "rows": 1024,
     "d_in": 128, "d_out": 1024, "batch": 1, "stoc_len": 64, ...}

``rows`` counts a-operand rows across the batch (``batch * N`` for 3D);
``macs = batch * rows_per_slice * d_out * d_in``; ``row_cycles = rows *
stoc_len`` (per-row stream cost — the first-order SC latency/energy proxy).
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import threading
from typing import Optional

__all__ = [
    "enable", "disable", "reset", "flush",
    "set_context", "clear_context", "is_enabled",
]

# ---------------------------------------------------------------------------
# Module state. _ENABLED is read directly by hot-path guards.
# ---------------------------------------------------------------------------
_ENABLED: bool = False
_MODE: str = "summary"                     # "summary" | "trace"
_PATH: Optional[str] = None
_SEQ: int = 0
_LOCK = threading.Lock()
_TLS = threading.local()                   # per-thread (op, block, unit)
# summary accumulator:
#   key -> [calls, rows, macs, row_cycles, d_in, d_out, dims_vary]
_SUMMARY: dict = {}
_TRACE: list = []
_SPILL = 100_000                           # trace-mode records per disk spill
_FH = None                                 # open spill file handle (trace mode)


def is_enabled() -> bool:
    return _ENABLED


def enable(path: str, mode: str = "summary") -> None:
    """Turn tracing on. ``mode``: 'summary' (aggregate) or 'trace' (per-call).

    Refuses to switch mode while unflushed records exist — call ``flush()``
    or ``reset()`` first (otherwise the buffered records of the old mode
    would be silently stranded)."""
    global _ENABLED, _MODE, _PATH
    if mode not in ("summary", "trace"):
        raise ValueError(f"trace mode must be 'summary' or 'trace', got {mode!r}")
    if _SEQ and mode != _MODE:
        raise RuntimeError(
            f"trace.enable: {_SEQ} unflushed records in mode {_MODE!r}; "
            f"flush() or reset() before switching to {mode!r}.")
    _MODE = mode
    _PATH = path
    _ENABLED = True


def disable() -> None:
    global _ENABLED
    _ENABLED = False


def reset() -> None:
    """Drop all accumulated records (keeps enabled state and context)."""
    global _SEQ, _FH
    with _LOCK:
        _SUMMARY.clear()
        _TRACE.clear()
        _SEQ = 0
        if _FH is not None:
            _FH.close()
            _FH = None


def set_context(op: Optional[str], block: Optional[int],
                unit: Optional[int] = None) -> None:
    """Tag subsequent sc_matmul calls with (operator, block, sub-unit).

    ``unit`` disambiguates calls sharing (op, block): MoE expert index,
    attention head index. Thread-local, so concurrent forwards (e.g. RULER
    --threads > 1) do not cross-tag. Overwrite per call site."""
    _TLS.ctx = (op, block, unit)


def clear_context() -> None:
    set_context(None, None, None)


def _open_spill():
    """Open the trace-mode spill file and write the header line."""
    global _FH
    out = _PATH
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    _FH = open(out, "w")
    hdr = _header()
    hdr["n_records"] = None          # unknown until the run ends
    _FH.write(json.dumps({"schema": "scmp-trace-v1", "header": hdr}) + "\n")


def record_matmul(*, rows: int, d_in: int, d_out: int, batch: int,
                  stoc_len: int, sc_prec: int, mode: str, granularity: str,
                  halve: bool, rng_levels: Optional[int], chunk_d: int,
                  smoothed: bool = False) -> None:
    """Record one sc_matmul call. Called from the sc_matmul entry when enabled.

    All arguments are host-side ints/strs (shape metadata) — never tensors.
    ``rows`` is a-rows per slice; total a-rows = batch * rows.
    ``rng_levels`` must be the RESOLVED enable-grid size (never None)."""
    global _SEQ
    op, block, unit = getattr(_TLS, "ctx", (None, None, None))
    rows_total = batch * rows
    macs = batch * rows * d_out * d_in
    row_cycles = rows_total * stoc_len
    with _LOCK:
        if _MODE == "trace":
            _TRACE.append({
                "seq": _SEQ,
                "op": op, "block": block, "unit": unit,
                "rows": rows_total, "d_in": d_in, "d_out": d_out,
                "batch": batch,
                "stoc_len": stoc_len, "sc_prec": sc_prec, "halve": halve,
                "mode": mode, "granularity": granularity,
                "rng_levels": rng_levels, "chunk_d": chunk_d,
                "smoothed": smoothed,
                "macs": macs, "row_cycles": row_cycles,
            })
            # Bounded memory: spill to disk periodically so long decode runs
            # cannot OOM the host (an OOM kill would also skip atexit and
            # lose everything).
            if len(_TRACE) >= _SPILL:
                if _FH is None:
                    _open_spill()
                for rec in _TRACE:
                    _FH.write(json.dumps(rec) + "\n")
                _TRACE.clear()
        else:
            # d_in/d_out stay OUT of the key: attention dims vary per decode
            # step (KV growth), which would mint one group per step. They are
            # kept as representative payload + a dims_vary flag instead;
            # macs/rows/row_cycles accumulate exact per-call values.
            key = (block, op, unit, stoc_len, sc_prec, halve, mode,
                   granularity, rng_levels, chunk_d, smoothed)
            acc = _SUMMARY.get(key)
            if acc is None:
                _SUMMARY[key] = [1, rows_total, macs, row_cycles,
                                 d_in, d_out, False]
            else:
                acc[0] += 1
                acc[1] += rows_total
                acc[2] += macs
                acc[3] += row_cycles
                if acc[4] != d_in or acc[5] != d_out:
                    acc[6] = True
        _SEQ += 1


def _header() -> dict:
    """Static config snapshot written once per output file.

    ``scramble_masks`` is the env-knob ECHO (int when parseable), not the
    per-call resolved mask count — that is ``min(scramble_masks,
    2**sc_prec)`` using each record's own ``sc_prec``, and it only applies
    when ``owen_mode`` is ``bitrev``."""
    try:
        masks = int(os.environ.get("SC_SCRAMBLE_MASKS", "64"))
    except ValueError:
        # Garbage env value: echo it raw rather than crash flush() — the
        # kernel's own validation raises at matmul time, not here.
        masks = os.environ.get("SC_SCRAMBLE_MASKS")
    return {
        "owen_mode": os.environ.get("SC_OWEN_MODE", "bitrev"),
        "scramble_masks": masks,
        "mode": _MODE,
        "n_records": _SEQ,
        "coverage": "sc_matmul only — FP16 ops (lm_head, embeddings, MoE "
                    "router gate, norms, softmax) are NOT recorded",
    }


def flush(path: Optional[str] = None, header_extra: Optional[dict] = None,
          reset_after: bool = True) -> Optional[str]:
    """Write accumulated records; returns the output path (None if nothing).

    ``path`` overrides the enable-time path (e.g. one file per sweep
    config) — EXCEPT in trace mode once spilling has begun: the spill file
    (enable-time path) is already on disk, so the remainder is appended
    there and the override is ignored with a warning. ``header_extra``
    merges caller metadata (model id, eval tag) into the header."""
    global _FH, _SEQ
    with _LOCK:
        if _SEQ == 0:
            return None
        if _MODE == "trace" and _FH is not None:
            # Already spilling: finish the spill file.
            if path is not None and path != _PATH:
                print(f"[trace] WARN: flush path override {path!r} ignored — "
                      f"trace already spilled to {_PATH!r}.", file=sys.stderr)
            for rec in _TRACE:
                _FH.write(json.dumps(rec) + "\n")
            # The header line (written at first spill) has n_records=null —
            # the final count is only known now. ALWAYS append the trailer so
            # the file carries its count even for a bare flush(); merge any
            # caller metadata into it.
            trailer = {"trailer": True, "n_records": _SEQ}
            if header_extra:
                trailer.update(header_extra)
            _FH.write(json.dumps(trailer) + "\n")
            _FH.close()
            _FH = None
            out = _PATH
        else:
            out = path or _PATH
            if out is None:
                return None
            os.makedirs(os.path.dirname(os.path.abspath(out)) or ".",
                        exist_ok=True)
            header = _header()
            if header_extra:
                header.update(header_extra)
            if _MODE == "trace":
                with open(out, "w") as f:
                    f.write(json.dumps({"schema": "scmp-trace-v1",
                                        "header": header}) + "\n")
                    for rec in _TRACE:
                        f.write(json.dumps(rec) + "\n")
            else:
                groups = []
                for (block, op, unit, sl, prec, halve, mode, gran, rngl,
                     ckd, smoothed), (calls, rows, macs, cyc, d_in, d_out,
                                      vary) in sorted(
                        _SUMMARY.items(),
                        key=lambda kv: (kv[0][0] if kv[0][0] is not None
                                        else -1, str(kv[0][1]),
                                        str(kv[0][2]), -kv[0][3])):
                    groups.append({
                        "block": block, "op": op, "unit": unit,
                        "stoc_len": sl, "sc_prec": prec, "halve": halve,
                        "mode": mode, "granularity": gran,
                        "rng_levels": rngl, "chunk_d": ckd,
                        "smoothed": smoothed,
                        "d_in": d_in, "d_out": d_out, "dims_vary": vary,
                        "calls": calls, "rows": rows, "macs": macs,
                        "row_cycles": cyc,
                    })
                with open(out, "w") as f:
                    json.dump({"schema": "scmp-trace-summary-v1",
                               "header": header, "groups": groups}, f,
                              indent=1)
        if reset_after:
            _SUMMARY.clear()
            _TRACE.clear()
            _SEQ = 0
        return out


def _atexit_flush() -> None:
    # Safety net: apps that enable via env but never call flush() still get
    # their file at interpreter exit. The header/trailer is marked so
    # consumers can tell an unattended flush (which may span multiple
    # configs) from a deliberate per-config one.
    if _ENABLED and _SEQ:
        try:
            flush(header_extra={"atexit": True})
        except Exception:
            pass


atexit.register(_atexit_flush)

# Env-driven enable (picked up when the package is first imported). A bad
# SC_MP_TRACE_MODE must not take down the whole package import (the callers'
# try/except guards catch ImportError, not ValueError) — warn and fall back.
_env_path = os.environ.get("SC_MP_TRACE", "").strip()
if _env_path:
    _env_mode = (os.environ.get("SC_MP_TRACE_MODE", "summary").strip()
                 or "summary")
    if _env_mode not in ("summary", "trace"):
        print(f"[trace] WARN: SC_MP_TRACE_MODE={_env_mode!r} is not "
              f"'summary'|'trace'; falling back to 'summary'.",
              file=sys.stderr)
        _env_mode = "summary"
    enable(_env_path, _env_mode)
