# Task: generalize MP threshold calibration into `scmp_kernels` (multi-architecture)

**Status:** design / not started — handoff for a group member.
**Owner:** _unassigned_
**Related:** `scmp_diffusion` PR #7 (FLOP-weighted budget + per-row qk, diffusion only).

## Goal

There are **two copies** of the mixed-precision threshold calibrator that have
diverged:

- `scmp_diffusion/scripts/calibrate_mp_thresholds.py` (DiT)
- `scmp_llm_llama/benchmark/ppl/calibrate_mp_thresholds.py` (Llama / Qwen / MoE)

They share the same lineage (`ThresholdCalibrator`, `_global_lambda`,
`_cost_assignments`, `_thresholds_from_counts`, `_normalize_metric`,
`_relative_l2_*`, `_bucket_index`, `_flatten_summary`) but each grew fixes/features
the other lacks, so a fix in one does **not** reach the other (e.g. the FLOP-weighted
budget from PR #7 only landed in diffusion).

**Move the shared calibration core into `scmp_kernels/mp/calibration.py`** (new
module, sibling of the existing `auto_calibrator.py`), architecture-agnostic, so both
apps import it. **Leave `scmp_diffusion/scripts/calibrate_mp_thresholds.py` in place
as a backup** (do not delete). Add LLM support via a thin adapter rather than a fork.

## Key finding: base the shared core on the **llama** calibrator, not diffusion's

`scmp_llm_llama`'s calibrator is the **superset** — it already has two of the three
"fixes" and several features diffusion's lacks. See `scmp_llm/CLAUDE.md`
("Two bugs that gated cross-layer (fixed — don't reintroduce)"):

| capability | diffusion calib | llama calib |
|---|---|---|
| per-row `qk` (not per-head) | yes (PR #7) | yes (already) |
| `rep_g = R_g/n_g` subsample repricing | yes | yes (already) |
| **FLOP / `macs_per_row` weighting** | yes (PR #7) | **NO — missing** |
| `W_g` per-group importance weights | no | yes (`group_weights`) |
| `grad` / `measured` / `act_global` methods | no | yes |
| MoE empty-expert handling | no | yes |
| `PendingMerger` (grad, keyed by call id) | no | yes |

So the **shared `ThresholdCalibrator` = llama's core + the `macs`/FLOP weighting from
PR #7**. Everything else llama already has; diffusion contributes only the FLOP fix.

### Why FLOP weighting matters (the one thing llama is still missing)

The budget is currently **row-weighted** (`cost = Σ rows · stoc_len`). Attention
(`qk`/`av`) is counted per-head-per-row — ~80% of rows but only ~2% of FLOPs — so a
row-weighted `avg48` table is really ~`avg70` in real compute, and the allocation
over-invests precision in the FLOP-cheap attention. Fix: `cost = Σ rows · macs · stoc_len`,
where `macs_per_row = out×in` (linears) or `N×head_dim` (qk/av). See PR #7 for the exact
diffusion implementation (`_global_lambda` receives `R·macs` counts; `_fit_group` `rep`
is multiplied by `macs`; the iso-budget `expected_avg_stoc_len` is FLOP-weighted).

## Proposed design

### 1. `scmp_kernels/mp/calibration.py` (new, generic)
Port from **llama** `benchmark/ppl/calibrate_mp_thresholds.py`, adding `macs`:

- `ThresholdCalibrator` — union of llama features + FLOP `macs`:
  - `add(operator, block_idx, metric_norm, errors_by_level, *, macs_per_row=1.0,
    grad_rows=None, timestep=0)` — records per-row metric + per-level SC error, the
    true row count `R_g`, **and** `op_macs[operator]`.
  - `export()` — global solve weights each group by `R_g · macs_g` (not `R_g`);
    `_fit_group` `rep = (R_g/n_g)·macs_g`; `expected_avg_stoc_len` is FLOP-weighted.
  - keep `group_weights` (`W_g`), `loss_weight_by_grad`/`grad_as_group_weight`,
    `budget_scope ∈ {per_bucket, global}`, `timestep_buckets`/`layer_buckets`.
- pure solver: `_cost_assignments`, `_global_lambda` (macs-weighted), `_thresholds_from_counts`.
- metric fns: `_normalize_metric` (MoE empty-input safe), `_relative_l2_rows/_heads`,
  `_cosine_dist_*`, `_bucket_index`.
- summary: `_flatten_summary`, `_write_summary_csv`.
- Operators are a **passed-in set** (already the case); diffusion timesteps degrade to
  `T=1` for LLMs — the core is arch-agnostic today.
- Export the public names from `scmp_kernels/mp/__init__.py`.

### 2. Arch-specific runners = thin adapters (stay in each app repo)
The seam is `calibrator.add(...)`. Each app keeps a small runner that walks the model
and feeds per-operator `(metric, level_errors, macs_per_row[, grad_rows])`:
- **DiT** (`scmp_diffusion`): hooks on `block.attn`/`block.mlp` → `qk, av, proj,
  mlp_fc1, mlp_fc2, input_proj`. Reference: the current `CalibrationRunner` +
  `_run_attention_linear_level` / `_run_mlp_linear_level` / `_build_input_proj_output`.
- **LLM** (`scmp_llm_llama`): hooks on `self_attn`/`mlp` → `q/k/v/o_proj,
  gate/up/down_proj, qk, av`; MoE `gate` skipped. Reference: `_make_sclinear_hook`,
  `_patch_attention_for_calibration`.

Optionally also move the per-level SC-error probes (`sc_linear_error_at_levels`,
`sc_bmm_error_at_levels`) into `calibration.py` so adding a new architecture is just
"list operators + point at weights". This is the bigger-reuse option; the minimal
version keeps the probes in each app.

### 3. Adoption
- `scmp_llm_llama` and `scmp_diffusion` import `ThresholdCalibrator` (+ helpers) from
  `scmp_kernels.mp` instead of their local copies. **Llama inherits the FLOP fix for
  free.** Keep the diffusion script as a backup entry point.

## Gotchas / must-not-reintroduce (from `scmp_llm/CLAUDE.md`)
- **`rep_g` pricing** and **per-row qk** are load-bearing for the global/cross-layer
  budget — preserve them.
- **MoE empty expert**: an expert can get a `(0, D)` input; `_normalize_metric` must
  handle empty, and the SCLinear hook must skip `x_flat.shape[0] == 0`.
- **`PendingMerger`** must key its grad accumulator by **call id** (`cid`), not
  `(operator, block_idx)` — MoE experts share op+layer with different token counts.
- **Owen scramble must match** between calibration and eval (`SC_OWEN_MODE`,
  `SC_SCRAMBLE_MASKS`); tables don't transfer across mask counts. `--recalibrate` after
  any change to the calibrator, Owen mode, or mask count.
- Each table records `expected_avg_stoc_len` + `global_lambda`; with the macs change,
  `expected_avg_stoc_len` is the **FLOP-weighted** realized budget — confirm it lands
  near target before trusting PPL/FID.

## Concrete steps
1. Create `scmp_kernels/mp/calibration.py` = llama core + `macs` (per PR #7). Export from `mp/__init__.py`.
2. Unit-smoke the solver: replicate PR #7's `_global_lambda` FLOP check (realized FLOP-weighted avg ≈ target; row-weighted ≈ old).
3. Rewire `scmp_llm_llama` calibrator to import the shared core; keep its arch hooks. Re-run one llama/qwen table; confirm `expected_avg_stoc_len` ≈ target (FLOP-weighted) and PPL vs the row-weighted baseline.
4. (Optional) Rewire diffusion's runner onto the shared core; keep the current script as backup.
5. (Optional) Move the SC-error probes into `calibration.py` and reduce each app runner to operator-enumeration.

## Verification
Needs a GPU (`gl-gpu`). Iso-budget: calibrate one budget, confirm `expected_avg_stoc_len`
≈ target in FLOP terms; then one gen/PPL run to confirm the repaired allocation. The
pure solver can be smoke-tested on CPU (see PR #7's standalone `_global_lambda` replication).

## References
- `scmp_diffusion` PR #7 — FLOP-weighted budget + per-row qk (diffusion). The macs
  threading + `_global_lambda`/`_fit_group`/iso-check edits are the template.
- `scmp_llm_llama/benchmark/ppl/calibrate_mp_thresholds.py` — the superset base to port.
- `scmp_llm/CLAUDE.md` §"Mixed-precision (MP) calibration" — methods, W_g, MoE, scramble.
- `scmp_kernels/mp/auto_calibrator.py` — the *other* MP method (free-boundary oracle);
  not part of this task, listed to avoid confusion.
