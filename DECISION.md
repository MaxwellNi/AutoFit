# DECISION

> Reconstructed after workspace reset. Historical decisions may be missing.

## Recovery note

- This file was recreated to unblock B11 v2 policy work.
- New decisions will be appended below.

## Strict-future policy upgrade status

- Reconstructed minimal pipeline to proceed without original repo files.
- With `min_label_delta_days=1.0` and `min_ratio_delta_rel=1e-4`, **no samples survive** for `goal_min=50, limit_rows=500k, h14` (all windows dropped by static ratio or delta-days).
- Next step requires either loosening thresholds or adopting time-based horizon selection to guarantee >=1 day gaps.

## Historical horizon learnability (from transcript)

- h1 fix2_sf full grid completed; baseline still near-perfect in corner cases.
- h3 full grid: near-perfect rate ~0.25 persisted.
- h7 full grid: near-perfect persisted in goal=500, limit_rows=500k corner.
- h14 minimal passed, but h14 full grid failed in goal=50, 500k due to `y≈ratio_from_last`.
- Conclusion: strict_future policy upgrade is required before rerunning full grids; horizon alone is insufficient.

## Recovery decision (20260127)

- Proceed with audit-first recovery (snapshots + checksums + smoke run).
- Do not restart full B11 v2 grids until strict_future is implemented and smoke passes.

## Phase0/1 status (20260127_032955)

- `runs/offers_core_v2_20260125_214521/offers_core.parquet` is missing.
- Strict_future AB wiring check could not run; Phase1 halted.
- Next required step: restore the real offers_core parquet before any AB/repro runs.

## strict_future wiring status (20260127_034904)

- SMOKE AB shows strict_future changes sample counts; smoke is wiring-only, not a mainline conclusion.
- Real offers_core_v2 rebuilt from `data/raw/offers` (200k rows).
- Real AB: `strict_future=1` with `min_label_delta_days=1.0` + `min_ratio_delta_rel=1e-4` produced **no samples**; AB incomplete.
- Real sf0 alignment still shows `y≈ratio_from_last (corr>0.9999)`.
- Conclusion: strict_future wiring is confirmed on smoke, but real AB is blocked; adjust thresholds or time-aligned selection before any grid.

## Phase3 real AB unlock (20260127_045144)

- Feasibility on rebuilt offers_core_v2 recommends `label_horizon=45` (pct_delta_days_lt_min<=0.05 rule).
- Real AB (sf0 vs sf1) completed on rebuilt offers_core_v2 with `limit_rows=100k` using entity-subset sampling.
- Alignment conclusion for both runs: `not_simple_y_eq_last_ratio` with `median_delta_days≈1.9167`, leakage_flag=false.
- strict_future produced measurable differences in counts (sf0 vs sf1) without STRONG WARNING.
- Decision: strict_future AB is unblocked on real data; next step can proceed after reviewing RESULT_AB_REAL and gates.

## Gate B sampling consistency (20260127_090025)

- Leakage audit now mirrors benchmark entity sampling (entity-subset after filter).
- Re-run leakage on sf0/sf1 produced non-empty y_stats (n_finite > 0).
