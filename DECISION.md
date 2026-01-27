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
