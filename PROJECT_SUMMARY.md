# Project Summary

## Audit summary (20260127_032955)

- Phase0 preflight completed; logs + manifest written under `runs/sanity_20260127_032955/`.
- Offers_core probe failed: `runs/offers_core_v2_20260125_214521/offers_core.parquet` missing.
- Phase1 strict_future AB not executed; Phase2/3 blocked pending data restore.

## Audit summary (20260127_034904)

- Smoke AB verifies strict_future wiring; results recorded in `runs/sanity_20260127_034904/RESULT_AB_SMOKE.md`.
- Real offers_core_v2 rebuilt from `data/raw/offers`.
- Real AB failed for strict_future=1 due to zero samples; cannot proceed to grid.

## Audit summary (20260127_045144)

- Feasibility on rebuilt offers_core_v2 recommends horizon=45 (pct_delta_days_lt_min<=0.05).
- Real AB (sf0/sf1) completed with non-zero samples; Gate D conclusion is not_simple_y_eq_last_ratio.
