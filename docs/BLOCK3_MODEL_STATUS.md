# Block 3 Model Benchmark Status

> Last updated: 2026-03-06
> Full results: `docs/BLOCK3_RESULTS.md`

## Snapshot

| Metric | Value |
|---|---:|
| Evaluation conditions | 48 (3 targets x 4 horizons x 4 ablations) |
| Models evaluated | 91 |
| Deduplicated records | 3,421 (from 259 metrics.json files) |
| Top 5 | NBEATS, AutoFitV734, AutoFitV735, NHITS, AutoFitV733 |
| Champion distribution | V733(12), TimesNet(12), NBEATS(8), DeepNPTS(8), NBEATSx(4), V734(2), Autoformer(1), DeepAR(1) |

## Category Status

All categories complete across all 4 ablations. No pending jobs.

| Category | Models | core_only | core_text | core_edgar | full |
|---|---:|---|---|---|---|
| ml_tabular | 15 | done | done | done | done |
| statistical | 5 | done | done | done | done |
| deep_classical | 5 | done | done | done | done |
| transformer_sota | 30+ | done | done | done | done |
| foundation | 10 | done | done | done | done |
| irregular | 2 | done | done | done | done |
| autofit | 18 | done | done | done | done |

## AutoFit Progression

| Version | Rank | Avg RMSE Rank | Conds | Champ |
|---|---:|---:|---:|---:|
| V1 | 56/91 | 48.68 | 44 | 0 |
| V2 | 36/91 | 37.57 | 44 | 0 |
| V3 | 48/91 | 45.17 | 41 | 0 |
| V4 | 39/91 | 40.76 | 46 | 0 |
| V5 | 52/91 | 47.13 | 46 | 0 |
| V6 | 55/91 | 48.57 | 46 | 0 |
| V7 | 61/91 | 50.09 | 46 | 0 |
| V72 | 60/91 | 49.60 | 40 | 0 |
| V733 | **5/91** | **9.51** | **47** | **12** |
| V734 | **2/91** | **7.92** | **48** | **2** |
| V735 | **3/91** | **8.38** | **48** | **0** |

## Notes

1. Horizons: {1, 7, 14, 30} days.
2. All baseline reruns with EDGAR timezone fix (commit `ae9626b`) and deterministic NF seeding (commit `d388310`) are complete.
3. V734 ranks #2 overall. V735 ranks #3. V733 has the most champion wins (12) despite ranking #5.
4. See `docs/BLOCK3_RESULTS.md` for the full 91-model leaderboard, all 48 condition champions, and V734 vs V735 head-to-head table.
