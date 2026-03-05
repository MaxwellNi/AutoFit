# Block 3 Model Benchmark Status

> Last Updated: 2026-03-05
> Full results: `docs/BLOCK3_RESULTS.md`
> V7.3 execution spec: `docs/BLOCK3_V73_RESEARCH_EXECUTION_SPEC_20260225.md`

## Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| condition_coverage | 48/48 | 3 targets x 4 horizons x 4 ablations |
| unique_models_evaluated | 90 | across 7 categories |
| unique_metric_records | 3,427 | deduplicated from 424 metrics.json files |
| top_5 | NBEATS, PatchTST, NHITS, AutoFitV734, AutoFitV733 | by avg per-condition RMSE rank |
| champion_distribution | NBEATS(14), TimesNet(12), DeepNPTS(6), V733(4), V734(2), DeepAR(2), FusedChamp(2), NBEATSx(2), PatchTST(2), Autoformer(1), NHITS(1) | 48 condition winners |

## Category Status

| Category | Models | Eval'd | core_only | core_text | core_edgar | full |
|---|---:|---:|---|---|---|---|
| ml_tabular | 15 | 15 | done | done | rerunning | rerunning |
| statistical | 5 | 5 | done | done | rerunning | rerunning |
| deep_classical | 4 | 4 | done | done | rerunning | rerunning |
| transformer_sota | 23+ | 27 | done | done | rerunning | rerunning |
| foundation | 11 | 11 | done | done | rerunning | rerunning |
| irregular | 2 | 2 | done | done | rerunning | rerunning |
| autofit | 17 | 17 | done | done | done* | done* |

*AutoFit core_edgar/full were run before EDGAR timezone fix; V735 depends on oracle refresh from rerun data.

## AutoFit Progression

| Version | Rank | Avg RMSE Rank | Champ |
|---|---:|---:|---:|
| V1 | 51/90 | 47.48 | 0 |
| V2 | 35/90 | 36.55 | 0 |
| V3 | 44/90 | 43.32 | 0 |
| V4 | 38/90 | 39.98 | 0 |
| V5 | 50/90 | 46.28 | 0 |
| V6 | 59/90 | 49.50 | 0 |
| V7 | 54/90 | 48.27 | 0 |
| V71 | 52/90 | 47.50 | 0 |
| V72 | 56/90 | 48.44 | 0 |
| V733 | **5/90** | **9.79** | **4** |
| V734 | **4/90** | **8.25** | **2** |
| V735 | pending | -- | -- |

## Active Job Queue (as of 2026-03-05)

### npin account (GPU)

Baseline reruns with EDGAR timezone fix and deterministic NF seeding:

- transformer_sota core_edgar/full: 6 jobs (RUNNING/PENDING)
- irregular core_edgar/full: 2 jobs (RUNNING)
- oracle_refresh: 1 job (PENDING, depends on all baseline reruns)
- V735 evaluation: 6 jobs (PENDING, depends on oracle_refresh)

### cfisch account (GPU + bigmem)

- deep_classical core_edgar/full: 4 jobs (RUNNING/PENDING)
- foundation core_edgar/full: 6 jobs (PENDING)
- V735 core_only/core_text: 6 jobs (PENDING)
- ml_tabular core_edgar/full: 4 jobs (bigmem, PENDING)
- statistical core_edgar/full: 6 jobs (bigmem, PENDING)

## Notes

1. Horizons used in this benchmark are {1, 7, 14, 30} days, not {7, 14, 30, 60}.
2. core_edgar and full ablation baselines are being re-evaluated after EDGAR timezone fix (commit `ae9626b`). The oracle table for V735 will be refreshed once these reruns complete.
3. V734 broke into overall top-5 and is the strongest AutoFit variant to date. V735 (exact per-condition oracle) is expected to match or exceed the standalone champion for each condition.
