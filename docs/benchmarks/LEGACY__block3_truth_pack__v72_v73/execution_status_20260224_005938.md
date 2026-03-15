- Snapshot UTC: **2026-02-24T00:59:38.010347+00:00**
- Queue policy: **V72-first**
- Strict matrix progress: **[########################] 104/104 (100.0%)**
- V7.2 coverage progress: **[############------------] 51/104 (49.0%)**

## Active Queue Summary

| metric | value |
|---|---:|
| running_total | 4 |
| pending_total | 8 |
| eta_optimistic_hours | 54.16 |
| eta_baseline_hours | 77.37 |
| eta_conservative_hours | 112.32 |

### Active Jobs by Group

| group | running | pending |
|---|---:|---:|
| autofit_v72_completion | 3 | 0 |
| autofit_resubmit | 1 | 0 |
| statistical_resubmit | 0 | 0 |
| v71_g01 | 0 | 0 |
| foundation_reference | 0 | 8 |
| other_active | 0 | 0 |

### Queue Bottlenecks

| reason | count |
|---|---:|
| (QOSGrpNodeLimit) | 8 |

## Nested Task Progress

### task1_outcome
- strict: [########################] 48/48 (100.0%)
- v72: [###############---------] 30/48 (62.5%)
- running_jobs: 3, pending_jobs: 1

| ablation | strict_progress | v72_progress | running_jobs | pending_jobs |
|---|---|---|---:|---:|
| core_only | [########################] 12/12 (100.0%) | [########################] 12/12 (100.0%) | 0 | 0 |
| core_text | [########################] 12/12 (100.0%) | [########----------------] 4/12 (33.3%) | 0 | 0 |
| core_edgar | [########################] 12/12 (100.0%) | [################--------] 8/12 (66.7%) | 2 | 0 |
| full | [########################] 12/12 (100.0%) | [############------------] 6/12 (50.0%) | 1 | 1 |

| target | strict_done/expected | v72_done/expected | missing_horizons |
|---|---:|---:|---|
| funding_raised_usd | 16/16 | 16/16 | - |
| investors_count | 16/16 | 10/16 | 1,7,14,30 |
| is_funded | 16/16 | 4/16 | 1,7,14,30 |

### task2_forecast
- strict: [########################] 32/32 (100.0%)
- v72: [################--------] 21/32 (65.6%)
- running_jobs: 1, pending_jobs: 4

| ablation | strict_progress | v72_progress | running_jobs | pending_jobs |
|---|---|---|---:|---:|
| core_only | [########################] 8/8 (100.0%) | [############------------] 4/8 (50.0%) | 0 | 1 |
| core_text | [########################] 8/8 (100.0%) | [############------------] 4/8 (50.0%) | 0 | 1 |
| core_edgar | [########################] 8/8 (100.0%) | [########################] 8/8 (100.0%) | 0 | 1 |
| full | [########################] 8/8 (100.0%) | [###############---------] 5/8 (62.5%) | 1 | 1 |

| target | strict_done/expected | v72_done/expected | missing_horizons |
|---|---:|---:|---|
| funding_raised_usd | 16/16 | 16/16 | - |
| investors_count | 16/16 | 5/16 | 1,7,14,30 |

### task3_risk_adjust
- strict: [########################] 24/24 (100.0%)
- v72: [------------------------] 0/24 (0.0%)
- running_jobs: 0, pending_jobs: 3

| ablation | strict_progress | v72_progress | running_jobs | pending_jobs |
|---|---|---|---:|---:|
| core_only | [########################] 8/8 (100.0%) | [------------------------] 0/8 (0.0%) | 0 | 1 |
| core_edgar | [########################] 8/8 (100.0%) | [------------------------] 0/8 (0.0%) | 0 | 1 |
| full | [########################] 8/8 (100.0%) | [------------------------] 0/8 (0.0%) | 0 | 1 |

| target | strict_done/expected | v72_done/expected | missing_horizons |
|---|---:|---:|---|
| funding_raised_usd | 12/12 | 0/12 | 1,7,14,30 |
| investors_count | 12/12 | 0/12 | 1,7,14,30 |

## V7.2 Completed Subtasks (by task/ablation/target)

| task | ablation | target | completed_horizons |
|---|---|---|---|
| task1_outcome | core_edgar | funding_raised_usd | 1,7,14,30 |
| task1_outcome | core_edgar | investors_count | 1,7,14,30 |
| task1_outcome | core_only | funding_raised_usd | 1,7,14,30 |
| task1_outcome | core_only | investors_count | 1,7,14,30 |
| task1_outcome | core_only | is_funded | 1,7,14,30 |
| task1_outcome | core_text | funding_raised_usd | 1,7,14,30 |
| task1_outcome | full | funding_raised_usd | 1,7,14,30 |
| task1_outcome | full | investors_count | 1,7 |
| task2_forecast | core_edgar | funding_raised_usd | 1,7,14,30 |
| task2_forecast | core_edgar | investors_count | 1,7,14,30 |
| task2_forecast | core_only | funding_raised_usd | 1,7,14,30 |
| task2_forecast | core_text | funding_raised_usd | 1,7,14,30 |
| task2_forecast | full | funding_raised_usd | 1,7,14,30 |
| task2_forecast | full | investors_count | 1 |

## V7.2 Missing Subtasks (by task/ablation/target)

| task | ablation | target | missing_horizons |
|---|---|---|---|
| task1_outcome | core_edgar | is_funded | 1,7,14,30 |
| task1_outcome | core_text | investors_count | 1,7,14,30 |
| task1_outcome | core_text | is_funded | 1,7,14,30 |
| task1_outcome | full | investors_count | 14,30 |
| task1_outcome | full | is_funded | 1,7,14,30 |
| task2_forecast | core_only | investors_count | 1,7,14,30 |
| task2_forecast | core_text | investors_count | 1,7,14,30 |
| task2_forecast | full | investors_count | 7,14,30 |
| task3_risk_adjust | core_edgar | funding_raised_usd | 1,7,14,30 |
| task3_risk_adjust | core_edgar | investors_count | 1,7,14,30 |
| task3_risk_adjust | core_only | funding_raised_usd | 1,7,14,30 |
| task3_risk_adjust | core_only | investors_count | 1,7,14,30 |
| task3_risk_adjust | full | funding_raised_usd | 1,7,14,30 |
| task3_risk_adjust | full | investors_count | 1,7,14,30 |
