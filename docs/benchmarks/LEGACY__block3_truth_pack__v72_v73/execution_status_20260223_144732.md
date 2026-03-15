- Snapshot UTC: **2026-02-23T14:47:32.770755+00:00**
- Queue policy: **V72-first**
- Strict matrix progress: **[########################] 104/104 (100.0%)**
- V7.2 coverage progress: **[########----------------] 35/104 (33.7%)**

## Active Queue Summary

| metric | value |
|---|---:|
| running_total | 8 |
| pending_total | 12 |
| eta_optimistic_hours | 84.45 |
| eta_baseline_hours | 120.64 |
| eta_conservative_hours | 172.9 |

### Active Jobs by Group

| group | running | pending |
|---|---:|---:|
| autofit_v72_completion | 5 | 4 |
| autofit_resubmit | 3 | 0 |
| statistical_resubmit | 0 | 0 |
| v71_g01 | 0 | 0 |
| foundation_reference | 0 | 8 |
| other_active | 0 | 0 |

### Queue Bottlenecks

| reason | count |
|---|---:|
| (QOSGrpNodeLimit) | 8 |
| (QOSMaxJobsPerUserLimit) | 4 |

## Nested Task Progress

### task1_outcome
- strict: [########################] 48/48 (100.0%)
- v72: [############------------] 25/48 (52.1%)
- running_jobs: 6, pending_jobs: 1

| ablation | strict_progress | v72_progress | running_jobs | pending_jobs |
|---|---|---|---:|---:|
| core_only | [########################] 12/12 (100.0%) | [########################] 12/12 (100.0%) | 0 | 0 |
| core_text | [########################] 12/12 (100.0%) | [######------------------] 3/12 (25.0%) | 1 | 0 |
| core_edgar | [########################] 12/12 (100.0%) | [##############----------] 7/12 (58.3%) | 3 | 0 |
| full | [########################] 12/12 (100.0%) | [######------------------] 3/12 (25.0%) | 2 | 1 |

| target | strict_done/expected | v72_done/expected | missing_horizons |
|---|---:|---:|---|
| funding_raised_usd | 16/16 | 13/16 | 30 |
| investors_count | 16/16 | 8/16 | 1,7,14,30 |
| is_funded | 16/16 | 4/16 | 1,7,14,30 |

### task2_forecast
- strict: [########################] 32/32 (100.0%)
- v72: [########----------------] 10/32 (31.2%)
- running_jobs: 2, pending_jobs: 5

| ablation | strict_progress | v72_progress | running_jobs | pending_jobs |
|---|---|---|---:|---:|
| core_only | [########################] 8/8 (100.0%) | [###---------------------] 1/8 (12.5%) | 1 | 1 |
| core_text | [########################] 8/8 (100.0%) | [###---------------------] 1/8 (12.5%) | 1 | 1 |
| core_edgar | [########################] 8/8 (100.0%) | [########################] 8/8 (100.0%) | 0 | 1 |
| full | [########################] 8/8 (100.0%) | [------------------------] 0/8 (0.0%) | 0 | 2 |

| target | strict_done/expected | v72_done/expected | missing_horizons |
|---|---:|---:|---|
| funding_raised_usd | 16/16 | 6/16 | 1,7,14,30 |
| investors_count | 16/16 | 4/16 | 1,7,14,30 |

### task3_risk_adjust
- strict: [########################] 24/24 (100.0%)
- v72: [------------------------] 0/24 (0.0%)
- running_jobs: 0, pending_jobs: 6

| ablation | strict_progress | v72_progress | running_jobs | pending_jobs |
|---|---|---|---:|---:|
| core_only | [########################] 8/8 (100.0%) | [------------------------] 0/8 (0.0%) | 0 | 2 |
| core_edgar | [########################] 8/8 (100.0%) | [------------------------] 0/8 (0.0%) | 0 | 2 |
| full | [########################] 8/8 (100.0%) | [------------------------] 0/8 (0.0%) | 0 | 2 |

| target | strict_done/expected | v72_done/expected | missing_horizons |
|---|---:|---:|---|
| funding_raised_usd | 12/12 | 0/12 | 1,7,14,30 |
| investors_count | 12/12 | 0/12 | 1,7,14,30 |

## V7.2 Completed Subtasks (by task/ablation/target)

| task | ablation | target | completed_horizons |
|---|---|---|---|
| task1_outcome | core_edgar | funding_raised_usd | 1,7,14 |
| task1_outcome | core_edgar | investors_count | 1,7,14,30 |
| task1_outcome | core_only | funding_raised_usd | 1,7,14,30 |
| task1_outcome | core_only | investors_count | 1,7,14,30 |
| task1_outcome | core_only | is_funded | 1,7,14,30 |
| task1_outcome | core_text | funding_raised_usd | 1,7,14 |
| task1_outcome | full | funding_raised_usd | 1,7,14 |
| task2_forecast | core_edgar | funding_raised_usd | 1,7,14,30 |
| task2_forecast | core_edgar | investors_count | 1,7,14,30 |
| task2_forecast | core_only | funding_raised_usd | 1 |
| task2_forecast | core_text | funding_raised_usd | 1 |

## V7.2 Missing Subtasks (by task/ablation/target)

| task | ablation | target | missing_horizons |
|---|---|---|---|
| task1_outcome | core_edgar | funding_raised_usd | 30 |
| task1_outcome | core_edgar | is_funded | 1,7,14,30 |
| task1_outcome | core_text | funding_raised_usd | 30 |
| task1_outcome | core_text | investors_count | 1,7,14,30 |
| task1_outcome | core_text | is_funded | 1,7,14,30 |
| task1_outcome | full | funding_raised_usd | 30 |
| task1_outcome | full | investors_count | 1,7,14,30 |
| task1_outcome | full | is_funded | 1,7,14,30 |
| task2_forecast | core_only | funding_raised_usd | 7,14,30 |
| task2_forecast | core_only | investors_count | 1,7,14,30 |
| task2_forecast | core_text | funding_raised_usd | 7,14,30 |
| task2_forecast | core_text | investors_count | 1,7,14,30 |
| task2_forecast | full | funding_raised_usd | 1,7,14,30 |
| task2_forecast | full | investors_count | 1,7,14,30 |
| task3_risk_adjust | core_edgar | funding_raised_usd | 1,7,14,30 |
| task3_risk_adjust | core_edgar | investors_count | 1,7,14,30 |
| task3_risk_adjust | core_only | funding_raised_usd | 1,7,14,30 |
| task3_risk_adjust | core_only | investors_count | 1,7,14,30 |
| task3_risk_adjust | full | funding_raised_usd | 1,7,14,30 |
| task3_risk_adjust | full | investors_count | 1,7,14,30 |
