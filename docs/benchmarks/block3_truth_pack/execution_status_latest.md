- Snapshot UTC: **2026-02-24T13:08:01.278591+00:00**
- Queue policy: **V72-first**
- Strict matrix progress: **[########################] 104/104 (100.0%)**
- V7.2 coverage progress: **[###############---------] 66/104 (63.5%)**

## Active Queue Summary

| metric | value |
|---|---:|
| running_total | 3 |
| pending_total | 50 |
| eta_optimistic_hours | 56.86 |
| eta_baseline_hours | 81.23 |
| eta_conservative_hours | 117.72 |

### Active Jobs by Group

| group | running | pending |
|---|---:|---:|
| autofit_v72_completion | 2 | 0 |
| autofit_resubmit | 1 | 0 |
| statistical_resubmit | 0 | 0 |
| v71_g01 | 0 | 0 |
| foundation_reference | 0 | 8 |
| other_active | 0 | 42 |

### Queue Bottlenecks

| reason | count |
|---|---:|
| (QOSGrpNodeLimit) | 47 |
| (Priority) | 2 |
| (Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions) | 1 |

## Nested Task Progress

### task1_outcome
- strict: [########################] 48/48 (100.0%)
- v72: [#####################---] 42/48 (87.5%)
- running_jobs: 3, pending_jobs: 11

| ablation | strict_progress | v72_progress | running_jobs | pending_jobs |
|---|---|---|---:|---:|
| core_only | [########################] 12/12 (100.0%) | [########################] 12/12 (100.0%) | 0 | 0 |
| core_text | [########################] 12/12 (100.0%) | [################--------] 8/12 (66.7%) | 0 | 8 |
| core_edgar | [########################] 12/12 (100.0%) | [######################--] 11/12 (91.7%) | 2 | 1 |
| full | [########################] 12/12 (100.0%) | [######################--] 11/12 (91.7%) | 1 | 2 |

| target | strict_done/expected | v72_done/expected | missing_horizons |
|---|---:|---:|---|
| funding_raised_usd | 16/16 | 16/16 | - |
| investors_count | 16/16 | 12/16 | 1,7,14,30 |
| is_funded | 16/16 | 14/16 | 30 |

### task2_forecast
- strict: [########################] 32/32 (100.0%)
- v72: [##################------] 24/32 (75.0%)
- running_jobs: 0, pending_jobs: 12

| ablation | strict_progress | v72_progress | running_jobs | pending_jobs |
|---|---|---|---:|---:|
| core_only | [########################] 8/8 (100.0%) | [############------------] 4/8 (50.0%) | 0 | 5 |
| core_text | [########################] 8/8 (100.0%) | [############------------] 4/8 (50.0%) | 0 | 5 |
| core_edgar | [########################] 8/8 (100.0%) | [########################] 8/8 (100.0%) | 0 | 1 |
| full | [########################] 8/8 (100.0%) | [########################] 8/8 (100.0%) | 0 | 1 |

| target | strict_done/expected | v72_done/expected | missing_horizons |
|---|---:|---:|---|
| funding_raised_usd | 16/16 | 16/16 | - |
| investors_count | 16/16 | 8/16 | 1,7,14,30 |

### task3_risk_adjust
- strict: [########################] 24/24 (100.0%)
- v72: [------------------------] 0/24 (0.0%)
- running_jobs: 0, pending_jobs: 27

| ablation | strict_progress | v72_progress | running_jobs | pending_jobs |
|---|---|---|---:|---:|
| core_only | [########################] 8/8 (100.0%) | [------------------------] 0/8 (0.0%) | 0 | 9 |
| core_edgar | [########################] 8/8 (100.0%) | [------------------------] 0/8 (0.0%) | 0 | 9 |
| full | [########################] 8/8 (100.0%) | [------------------------] 0/8 (0.0%) | 0 | 9 |

| target | strict_done/expected | v72_done/expected | missing_horizons |
|---|---:|---:|---|
| funding_raised_usd | 12/12 | 0/12 | 1,7,14,30 |
| investors_count | 12/12 | 0/12 | 1,7,14,30 |

## V7.2 Completed Subtasks (by task/ablation/target)

| task | ablation | target | completed_horizons |
|---|---|---|---|
| task1_outcome | core_edgar | funding_raised_usd | 1,7,14,30 |
| task1_outcome | core_edgar | investors_count | 1,7,14,30 |
| task1_outcome | core_edgar | is_funded | 1,7,14 |
| task1_outcome | core_only | funding_raised_usd | 1,7,14,30 |
| task1_outcome | core_only | investors_count | 1,7,14,30 |
| task1_outcome | core_only | is_funded | 1,7,14,30 |
| task1_outcome | core_text | funding_raised_usd | 1,7,14,30 |
| task1_outcome | core_text | is_funded | 1,7,14,30 |
| task1_outcome | full | funding_raised_usd | 1,7,14,30 |
| task1_outcome | full | investors_count | 1,7,14,30 |
| task1_outcome | full | is_funded | 1,7,14 |
| task2_forecast | core_edgar | funding_raised_usd | 1,7,14,30 |
| task2_forecast | core_edgar | investors_count | 1,7,14,30 |
| task2_forecast | core_only | funding_raised_usd | 1,7,14,30 |
| task2_forecast | core_text | funding_raised_usd | 1,7,14,30 |
| task2_forecast | full | funding_raised_usd | 1,7,14,30 |
| task2_forecast | full | investors_count | 1,7,14,30 |

## V7.2 Missing Subtasks (by task/ablation/target)

| task | ablation | target | missing_horizons |
|---|---|---|---|
| task1_outcome | core_edgar | is_funded | 30 |
| task1_outcome | core_text | investors_count | 1,7,14,30 |
| task1_outcome | full | is_funded | 30 |
| task2_forecast | core_only | investors_count | 1,7,14,30 |
| task2_forecast | core_text | investors_count | 1,7,14,30 |
| task3_risk_adjust | core_edgar | funding_raised_usd | 1,7,14,30 |
| task3_risk_adjust | core_edgar | investors_count | 1,7,14,30 |
| task3_risk_adjust | core_only | funding_raised_usd | 1,7,14,30 |
| task3_risk_adjust | core_only | investors_count | 1,7,14,30 |
| task3_risk_adjust | full | funding_raised_usd | 1,7,14,30 |
| task3_risk_adjust | full | investors_count | 1,7,14,30 |
