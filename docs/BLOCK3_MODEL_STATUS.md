# Block 3 Model Benchmark Status

> Last Updated: 2026-02-23T14:47:32.770755+00:00
> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`

## Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| strict_condition_completion | 104/104 | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |
| v72_condition_completion | 35/104 | `docs/benchmarks/block3_truth_pack/missing_key_manifest.csv` |
| running_total | 8 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| pending_total | 12 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| eta_baseline_hours | 120.64 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |

## Progress Bars

- strict matrix: `[########################] 104/104 (100.0%)`
- v7.2 coverage: `[########----------------] 35/104 (33.7%)`

## Active Queue Groups

| group | running | pending |
|---|---:|---:|
| autofit_v72_completion | 5 | 4 |
| autofit_resubmit | 3 | 0 |
| statistical_resubmit | 0 | 0 |
| v71_g01 | 0 | 0 |
| foundation_reference | 0 | 8 |
| other_active | 0 | 0 |

## Queue Governance

- V72-first policy ledger: `docs/benchmarks/block3_truth_pack/queue_actions_latest.json`
- Actions are recommendation-only; this report does not mutate live queue state.

