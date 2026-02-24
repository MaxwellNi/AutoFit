# Block 3 Model Benchmark Status

> Last Updated: 2026-02-24T13:08:01.278591+00:00
> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`

## Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| strict_condition_completion | 104/104 | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |
| v72_condition_completion | 66/104 | `docs/benchmarks/block3_truth_pack/missing_key_manifest.csv` |
| contract_assertion | pass/fail in latest audit | `docs/benchmarks/block3_truth_pack/v72_runtime_contract_audit.json` |
| key_job_manifest_keys | see latest file | `docs/benchmarks/block3_truth_pack/v72_key_job_manifest.csv` |
| memory_plan_keys | see latest file | `docs/benchmarks/block3_truth_pack/v72_memory_plan.json` |
| running_total | 3 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| pending_total | 50 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| eta_baseline_hours | 81.23 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |

## Progress Bars

- strict matrix: `[########################] 104/104 (100.0%)`
- v7.2 coverage: `[###############---------] 66/104 (63.5%)`

## Active Queue Groups

| group | running | pending |
|---|---:|---:|
| autofit_v72_completion | 2 | 0 |
| autofit_resubmit | 1 | 0 |
| statistical_resubmit | 0 | 0 |
| v71_g01 | 0 | 0 |
| foundation_reference | 0 | 8 |
| other_active | 0 | 42 |

## Queue Governance

- V72-first policy ledger: `docs/benchmarks/block3_truth_pack/queue_actions_latest.json`
- Mandatory execution contract: `docs/BLOCK3_EXECUTION_CONTRACT.md`
- Actions are recommendation-only; this report does not mutate live queue state.

