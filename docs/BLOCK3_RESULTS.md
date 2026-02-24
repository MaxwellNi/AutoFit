# Block 3 Benchmark Results

> Last Updated: 2026-02-24T00:59:38.010347+00:00
> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`

## Strict Benchmark Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| strict_condition_completion | 104/104 | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |
| v72_condition_completion | 51/104 | `docs/benchmarks/block3_truth_pack/missing_key_manifest.csv` |
| champion_deep_classical | 62 | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |
| champion_transformer_sota | 36 | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |
| champion_foundation | 6 | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |
| champion_autofit | 0 | `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv` |
| policy_training_report | available | `docs/benchmarks/block3_truth_pack/v72_policy_training_report.json` |
| key_job_manifest | available | `docs/benchmarks/block3_truth_pack/v72_key_job_manifest.csv` |
| memory_plan | available | `docs/benchmarks/block3_truth_pack/v72_memory_plan.json` |
| running_total | 4 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| pending_total | 8 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| eta_baseline_hours | 77.37 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |

## Notes

1. Full nested task/subtask progress is maintained in `docs/benchmarks/block3_truth_pack/execution_status_latest.md`.
2. Root-cause and closure matrix are maintained in the master evidence document.
