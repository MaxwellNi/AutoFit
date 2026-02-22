# Block 3 Model Benchmark Status

> Last Updated: 2026-02-22 00:56:17 UTC
> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`

## Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| raw_records | 14491 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_records | 5109 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| legacy_unverified_records | 9382 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_condition_completion | 104/104 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| v72_rows_raw | 24 | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| v72_overlap_keys_v7_v72_non_autofit | 24 | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| v72_global_normalized_mae_improvement_pct_vs_v7 | 1.4909248231112944 | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| v72_investors_count_gap_reduction_pct_vs_v7 | -2.917353146000108 | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| champion_distribution | deep_classical=62, transformer_sota=36, foundation=6, autofit=0 | `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv` |
| running_total | 8 | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |
| pending_total | 76 | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |
| v72_local_4090_task1_core_only | 12/12 completed | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |
| v72_local_4090_task2_core_edgar | 8/8 completed | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |
| v72_local_4090_overall | 20/20 completed | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |

## Notes

1. Detailed tables are centralized in the master evidence document.
2. Synced outputs from `dual3090_phase7` and `phase7_v72_4090` are included in strict-comparable aggregation.
3. This file is intentionally lightweight to avoid drift.
