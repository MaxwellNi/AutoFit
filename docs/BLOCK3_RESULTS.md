# Block 3 Benchmark Results

> Last Updated: 2026-02-22 00:56:17 UTC
> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`

## Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| strict_records | 5109 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_completion_ratio | 1.0 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| v71_win_rate_vs_v7 | 0.44 | `docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv` |
| v71_median_relative_gain_vs_v7_pct | -0.33387569285466684 | `docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv` |
| v72_rows_raw | 24 | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| v72_overlap_keys_v7_v72_non_autofit | 24 | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| v72_vs_v7_win_rate_pct | 50.0 | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| v72_global_normalized_mae_improvement_pct_vs_v7 | 1.4909248231112944 | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| v72_investors_count_gap_reduction_pct_vs_v7 | -2.917353146000108 | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| v72_overall_pass | false | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| autofit_global_champion_keys | 0/104 | `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv` |
| critical_failures | 4 | `docs/benchmarks/block3_truth_pack/failure_taxonomy.csv` |
| running_total | 8 | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |
| pending_total | 76 | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |
| v72_local_4090_completed_pairs | 20/20 | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |
| v72_local_4090_completion_ratio | 1.0 | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |
| v72_local_4090_fairness_all_pass | true | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |

## Notes

1. Use the master evidence document for full task/subtask, ladder, and SOTA analysis tables.
2. Strict leaderboard remains dominated by deep_classical and transformer_sota under fairness filtering.
3. This file keeps only high-level snapshot metrics.
