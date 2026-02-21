# Block 3 Benchmark Results

> Last Updated: 2026-02-21 22:55:00 UTC
> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`
> Agent handoff: `docs/PROJECT_HANDOFF_20260221.md`

## Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| strict_records | 4489 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_completion_ratio | 1.0 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| v71_win_rate_vs_v7 | 0.4536082474226804 | `docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv` |
| v71_median_relative_gain_vs_v7_pct | -0.33387569285466684 | `docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv` |
| critical_failures | 4 | `docs/benchmarks/block3_truth_pack/failure_taxonomy.csv` |
| running_total | 8 | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |
| pending_total | 77 | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |
| v72_local_4090_completed_pairs | 20/20 | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |
| v72_local_4090_completion_ratio | 1.0 | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |
| v72_local_4090_fairness_all_pass | true | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |

## Notes

1. Use the master evidence document for full task/subtask, ladder, and SOTA analysis tables.
2. This file keeps only high-level snapshot metrics.
3. Local 4090 AutoFitV72 detailed records for Iris-side pull and comparison:
   `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.md`.
