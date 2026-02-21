# Block 3 Model Benchmark Status

> Last Updated: 2026-02-21 22:55:00 UTC
> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`
> Agent handoff: `docs/PROJECT_HANDOFF_20260221.md`

## Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| raw_records | 13871 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_records | 4489 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| legacy_unverified_records | 9382 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_condition_completion | 104/104 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| running_total | 8 | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |
| pending_total | 77 | `docs/benchmarks/block3_truth_pack/slurm_snapshot.json` |
| v72_local_4090_task1_core_only | 12/12 completed | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |
| v72_local_4090_task2_core_edgar | 8/8 completed | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |
| v72_local_4090_overall | 20/20 completed | `docs/benchmarks/block3_v72_local_4090/v72_completion_summary_20260221.json` |

## Notes

1. Detailed tables are centralized in the master evidence document.
2. This file is intentionally lightweight to avoid drift.
3. Local AutoFitV72 per-pair metrics are available at
   `docs/benchmarks/block3_v72_local_4090/v72_metrics_autofitv72_20260221.csv`.
