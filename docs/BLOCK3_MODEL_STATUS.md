# Block 3 Model Benchmark Status

> Last Updated: 2026-02-26T00:08:32.075104+00:00
> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`
> V7.3 execution spec: `docs/BLOCK3_V73_RESEARCH_EXECUTION_SPEC_20260225.md`

## Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| strict_condition_completion | 104/104 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| v72_missing_keys | 16 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| v72_progress_bar | [####################----] 88/104 (84.6%) | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| running_total | 0 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| pending_total | 20 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |

## Notes

1. V7.3 handoff artifacts are generated under `docs/benchmarks/block3_truth_pack/`.
2. Non-GitHub synchronization uses pull-from-iris rsync workflow on the target GPU host.
