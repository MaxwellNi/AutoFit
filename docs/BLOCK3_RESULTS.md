# Block 3 Benchmark Results

> Last Updated: 2026-02-26T00:08:32.075104+00:00
> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`
> V7.3 execution spec: `docs/BLOCK3_V73_RESEARCH_EXECUTION_SPEC_20260225.md`

## Strict Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| strict_records | 3665 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_condition_completion | 104/104 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| v72_pilot_overall_pass | False | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| v72_global_improvement_pct | -0.37433548220324225 | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| v72_investors_gap_reduction_pct | -2.9660362701370064 | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |
| v73_reuse_manifest_rows | 104 | `docs/benchmarks/block3_truth_pack/v73_reuse_manifest.csv` |

## Notes

1. V7.3 aims to close remaining key coverage and lane-specific performance gaps without altering fairness rules.
2. Reuse-first policy avoids redundant reruns of already materialized strict comparable keys.
