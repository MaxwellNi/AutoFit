# Block 3 Model Benchmark Status

> Last Updated: 2026-02-26 17:26:24 UTC
> Full strict-comparable table: `docs/BLOCK3_FULL_SOTA_BENCHMARK.md`
> Master evidence: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`

## Global Coverage Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| raw_records | 14929 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_records | 3666 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_records_raw | 5547 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| legacy_unverified_records | 9382 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_condition_completion | 104/104 (100.0%) | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |

## Full SOTA Champion Structure

| Metric | Value | Evidence |
|---|---|---|
| champion_family_distribution | deep_classical=62, transformer_sota=36, foundation=6 | `docs/benchmarks/block3_truth_pack/full_sota_104_summary.json` |
| top_champion_models | NBEATS=39, PatchTST=24, NHITS=23, KAN=7, Chronos=6, NBEATSx=4, DLinear=1 | `docs/benchmarks/block3_truth_pack/full_sota_104_summary.json` |
| full_sota_table_rows | 104 | `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv` |

## Active Queue Snapshot (V72-first)

| Metric | Value | Evidence |
|---|---:|---|
| running_total | 1 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| pending_total | 22 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| v72_progress | 89/104 (85.6%) | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| v72_missing_keys | 15 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| queue_bottleneck_top_reason | QOSGrpNodeLimit | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |

## Reading Order

1. `docs/BLOCK3_FULL_SOTA_BENCHMARK.md`
2. `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv`
3. `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`
