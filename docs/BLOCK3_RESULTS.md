# Block 3 Benchmark Results

> Last Updated: 2026-02-26 17:26:24 UTC
> Full strict-comparable table: `docs/BLOCK3_FULL_SOTA_BENCHMARK.md`
> Master evidence: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`

## Full SOTA Completion

| Metric | Value | Evidence |
|---|---:|---|
| strict_completed_conditions | 104/104 (100.0%) | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_records | 3666 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| strict_records_raw | 5547 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| full_sota_table_rows | 104 | `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv` |

## Champion Distribution (104 Conditions)

| Metric | Value | Evidence |
|---|---|---|
| champion_family_distribution | deep_classical=62, transformer_sota=36, foundation=6 | `docs/benchmarks/block3_truth_pack/full_sota_104_summary.json` |
| top_champion_models | NBEATS=39, PatchTST=24, NHITS=23, KAN=7, Chronos=6, NBEATSx=4, DLinear=1 | `docs/benchmarks/block3_truth_pack/full_sota_104_summary.json` |

## V7.2 Progress Relative To Full SOTA

| Metric | Value | Evidence |
|---|---:|---|
| v72_completed_conditions | 89/104 (85.6%) | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| v72_missing_keys | 15 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| running_total | 1 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |
| pending_total | 22 | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |

## Primary Artifacts

1. `docs/BLOCK3_FULL_SOTA_BENCHMARK.md` (one-table 104-condition strict result view)
2. `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv` (machine-readable 104-condition table)
3. `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv` (source condition leaderboard)
4. `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md` (complete evidence narrative)
