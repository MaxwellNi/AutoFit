# Block 3 Benchmark Design (Current)

## Core Design

1. Strict-comparable benchmark lattice:
   - task x ablation x target x horizon = 104 expected conditions.
2. Unified condition ledger:
   - `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv`
3. Unified champion ledger:
   - `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv`

## Reporting Design

1. Human-readable full-table view:
   - `docs/BLOCK3_FULL_SOTA_BENCHMARK.md`
2. Machine-readable full-table view:
   - `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv`
3. Status and queue view:
   - `docs/benchmarks/block3_truth_pack/execution_status_latest.json`

## Governance Design

1. Freeze pointer only:
   - `docs/audits/FULL_SCALE_POINTER.yaml`
2. Read-only freeze assets.
3. No test-feedback-driven model selection.
