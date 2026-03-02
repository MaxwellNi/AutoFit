# Block 3 Benchmark Design (Current)

## Core Design

1. Strict-comparable benchmark lattice:
   - `task x ablation x target x horizon = 104` expected conditions
2. Unified condition ledger:
   - `docs/benchmarks/block3_truth_pack/condition_inventory_full.csv`
3. Unified champion ledger:
   - `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv`

## Reporting Design

1. Human-readable canonical benchmark view:
   - `docs/BLOCK3_FULL_SOTA_BENCHMARK.md`
2. Machine-readable canonical benchmark view:
   - `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv`
3. Current execution and queue view:
   - `docs/benchmarks/block3_truth_pack/execution_status_latest.json`
4. Current V7.2 standing:
   - `docs/benchmarks/block3_truth_pack/v72_rank_104_table_latest.csv`
   - `docs/benchmarks/block3_truth_pack/v72_rank_104_summary_latest.json`

## Observed V7.2 Failure Pattern

1. V7.2 is complete on the strict lattice (`104/104`) but it does not win any strict condition.
2. The dominant failure mode is lane imbalance:
   - `funding_raised_usd` is near the frontier
   - `investors_count` is the main loss driver
   - `is_funded` remains materially behind the benchmark leaders
3. The design implication is that V7.3 should not tune globally first. It should fix the count lane first, then the binary lane.

## V7.3 Design Rules

1. Reuse-first:
   - do not rerun already materialized strict conditions unless a deliberate V7.3 experiment requires it
2. Lane-specific redesign:
   - count lane: two-part count head, non-negative and spike-safe guards
   - binary lane: hazard head with OOF-only calibration selection
   - heavy-tail lane: robust dual-objective losses with tail diagnostics
3. Evidence-first:
   - all policy updates must be justified by strict baseline evidence and OOF-only evaluation

## Governance Design

1. Freeze pointer only:
   - `docs/audits/FULL_SCALE_POINTER.yaml`
2. Read-only freeze assets.
3. No test-feedback-driven model selection.
4. Contract assertion before every execution or submission path.
