# Block 3 Project Context (Documentation Entry)

This file mirrors the current project execution context for Block 3 benchmarking and model iteration.

## Scope

1. Data freeze: `TRAIN_WIDE_FINAL` with pointer `docs/audits/FULL_SCALE_POINTER.yaml`.
2. Benchmark scope: strict-comparable condition lattice (`104/104` completed).
3. Reference artifacts:
   - `docs/BLOCK3_FULL_SOTA_BENCHMARK.md`
   - `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv`
   - `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`

## Current Status

1. Full SOTA strict benchmark is complete (`104/104`).
2. V7.2 completion is still in progress (latest status in `docs/benchmarks/block3_truth_pack/execution_status_latest.json`).
3. Queue policy remains V72-first, with bottleneck reasons logged in `docs/benchmarks/block3_truth_pack/slurm_snapshot.json`.

## Constraints

1. Freeze assets under `runs/*_20260203_225620/` are read-only.
2. No benchmark conclusions may use non-strict rows.
3. Evaluation and selection must remain train/validation/OOF only.
