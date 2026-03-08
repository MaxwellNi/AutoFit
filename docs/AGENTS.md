# Block 3 Project Context (Documentation Entry)

This file records the current handoff baseline for Block 3 benchmarking and model iteration.

## Scope

1. Data freeze: `TRAIN_WIDE_FINAL` with pointer `docs/audits/FULL_SCALE_POINTER.yaml`.
2. Benchmark scope: Phase 9 fair benchmark — 100 models across 104 conditions.
3. Active AutoFit versions: V734, V735, V736 only (all prior versions dropped).
4. Canonical results directory: `runs/benchmarks/block3_phase9_fair/`.

## End-to-End Chronology

1. The WIDE2 freeze was sealed and verified. All five freeze gates passed, and Block 3 now reads only through `FreezePointer`.
2. Phase 7 benchmark completed with 91 models and 6,670 records.
3. Phase 8 extended the benchmark to 99 models and 7,478 records.
4. Four critical bugs were discovered that invalidated all Phase 7/8 results:
   - TSLib per-entity prediction (all entities received identical forecasts)
   - Foundation model hardcoded `prediction_length=7`
   - Moirai/Moirai2 50-entity cap (unfair vs Chronos which processed all)
   - ml_tabular single-horizon restriction
5. Phase 9 re-benchmark initiated with all bug fixes applied. 5,083 valid records from 50 models copied to clean directory.
6. AutoFit V1-V733 and FusedChampion dropped. Only V734/V735/V736 retained.

## Current Status

1. Phase 9 fair benchmark in progress — 5,083 valid records, 50 models materialized.
2. 66 SLURM scripts prepared in `.slurm_scripts/phase9/` for remaining models.
3. NF training configs use their original committed values (no equalization applied).
4. All Phase 7/8 results marked deprecated in `runs/benchmarks/block3_20260203_225620_phase7/DEPRECATED.md`.

## Canonical References

1. Results summaries:
   - `docs/BLOCK3_RESULTS.md`
   - `docs/BLOCK3_MODEL_STATUS.md`
2. Phase 9 plan:
   - `docs/PHASE9_ULTIMATE_PLAN.md`
3. Execution contract:
   - `docs/BLOCK3_EXECUTION_CONTRACT.md`

## Constraints

1. Freeze assets under `runs/*_20260203_225620/` are read-only.
2. Only V734/V735/V736 are active AutoFit versions.
3. Canonical output directory: `runs/benchmarks/block3_phase9_fair/`.
4. NF configs must match their committed (production) values.

## Immediate Next Work

1. Submit Phase 9 SLURM scripts for remaining ~50 models.
2. Aggregate results once Phase 9 re-runs complete.
3. Rebuild leaderboard and paper tables from Phase 9 clean data.
4. TCAV-style concept importance analysis.
