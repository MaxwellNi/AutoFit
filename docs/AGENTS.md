# Block 3 Documentation Entry

This file is the documentation-side handoff entry for the current project state.

## Current Authority

Read these first:
1. Root `AGENTS.md`
2. `.local_mandatory_preexec.md`
3. `docs/CURRENT_SOURCE_OF_TRUTH.md`
4. `docs/PHASE9_V739_FACT_ALIGNMENT.md`
5. `docs/BLOCK3_MODEL_STATUS.md`
6. `docs/BLOCK3_RESULTS.md`
7. `docs/benchmarks/phase9_current_snapshot.md`
8. `docs/V739_CURRENT_RUN_MONITOR.md`
9. `docs/PHASE12_TEXT_RERUN_EXECUTION.md`

## Current Scope

1. Canonical benchmark: `runs/benchmarks/block3_phase9_fair/`
2. Canonical AutoFit baseline: `AutoFitV739`
3. Canonical data source: WIDE2 freeze via `docs/audits/FULL_SCALE_POINTER.yaml`
4. Current real benchmark state:
   - raw complete models: `77`
   - filtered complete models: `59`
   - V739 landed conditions: `0/104`
   - observed live V739 queue still targets legacy `runs/benchmarks/block3_phase10/v739/`
   - text embeddings: available

## Interpret the Current Benchmark Correctly

1. The current physical Phase 9 results still use the seed-replication reinterpretation of `core_text` and `full`.
2. Real text-enabled reruns have not landed yet.
3. Observed live V739 jobs are not yet pointed at the canonical fair benchmark root.
4. Old V72/V73 truth-pack materials are not current operational truth.
5. `docs/PHASE9_V739_FACT_ALIGNMENT.md` should be used whenever old claims conflict with current artifacts.

## What to Ignore as Current State

1. Anything in `docs/_legacy_repo/`
2. Historical V72/V73 master documents
3. Old Phase 7/8 summaries
4. Archived oracle-leaked AutoFit narratives
5. The archived large result table in `docs/_legacy_repo/BLOCK3_RESULTS_table_20260314.md`

## Current Next Actions

1. Land V739 results.
2. Close the remaining partial-model gaps.
3. Run the real text/full reruns now that text embedding artifacts exist.
4. Use `docs/V739_CURRENT_RUN_MONITOR.md` and `docs/PHASE12_TEXT_RERUN_EXECUTION.md` as the current execution docs.
5. Only after that should a new AutoFit version be started.
