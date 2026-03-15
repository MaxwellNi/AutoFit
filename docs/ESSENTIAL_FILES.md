# Essential Files (Current Project Surface)

This checklist defines the minimum files a future contributor must read before treating any project statement as current fact.

## A. Mandatory Current-State Files

- Root `AGENTS.md`
- `.local_mandatory_preexec.md`
- `docs/CURRENT_SOURCE_OF_TRUTH.md`
- `docs/PHASE9_V739_FACT_ALIGNMENT.md`
- `docs/BLOCK3_MODEL_STATUS.md`
- `docs/BLOCK3_RESULTS.md`
- `docs/benchmarks/phase9_current_snapshot.md`
- `docs/V739_CURRENT_RUN_MONITOR.md`
- `docs/PHASE12_TEXT_RERUN_EXECUTION.md`
- `docs/MINIMAL_CURRENT_HANDOFF.md`
- `docs/BLOCK3_EXECUTION_CONTRACT.md`

## B. Mandatory Current Code Paths

- `src/narrative/data_preprocessing/block3_dataset.py`
- `src/narrative/block3/models/registry.py`
- `src/narrative/block3/models/nf_adaptive_champion.py`
- `src/narrative/block3/models/tslib_models.py`
- `scripts/run_block3_benchmark_shard.py`
- `scripts/aggregate_block3_results.py`
- `scripts/build_phase9_current_snapshot.py`

## C. Mandatory Current Artifact Paths

- `runs/benchmarks/block3_phase9_fair/all_results.csv`
- `runs/benchmarks/block3_phase9_fair/REPLICATION_MANIFEST.json`
- `runs/text_embeddings/embedding_metadata.json`
- `runs/text_embeddings/text_embeddings.parquet`

## D. Historical or Reference Material (Do Not Treat as Current State)

- `docs/_legacy_repo/`
- `docs/benchmarks/LEGACY__block3_truth_pack__v72_v73/` (historical truth-pack line)
- `docs/references/`
