# Phase 12 Real Text Rerun Execution

> Last verified: 2026-03-15
> Scope: the first real `core_text` / `full` reruns on top of the current Phase 9 fair benchmark line.
> Status: 40 jobs SUBMITTED (20 RUNNING + 14 PENDING + 5 bigmem RUNNING + 1 cfisch gap-fill)

## Verified Current Readiness

1. Text embedding artifacts are present and non-empty:
   - `runs/text_embeddings/text_embeddings.parquet`
   - `runs/text_embeddings/pca_model.pkl`
   - `runs/text_embeddings/embedding_metadata.json`
2. Verified embedding metadata:
   - `n_total_rows = 5,774,931`
   - `n_unique_texts = 69,697`
   - `n_entities = 22,569`
   - `pca_dim = 64`
3. The Phase 12 preparation script is now usable under plain system Python:
   - `scripts/phase12_prepare_text_rerun.py --check-only`
4. Generated rerun scripts already exist under:
   - `.slurm_scripts/phase12/rerun/`
   - current count: `40` job scripts + `1` submission helper
5. The generated submission helper is:
   - `.slurm_scripts/phase12/rerun/submit_all_phase12_rerun.sh`
6. Current canonical fair benchmark still has only one stale physical `core_text` metrics file left in-place:
   - `runs/benchmarks/block3_phase9_fair/task1_outcome/ml_tabular/core_text/metrics.json`
   - record count: `4`
7. No physical `full/metrics.json` file is currently landed under `runs/benchmarks/block3_phase9_fair/`.

## What This Work Is Fixing

1. The current physical Phase 9 fair benchmark still represents the seed-replication interpretation:
   - `core_text -> core_only_seed2`
   - `full -> core_edgar_seed2`
2. That temporary interpretation remains the current benchmark reality only until the real text-enabled reruns land.
3. Phase 12 is the benchmark-correction step that will restore genuine `core_text` and `full` conditions.

## Canonical Tools

- Preparation / readiness / invalidation / submission entrypoint:
  - `scripts/phase12_prepare_text_rerun.py`
- Generated job scripts:
  - `.slurm_scripts/phase12/rerun/*.sh`
- Submission helper:
  - `.slurm_scripts/phase12/rerun/submit_all_phase12_rerun.sh`
- Canonical benchmark output root:
  - `runs/benchmarks/block3_phase9_fair/`

## Correct Execution Sequence

### 1. Verify readiness

```bash
python3 scripts/phase12_prepare_text_rerun.py --check-only
```

Expected current behavior:
- embeddings are reported as present
- the script no longer falsely blocks because `pandas` is missing

### 2. Back up any stale physical `core_text` / `full` outputs

```bash
python3 scripts/phase12_prepare_text_rerun.py --invalidate
```

This renames stale outputs instead of deleting them.

### 3. Regenerate the rerun scripts if needed

```bash
python3 scripts/phase12_prepare_text_rerun.py --generate
```

### 4. Submit the real reruns

```bash
bash .slurm_scripts/phase12/rerun/submit_all_phase12_rerun.sh
```

## Submission Record (2026-03-15)

| Account | Job IDs | Partition | Content | Status |
| --- | --- | --- | --- | --- |
| npin | 5251977-5251981 | gpu | deep_classical ct/fu (t1/t2/t3) | RUNNING |
| npin | 5251982-5251986 | gpu | foundation ct/fu (t1/t2/t3) | RUNNING |
| npin | 5251987-5251991 | gpu | irregular ct/fu (t1/t2/t3) | 1R + 4PD |
| npin | 5251992-5251996 | bigmem | statistical ct/fu (t1/t2/t3) | RUNNING |
| cfisch | 5252017-5252021 | gpu | af39=autofit ct/fu (t1/t2/t3) | PENDING |
| cfisch | 5252022-5252026 | bigmem | ml_tabular ct/fu (t1/t2/t3) | PENDING |
| cfisch | 5252027-5252031 | gpu | transformer_sota ct/fu (t1/t2/t3) | PENDING |
| cfisch | 5252032-5252036 | gpu | tslib_sota ct/fu (t1/t2/t3) | PENDING |

**Key script fixes applied before submission:**
- Removed explicit HF_HOME (system default works correctly)
- Added `umask 002` for group-writable output files
- Added `--requeue --signal=USR1@120` for checkpoint safety
- Pre-flight check: `if [[ ! -f "runs/text_embeddings/text_embeddings.parquet" ]]; then exit 3`

## Monitoring Commands

### Queue monitoring

```bash
squeue -u npin,cfisch | egrep 'JOBID|p12_'
```

### Output monitoring

```bash
find runs/benchmarks/block3_phase9_fair -path '*/core_text/metrics.json' -o -path '*/full/metrics.json' | sort
```

### Rebuild the project state after landings

```bash
python3 scripts/build_phase9_current_snapshot.py
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/aggregate_block3_results.py
```

## Landing Criteria

The Phase 12 reruns should only be treated as landed after all of the following are true:

1. physical `metrics.json` files exist under canonical `core_text/` and `full/` directories in `runs/benchmarks/block3_phase9_fair/`
2. new rows appear in `runs/benchmarks/block3_phase9_fair/all_results.csv`
3. `docs/benchmarks/phase9_current_snapshot.md` is rebuilt
4. `docs/BLOCK3_MODEL_STATUS.md` and `docs/BLOCK3_RESULTS.md` are refreshed from the rebuilt artifacts

## Non-Negotiable Rules

1. Do not treat `core_only_seed2` / `core_edgar_seed2` as real text-enabled ablations.
2. Do not claim any text effect until canonical `core_text` / `full` metrics land.
3. Do not write new outputs anywhere outside `runs/benchmarks/block3_phase9_fair/`.
