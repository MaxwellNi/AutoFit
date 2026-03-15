# Current Source of Truth

> Last verified: 2026-03-15
> Verified by direct scans of `runs/benchmarks/block3_phase9_fair/`, `runs/text_embeddings/`, live `squeue -u npin,cfisch`, and `sacct` for completed V739 jobs.

This file is the authoritative documentation entry point for the current Block 3 project state.
If any other document disagrees with this file, prefer this file and the evidence paths cited below.

## Authoritative Sources (Read in This Order)

1. Root `AGENTS.md`
2. `.local_mandatory_preexec.md`
3. `docs/CURRENT_SOURCE_OF_TRUTH.md`
4. `docs/PHASE9_V739_FACT_ALIGNMENT.md`
5. `docs/BLOCK3_MODEL_STATUS.md`
6. `docs/BLOCK3_RESULTS.md`
7. `docs/benchmarks/phase9_current_snapshot.md`
8. `docs/V739_CURRENT_RUN_MONITOR.md`
9. `docs/PHASE12_TEXT_RERUN_EXECUTION.md`
10. `runs/benchmarks/block3_phase9_fair/all_results.csv`
11. `runs/benchmarks/block3_phase9_fair/REPLICATION_MANIFEST.json`

## Verified Current Facts

| Fact | Current value | Evidence |
| --- | --- | --- |
| Canonical benchmark directory | `runs/benchmarks/block3_phase9_fair/` | direct scan |
| Raw metrics files | `91` | direct scan 2026-03-15 |
| Raw metric records | `10213` | direct scan 2026-03-15 (Phase 12 landing) |
| Raw models materialized | `91` | direct scan 2026-03-15 |
| Raw complete models (`≥104`) | `80` | direct scan 2026-03-15 |
| Raw partial models | `11` | direct scan 2026-03-15 |
| Current AutoFit baseline | `AutoFitV739` only | Root `AGENTS.md`, `.local_mandatory_preexec.md` |
| V739 landed benchmark conditions | `112/112` (ALL COMPLETE) | direct scan: 12 metrics.json under `*/autofit/*` |
| V739 ablations | core_only=28, core_edgar=28, core_text=28, full=28 | direct scan |
| V739 quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| V739 mean rank | **#13/80** (top 16%, 56 universal conditions) | per-condition ranking across 56 conditions |
| V739 mean rank score | 14.38 | lower is better |
| Text embedding artifacts | `AVAILABLE` | `runs/text_embeddings/embedding_metadata.json` |

## What the Current Benchmark Means

1. Phase 7 and Phase 8 results are historical only. They are not valid current benchmark evidence.
2. V734-V738 are retired because of oracle test-set leakage. They must not be used as baselines, ranking references, or implementation templates.
3. The current clean AutoFit line starts at V739, which uses validation-based selection (`val_raw`) instead of oracle tables.
4. The current physical Phase 9 fair benchmark still reflects the seed-replication reinterpretation:
   - `core_text` was renamed to `core_only_seed2`
   - `full` was renamed to `core_edgar_seed2`
   - see `runs/benchmarks/block3_phase9_fair/REPLICATION_MANIFEST.json`
5. Text embedding artifacts now exist. Phase 12 text reruns have been submitted (40 jobs, 2026-03-15) and are running. Once they land, `core_text` / `full` will reflect real text features.

## Current Execution Reality

1. Live queue snapshot verified on 2026-03-15:
   - `29 RUNNING` (15 npin/gpu Phase 12 + 5 npin/bigmem Phase 12 + 4 npin/l40s gap-fill + 9 cfisch/gpu gap-fill/FM)
   - `19 PENDING` (14 cfisch/gpu Phase 12 + 5 cfisch/bigmem Phase 12)
   - Gap-fill TSLib: 12 jobs (4 l40s + 8 gpu), 10 models × missing conditions
   - Gap-fill FM: 4/5 COMPLETED, 1 RUNNING (cf_gf_fm_t1_ces2)
   - Phase 12 text reruns: 20 RUNNING + 14 PENDING (all 8 model categories)
2. V739 status:
   - **ALL 12 V739 jobs COMPLETED** (8 l40s + 4 cfisch gpu)
   - 112 records landed in canonical `runs/benchmarks/block3_phase9_fair/*/autofit/*/metrics.json`
   - benchmark ranking established: **#13/80** by mean rank (14.38) across 56 universal conditions
   - V739 won 3 conditions (1 per task)
   - V739 is empirically valid: 0 NaN/Inf, 0 fallback, 100% fairness pass
3. Current text embedding status:
   - `runs/text_embeddings/text_embeddings.parquet` exists
   - `runs/text_embeddings/pca_model.pkl` exists
   - `runs/text_embeddings/embedding_metadata.json` exists
   - metadata reports `5,774,931` total rows and `64` PCA dimensions

## Current Priorities

1. ~~Land the first valid V739 results.~~ ✅ DONE (112/112 conditions landed).
2. Finish gap-fill for the 12 remaining partial models (NegBinGLM excluded = structural failure).
3. ~~Submit real text-enabled reruns.~~ ✅ DONE (40 Phase 12 jobs submitted, 20 running).
4. Wait for Phase 12 text reruns to land.
5. Consolidate the clean benchmark surface.
6. Only after Phase 9 coverage and real text/full reruns are stable should any V740+ work begin.

## What Is No Longer Current

1. Everything under `docs/_legacy_repo/` is historical archive material.
2. The old V72/V73 truth-pack line under `docs/benchmarks/LEGACY__block3_truth_pack__v72_v73/` is historical evidence, not the current operational truth for Phase 9 / V739.
3. Research/reference notes under `docs/references/` are background knowledge only. They are useful for design, but they are not status documents.
4. The archived full local checklist at `docs/_legacy_repo/local_mandatory_preexec_full_20260313.md` is preserved for history, not for current execution truth.
5. The archived large result table at `docs/_legacy_repo/BLOCK3_RESULTS_table_20260314.md` is preserved for traceability, not for current operational reading.

## Validation Commands

```bash
python3 scripts/build_phase9_current_snapshot.py
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/aggregate_block3_results.py
squeue -u npin,cfisch
for jid in $(squeue -u npin,cfisch -h -o '%i %j' | awk '$2 ~ /v739/ {print $1}'); do
  scontrol show job "$jid" | egrep 'JobId=|JobName=|Command=|WorkDir=|StdOut=|StdErr='
done
```
