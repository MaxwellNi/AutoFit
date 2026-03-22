# Current Source of Truth

> Last verified: 2026-03-22 02:30 CET
> Verified by direct scans of `runs/benchmarks/block3_phase9_fair/`, `runs/text_embeddings/`, live `squeue -u npin`, and `sacct` for completed jobs.

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
| Raw metric records | `15608` | direct scan 2026-03-22 02:10 |
| Raw models materialized | `137` | direct scan (116 real + 21 retired AutoFit@1) |
| Audit-excluded models | `24` | AUDIT_EXCLUDED_MODELS (17 old + 7 Finding H) |
| Active (leaderboard) models | `92` | 116 raw - 24 excluded |
| Raw complete models (`@160`) | `77` | direct scan (64 active + 13 excluded @160) |
| Active complete models (`@160`) | `64` | 77 - 13 excluded@160 |
| Incomplete active models | `28` | 92 - 64 |
| Per-ablation | co=2803, s2=2004, ce=2780, e2=2490, ct=2764, fu=2767 | direct scan |
| Conditions per model | `160` | t1(72) + t2(48) + t3(40) |
| Current AutoFit baseline | `AutoFitV739` only | Root `AGENTS.md` |
| V739 landed conditions | `120/160` | co=28, ce=28, ct=28, fu=28, s2=3, e2=5 |
| V739 quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| V739 mean rank | **#13** (top 14%, 92 active models) | per-condition ranking |
| Text embedding artifacts | `AVAILABLE` | `runs/text_embeddings/embedding_metadata.json` |
| Phase 12 text reruns | `48/48 COMPLETED` | core_text+full 91/91 models |
| Phase 15 new models | 23 submitted, 15 valid, 8 excluded (Finding H) | direct scan |
| Live jobs | `67` (33R + 34PD) | squeue 2026-03-22 02:30 |

## What the Current Benchmark Means

1. Phase 7 and Phase 8 results are historical only. They are not valid current benchmark evidence.
2. V734-V738 are retired because of oracle test-set leakage. They must not be used as baselines, ranking references, or implementation templates.
3. The current clean AutoFit line starts at V739, which uses validation-based selection (`val_raw`) instead of oracle tables.
4. The current physical Phase 9 fair benchmark has 6 ablations:
   - `core_only` (co) — baseline numeric features
   - `core_only_seed2` (s2) — seed stability check
   - `core_edgar` (ce) — EDGAR features added
   - `core_edgar_seed2` (e2) — EDGAR + seed2
   - `core_text` (ct) — text embeddings added
   - `full` (fu) — all features combined
5. Text embedding artifacts exist. Phase 12 text reruns COMPLETED (48/48 jobs). `core_text` / `full` reflect real text features.
6. Phase 15 added 23 new TSLib models. 8 produce 100% constant predictions (Finding H) and are excluded. 15 valid models remain at ~67/160 conditions.

## Current Execution Reality

1. Live queue snapshot verified on 2026-03-22 02:30:
   - `33 RUNNING` (17 g2_ac_v2/gpu + 5 l40_ac/l40s + 3 hp_ac/hopper + 5 af739/gpu + 3 old gpu_ac)
   - `34 PENDING` (17 l2_ac_v2/l40s + 17 h2_ac_v2/hopper)
   - accel_v2 scripts: 23 models × 17 conditions × 3 partitions = 51 new optimized jobs
   - Old v1 l40s/hopper PENDING cancelled, replaced by accel_v2
2. V739 status:
   - **120/160 conditions landed** (co=28, ce=28, ct=28, fu=28, s2=3, e2=5)
   - 5 af739 jobs RUNNING but very slow (~3 conditions per 2-day allocation)
   - V739 is empirically valid: 0 NaN/Inf, 0 fallback, 100% fairness pass
3. Finding H (discovered 2026-03-22):
   - 8 P15 models produce 100% constant predictions (0% fairness pass)
   - CFPT, DeformableTST, MICN, PathFormer, SEMPO, SparseTSF, TimeBridge, TimePerceiver
   - Added to AUDIT_EXCLUDED_MODELS in aggregate_block3_results.py
   - Removed from accel_v2 scripts → ~30% faster per job
4. Critical gaps:
   - s2 (core_only_seed2): only 2004 records, 27 models missing t2 s2
   - e2 (core_edgar_seed2): only 2490 records, ETSformer/LightTS/Pyraformer/Reformer at 0/28 e2
   - V739: 40 missing s2+e2 conditions
5. Text embeddings:
   - `runs/text_embeddings/text_embeddings.parquet` — 5,774,931 rows, 64 PCA dims
   - Phase 12 all 48/48 complete. core_text+full coverage: 91/91 models

## Current Priorities

1. ~~Land the first valid V739 results.~~ ✅ DONE (120/160 conditions landed, 112 co+ce+ct+fu complete).
2. ~~Finish gap-fill for partial models.~~ ✅ 64 active @160. 28 incomplete covered by accel_v2.
3. ~~Submit real text-enabled reruns.~~ ✅ Phase 12 DONE (48/48 COMPLETED).
4. ~~Phase 12 text reruns to land.~~ ✅ DONE. core_text+full 91/91 models.
5. ~~Phase 15 new TSLib models.~~ ✅ Submitted. 15 valid, 8 excluded (Finding H).
6. Complete V739 s2/e2 gap-fill (5 af739 RUNNING, slow).
7. Complete e2 gap for ETSformer/LightTS/Pyraformer/Reformer (0/28 e2 each).
8. Complete P15 model gap-fill to 160/160 via accel_v2 (67 jobs running).
9. Only after all jobs complete and coverage stable should V740+ work begin.

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
