# Current Source of Truth

> Last verified: 2026-03-22 20:15 CET
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
| Raw metric records | `15864` | direct scan 2026-03-22 20:12 |
| Raw models materialized | `137` | direct scan (116 real + 21 retired AutoFit@1) |
| Audit-excluded models | `24` | AUDIT_EXCLUDED_MODELS (17 old + 7 Finding H) |
| Active (leaderboard) models | `92` | 116 raw - 24 excluded |
| Raw complete models (`@160`) | `77` | direct scan (64 active + 13 excluded @160) |
| Active complete models (`@160`) | `64` | 77 - 13 excluded@160 |
| Incomplete active models | `28` | 92 - 64 |
| Per-ablation | co=2849, s2=2176, ce=2780, e2=2786, ct=2640, fu=2633 | direct scan 2026-03-22 20:12 |
| Conditions per model | `160` | t1(72) + t2(48) + t3(40) |
| Current AutoFit baseline | `AutoFitV739` only | Root `AGENTS.md` |
| V739 landed conditions | `125/160` | co=28, ce=28, ct=28, fu=28, s2+e2 gap-filling (2R+3 resubmitted) |
| V739 quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| V739 mean rank | **#13** (top 14%, 92 active models) | per-condition ranking (last computed) |
| Text embedding artifacts | `AVAILABLE` | `runs/text_embeddings/embedding_metadata.json` |
| Phase 12 text reruns | `48/48 COMPLETED` | core_text+full 91/91 models |
| Phase 15 new models | 23 submitted, 15 valid, 8 excluded (Finding H), ~69/160 | direct scan |
| Live jobs | `60` (28R + 32PD) | squeue 2026-03-22 20:12 |

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

1. Live queue snapshot verified on 2026-03-22 20:10 CET:
   - `29 RUNNING` (17 g2_ac_v2/gpu + 5 l2_ac_v2/l40s + 1 h2_ac_v2/hopper + 2 af739/gpu + 1 gpu_cos2_t2 + 3 af739 resubmit/gpu PD)
   - `31 PENDING` (12 l2_ac_v2/l40s + 16 h2_ac_v2/hopper + 3 af739 resubmit)
   - **60 total** (3 af739 TIMEOUT resubmitted today)
   - Partition constraints (sinfo verified): L40S max 4-5 concurrent (2 nodes, CPU bottleneck), Hopper max 1 (1 node, 201G RAM)
   - **Critical fix**: `--requeue` alone does NOT auto-restart on TIMEOUT. Added `_requeue_handler()` trap to all 56 scripts.
2. V739 status:
   - **125/160 conditions landed** (+5 from 120, s2/e2 gap-filling)
   - 2 af739 jobs RUNNING (t1_e2@32h, t3_e2@47h near timeout), 3 resubmitted (t1_s2, t2_s2, t2_e2)
   - V739 gap: t1_s2(9), t1_e2(10), t2_s2(6), t2_e2(5), t3_e2(5) = 35 conditions missing
   - V739 is empirically valid: 0 NaN/Inf, 0 fallback, 100% fairness pass
3. Finding H (discovered 2026-03-22):
   - 8 P15 models produce 100% constant predictions (0% fairness pass)
   - CFPT, DeformableTST, MICN, PathFormer, SEMPO, SparseTSF, TimeBridge, TimePerceiver
   - Added to AUDIT_EXCLUDED_MODELS in aggregate_block3_results.py
   - Removed from accel_v2 scripts → ~30% faster per job
4. accel_v2 progress (g2_ac GPU jobs, 18h into 2-day allocation):
   - e2: 30/23 models started (near complete, processing CARD) → will complete ✓
   - co: 5/23 → needs 1 requeue
   - ct: 6/23 → needs 1 requeue
   - s2: 5/23 → needs 1 requeue
   - ce: 4/23 → needs 2+ requeues
   - fu: 4/23 → needs 2+ requeues
   - **Critical**: `--requeue + --signal=USR1@120` alone does NOT trigger auto-restart on TIMEOUT
   - **Fixed**: Added `_requeue_handler()` trap to all 56 scripts (2026-03-22 20:10)
5. Critical gaps:
   - s2 (core_only_seed2): 2176 records (↑172 from 2004), 27 models missing t2 s2 (gpu_cos2_t2 RUNNING 9/27)
   - e2 (core_edgar_seed2): 2786 records (↑296 from 2490), ETSformer/LightTS/Pyraformer/Reformer gap covered by accel_v2
   - V739: 35 missing s2+e2 conditions (2 af739 RUNNING + 3 resubmitted)
6. Text embeddings:
   - `runs/text_embeddings/text_embeddings.parquet` — 5,774,931 rows, 64 PCA dims
   - Phase 12 all 48/48 complete. core_text+full coverage: 91/91 models
7. Audit-excluded models: 24 total (was 23, added NegativeBinomialGLM as Structural)

## Current Priorities

1. ~~Land the first valid V739 results.~~ ✅ DONE (124/160 conditions landed, 112 co+ce+ct+fu complete).
2. ~~Finish gap-fill for partial models.~~ ✅ 64 active @160. 28 incomplete covered by accel_v2.
3. ~~Submit real text-enabled reruns.~~ ✅ Phase 12 DONE (48/48 COMPLETED).
4. ~~Phase 12 text reruns to land.~~ ✅ DONE. core_text+full 91/91 models.
5. ~~Phase 15 new TSLib models.~~ ✅ Submitted. 15 valid, 8 excluded (Finding H).
6. ~~Cancel old v1 l40s/hopper jobs.~~ ✅ DONE. 8 old v1 jobs cancelled, freed L40S for v2.
7. Complete V739 s2/e2 gap-fill (2 af739 RUNNING, 3 TIMEOUT — very slow, ~3 conds per 2d).
8. Complete e2 gap for ETSformer/LightTS/Pyraformer/Reformer (0/28 e2 each — covered by accel_v2).
9. Complete P15 model gap-fill to 160/160 via accel_v2 (54 total jobs, auto-requeue).
10. g2_ac_v2 auto-requeue: e2/ct complete first run; co/s2/ce/fu need 2-3 requeues.
11. Only after all jobs complete and coverage stable should V740+ work begin.

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
