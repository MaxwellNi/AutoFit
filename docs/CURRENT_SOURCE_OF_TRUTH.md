# Current Source of Truth

> Last verified: 2026-03-26 11:49 CET
> Verified by direct scans of `runs/benchmarks/block3_phase9_fair/`, live `squeue -u npin`, `sacct`, benchmark aggregation scripts.

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
| Raw metric records | `16077` | direct scan 2026-03-26 |
| Raw models materialized | `137` | direct scan (116 real + 21 retired AutoFit@1) |
| Audit-excluded models | `24` | AUDIT_EXCLUDED_MODELS (17 old + 7 Finding H) |
| Active (leaderboard) models | `92` | 116 raw - 24 excluded |
| Raw complete models (`@160`) | `75` | direct unique-condition scan |
| Active complete models (`@160`) | `62` | 75 raw complete - 13 excluded complete models |
| Incomplete active models | `30` | 92 - 62 |
| Per-ablation | co=2849, s2=2139, ce=2780, e2=2778, ct=2764, fu=2767 | direct scan 2026-03-26 |
| Conditions per model | `160` | t1(72) + t2(48) + t3(40) |
| Current AutoFit baseline | `AutoFitV739` only | Root `AGENTS.md` |
| V739 landed conditions | `132/160` | co=28, ce=28, ct=28, fu=28, s2/e2 gap-filling (5R) |
| V739 quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| V739 mean rank | **#13** (top 14%, 92 active models) | per-condition ranking (last computed) |
| Post-filter distinct models in `all_results.csv` | `107` | includes 21 retired AutoFit legacy lines that still pass fairness/coverage filters |
| Post-filter non-retired models | `86` | `all_results.csv` minus retired AutoFit legacy lines |
| Text embedding artifacts | `AVAILABLE` | `runs/text_embeddings/embedding_metadata.json` |
| Phase 12 text reruns | `48/48 COMPLETED` | core_text+full 91/91 models |
| Phase 15 new models | 23 submitted, 15 valid, 8 excluded (Finding H), 78/160 | direct scan |
| Live jobs | `58` (29R + 29PD) | squeue 2026-03-26 11:49: gpu 23R+1PD, l40s 3R+14PD, hopper 3R+14PD |

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

1. Live queue snapshot verified on 2026-03-26 11:49 CET:
   - `29 RUNNING` = `23 gpu + 3 l40s + 3 hopper`
   - `29 PENDING` = `1 gpu + 14 hopper + 14 l40s`
   - **58 total**
   - Current gpu runners: 17 `g2_ac_*` + 5 `af739_*` + 1 `gpu_cos2_t2`
   - Current gpu pending: `v740_mb_case1_v739` (local-only resumable V740 compare, not part of canonical benchmark)
   - **ModernTCN bottleneck** remains the dominant throughput limiter for non-e2 accel jobs
   - Partition constraints (`sinfo` verified): `gpu=756G`, `l40s=515G`, `hopper=2063754MB (~2.06TB)`; the earlier `hopper=201G` claim was a unit-reading error
2. V739 status:
   - **132/160 conditions landed** (+1 from 131, s2/e2 gap-filling)
   - 5 af739 jobs RUNNING: `t1_e2`, `t1_s2`, `t2_s2`, `t2_e2`, `t3_e2`
   - Missing structure: `t1_e2=9`, `t1_s2=8`, `t2_e2=4`, `t2_s2=4`, `t3_e2=3` (28 total)
   - V739 is empirically valid: 0 NaN/Inf, 0 fallback, 100% fairness pass
3. Finding H (discovered 2026-03-22):
   - 8 P15 models produce 100% constant predictions (0% fairness pass)
   - CFPT, DeformableTST, MICN, PathFormer, SEMPO, SparseTSF, TimeBridge, TimePerceiver
   - Added to AUDIT_EXCLUDED_MODELS in aggregate_block3_results.py
   - Removed from accel_v2 scripts → ~30% faster per job
4. accel_v2 progress (g2_ac GPU jobs, 32h into 2-day allocation, ~16h remaining):
   - e2: 33-34 new + 65 skip (near complete, processing FreTS/FilterTS) → will complete ✓
   - co: 6 new + 98-121 skip → ALL stuck on ModernTCN (40min/epoch), needs 1-2 requeues
   - ct: 9 new + 93 skip → on MSGNet, moderate progress, needs 1 requeue
   - s2: 6 new + 98 skip → ALL stuck on ModernTCN, needs 1-2 requeues
   - ce: 6 new + 98 skip → stuck on ModernTCN, needs 1-2 requeues
   - fu: 7 new + 93 skip → on Fredformer, moderate progress, needs 1 requeue
   - **ModernTCN universal bottleneck**: 20M params, ~40min/epoch — causes most TIMEOUT situations
   - **Fixed**: Added `_requeue_handler()` trap to all 56 scripts (2026-03-22 20:10)
5. Critical gaps:
   - s2 (core_only_seed2): `gpu_cos2_t2` remains the canonical task2 seed2 gap-fill and has now been **cancelled/requeued onto the trimmed 23-model list** (`5284505`)
   - e2 (core_edgar_seed2): 2778 records (+54 from 2724), accel_v2 e2 scripts producing the bulk of recent growth
   - V739: 28 missing s2+e2 conditions, and **all 5 required gap-fill jobs are now running**
6. Text embeddings:
   - `runs/text_embeddings/text_embeddings.parquet` — 5,774,931 rows, 64 PCA dims
   - Phase 12 all 48/48 complete. core_text+full coverage: 91/91 models
7. Audit-excluded models: 24 total (was 23, added NegativeBinomialGLM as Structural)

## Current Priorities

1. ~~Land the first valid V739 results.~~ ✅ DONE (124/160 conditions landed, 112 co+ce+ct+fu complete).
2. ~~Finish gap-fill for partial models.~~ ✅ Current full-160 active frontier is 62 models; remaining incomplete active models are covered by accel_v2 or structural OOM exceptions.
3. ~~Submit real text-enabled reruns.~~ ✅ Phase 12 DONE (48/48 COMPLETED).
4. ~~Phase 12 text reruns to land.~~ ✅ DONE. core_text+full 91/91 models.
5. ~~Phase 15 new TSLib models.~~ ✅ Submitted. 15 valid, 8 excluded (Finding H).
6. ~~Cancel old v1 l40s/hopper jobs.~~ ✅ DONE. 8 old v1 jobs cancelled, freed L40S for v2.
7. Complete V739 s2/e2 gap-fill (5 af739 RUNNING).
8. Complete e2 gap for ETSformer/LightTS/Pyraformer/Reformer (0/28 e2 each — covered by accel_v2).
9. Complete P15 model gap-fill to 160/160 via accel_v2 (54 total jobs, auto-requeue).
10. g2_ac_v2 auto-requeue: e2/ct complete first run; co/s2/ce/fu need 2-3 requeues, with `gpu_cos2_t2` now corrected to the trimmed 23-model list.
11. Local-only V740 work is now permitted via resumable side-paths that do **not** touch the canonical benchmark harness; the first such job is `v740_mb_case1_v739`.

## What Is No Longer Current

1. Everything under `docs/_legacy_repo/` is historical archive material.
2. The old V72/V73 truth-pack line under `docs/benchmarks/LEGACY__block3_truth_pack__v72_v73/` is historical evidence, not the current operational truth for Phase 9 / V739.
3. Research/reference notes under `docs/references/` are background knowledge only. They are useful for design, but they are not status documents.
4. The archived full local checklist at `docs/_legacy_repo/local_mandatory_preexec_full_20260313.md` is preserved for history, not for current execution truth.
5. The archived large result table at `docs/_legacy_repo/BLOCK3_RESULTS_table_20260314.md` is preserved for traceability, not for current operational reading.

## Validation Commands

```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/build_phase9_current_snapshot.py
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/aggregate_block3_results.py
squeue -u npin,cfisch
for jid in $(squeue -u npin,cfisch -h -o '%i %j' | awk '$2 ~ /(v739|af739)/ {print $1}'); do
  scontrol show job "$jid" | egrep 'JobId=|JobName=|Command=|WorkDir=|StdOut=|StdErr='
done
```
