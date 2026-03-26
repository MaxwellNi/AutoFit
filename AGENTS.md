# Agent Context

## Mission

Maintain and extend Block 3 on the finalized WIDE2 freeze using the **current clean benchmark line only**:
- canonical benchmark root: `runs/benchmarks/block3_phase9_fair/`
- canonical AutoFit baseline: `AutoFitV739`
- canonical data pointer: `docs/audits/FULL_SCALE_POINTER.yaml`

## Read First

Every future contributor should read these files in order before making claims or changes:
1. `.local_mandatory_preexec.md`
2. `docs/CURRENT_SOURCE_OF_TRUTH.md`
3. `docs/PHASE9_V739_FACT_ALIGNMENT.md`
4. `docs/BLOCK3_MODEL_STATUS.md`
5. `docs/BLOCK3_RESULTS.md`
6. `docs/benchmarks/phase9_current_snapshot.md`
7. `docs/V739_CURRENT_RUN_MONITOR.md`
8. `docs/PHASE12_TEXT_RERUN_EXECUTION.md`
9. `docs/PHASE9_V739_LESSONS_LEARNED.md`

## Verified Current State (2026-03-26 10:19)

- Freeze: complete and read-only
- Canonical benchmark: `runs/benchmarks/block3_phase9_fair/`
- Raw benchmark scan (2026-03-26 10:18):
  - `16077` raw records in metrics.json (+54 from 16023@2026-03-25, all from e2 ablation)
  - `137` raw models in benchmark (116 real + 21 retired AutoFit)
  - `24` audit-excluded models (unchanged)
  - `92` active (leaderboard) models = 116 raw - 24 excluded
  - `75` raw models at 160/160, `62` active (leaderboard) models at 160/160
  - `28` incomplete active models
  - P15 valid models: at 78/160 (+2 from 76, accel_v2 producing)
  - `160` max per model: t1(72) + t2(48) + t3(40). task3 has no core_only_seed2.
  - Per-ablation: co=2849, s2=2139, ce=2780, e2=2778(+54), ct=2764, fu=2767
  - `87%` task redundancy: ~72 truly unique evaluation cells
- AutoFit:
  - V734-V738 are retired due to oracle test-set leakage
  - V739 is the only valid current AutoFit baseline
  - V739 landed conditions: `132/160` (+1 from 131, s2/e2 gap-filling)
  - V739 s2/e2 gap-fill: 5R af739 — all running (t1_e2 28h, t1_s2 27h, t2_s2 27h, t2_e2 27h, t3_e2 19h)
  - V739 quality: 0 NaN/Inf, 0 fallback, 100% fairness pass
- Top-5 models by mean rank: PatchTST (4.28), NHITS (4.38), NBEATS (5.01), NBEATSx (5.81), ChronosBolt (7.42)
- Dominant champion model: NBEATS — 65/160 conditions won (40.6%)
- Champion analysis: `docs/references/BLOCK3_CHAMPION_COMPONENT_ANALYSIS.md` updated 2026-03-18
  - Key finding: text embeddings HURT (core_text wins only 11.7% of pairs)
  - Key finding: 87% task redundancy (~72 truly unique cells, not 160)
  - DeepNPTS is task1-only specialist (16/16 wins on task1_outcome)
  - PatchTST best mean rank (4.28) but only 4 wins (generalist)
  - Seed2 highly stable: avg |delta| 0.138%, median 0.00%
  - EDGAR mixed: wins 34.7% of pairs (target-dependent)
  - Full vs core_only: closest margin (45.0% vs 42.0%)
- Model registry: `docs/references/MODEL_REGISTRY.md` — 152 registered, 117 active, 99 in benchmark
- Model completion tiers (2026-03-26):
  - @160 (complete): 62 active + 13 excluded = 75 raw (unchanged)
  - @159: XGBoost (structural OOM, UNFIXABLE)
  - @157: XGBoostPoisson (structural OOM, UNFIXABLE)
  - @132: AutoFitV739 (5R af739, ~3-5 conditions/2d)
  - @114: Chronos2, TTM (foundation seed2/e2 partial)
  - @105: Crossformer/MSGNet/MambaSimple/PAttn (accel_v2 covering gaps)
  - @93: ETSformer/LightTS/Pyraformer/Reformer (accel_v2 covering e2 gap)
  - @78: 15 valid P15 models (accel_v2 covering, ModernTCN bottleneck)
  - @69: 8 excluded P15 (Finding H)
  - s2 gap: gpu_cos2_t2 running with trimmed 23-model list
  - NegativeBinomialGLM: excluded (structural failure, 21 records)
  - **8 P15 models EXCLUDED (Finding H)**: CFPT, DeformableTST, PathFormer, SEMPO, SparseTSF, TimeBridge, TimePerceiver — all 100% constant predictions
- Phase 12 text reruns:
  - 48+1 total scripts (40 original + 8 t3_ct + 1 OOM fix)
  - 48 COMPLETED, cfisch tslib 6 TIMEOUT@2d (CATS/FITS/KANAD/WPMixer ct/fu partial)
  - ml_t_t1_fu_fix OOM@200G → 320G also OOM → resubmitted on bigmem 640G (job 5262824)
  - core_text coverage: **91/91** models (ALL categories complete)
  - full coverage: **91/91** models (ALL, NegBinGLM has partial records)
  - Scripts: `.slurm_scripts/phase12/rerun/`
  - All scripts: umask 002, --requeue, text embedding pre-flight check
- Text embeddings:
  - artifacts exist in `runs/text_embeddings/` (5.77M rows, 64 PCA dims, float32, 90MB, 0 NaN)
  - PCA embedding columns are NUMERIC (float32) and survive `select_dtypes(include=[np.number])`
  - Phase 12 text reruns LANDED real core_text/full results for all valid models
  - cfisch has GROUP access to npin's HF cache (chmod g+rx fix applied 2026-03-15)
- Phase 15 new TSLib model expansion:
  - 23 new models submitted (2026-03-16): CARD, CFPT, DeformableTST, DUET, FiLM, FilterTS, FreTS, Fredformer, MICN, ModernTCN, NonstationaryTransformer, PDF, PIR, PathFormer, SCINet, SEMPO, SRSNet, SegRNN, SparseTSF, TimeBridge, TimePerceiver, TimeRecipe, xPatch
  - 8 encoder-only forward(x) models patched: DeformableTST, Fredformer, ModernTCN, PDF, PathFormer, SparseTSF, TimeRecipe, xPatch
  - 5 excluded (structural): Koopa (NaN divergence), CycleNet/TQNet (cycle_index), Mamba (mamba_ssm), TiRex (NX-AI tirex not on PyPI)
  - **8 excluded (Finding H, 2026-03-22)**: CFPT, DeformableTST, MICN(already G), PathFormer, SEMPO, SparseTSF, TimeBridge, TimePerceiver — all 100% constant predictions, 0% fairness pass
  - **15 valid P15 models**: CARD, DUET, FiLM, FilterTS, FreTS, Fredformer, ModernTCN, NonstationaryTransformer, PDF, PIR, SCINet, SRSNet, SegRNN, TimeRecipe, xPatch
  - Code commit: `e177f6f` (encoder-only forward fix)
  - Bug-fix commits: `c4d214e` (6 bugs), `1185617` (n_vars fix), `0373037` (fix11)
  - Seed2 harness fix: commit `a9162c2`
  - accel_v2 optimization: 23 working models only (removed 10 broken/excluded), ~30% faster per job
  - accel_v2 submitted (2026-03-22 02:00): 17 GPU + 17 L40S + 17 Hopper = 51 optimized jobs
  - 8 old v1 jobs cancelled (2026-03-22 11:40): 5 l40s + 3 hopper (contained broken models, near timeout)
  - Admin Julien approved l40s/hopper usage ("as you have a rationale for this bigger cpu/gpu ratio this is not waste")
  - Partition limits verified (sinfo 2026-03-22): L40S=2 nodes/32c/503G (max 4 jobs), Hopper=1 node/112c/201G (max 1 job)
  - Total active (2026-03-26 10:19): npin 30R+27PD = 57 jobs
  - GPU: 23 RUNNING (17 g2_ac + 5 af739 + 1 gpu_cos2_t2)
  - L40S: 4 RUNNING + 13 PENDING
  - Hopper: 3 RUNNING (h2_ac_t1_e2/fu/s2 on iris-197) + 14 PENDING
  - g2_ac progress: task3 near-complete; task1/task2 47-77% done, all stuck on ModernTCN bottleneck
  - e2 ablation is sole source of recent record growth (+54)
  - Auto-requeue fix: trap handler in all scripts. Second-round (resubmitted) jobs auto-requeue correctly.
  - ModernTCN bottleneck: 20M params, 30min/epoch — all non-e2 accel jobs stuck on this model.

## Canonical Directories

- Results: `runs/benchmarks/block3_phase9_fair/`
- Text embeddings: `runs/text_embeddings/`
- Deprecated outputs archive: `runs/benchmarks/_deprecated_archive/`
- Documentation archive: `docs/_legacy_repo/`
- Reference-only research notes: `docs/references/`

## Hard Constraints

- Only commit/push: `scripts/`, `src/`, `configs/`, `docs/`
- Never commit anything under `runs/`
- Never modify files under `runs/*_20260203_225620/`
- Block 3 must read the freeze via `FreezePointer` and `docs/audits/FULL_SCALE_POINTER.yaml`
- Do not use V734-V738 code paths, tables, or narratives as active baselines

## Current Interpretation Rules

1. Phase 7 / Phase 8 are historical only.
2. The old V72 / early V73 truth-pack line is historical only.
3. `docs/_legacy_repo/` is archive material, not current operational truth.
4. `docs/references/` is useful background knowledge, not status truth.
5. Current Phase 9 results still represent:
   - `core_only`
   - `core_only_seed2`
   - `core_edgar`
   - `core_edgar_seed2`
   until the real text-enabled reruns land.
6. `docs/PHASE9_V739_FACT_ALIGNMENT.md` defines the corrected statements that future contributors must use when old/legacy claims conflict with current evidence.

## Key Current Scripts

- Freeze verification: `scripts/block3_verify_freeze.py`
- Benchmark harness: `scripts/run_block3_benchmark_shard.py`
- Results aggregation: `scripts/aggregate_block3_results.py`
- Current-state fact snapshot: `scripts/build_phase9_current_snapshot.py`
- Phase 12 rerun preparation: `scripts/phase12_prepare_text_rerun.py`
- Dataset interface: `src/narrative/data_preprocessing/block3_dataset.py`
- Registry: `src/narrative/block3/models/registry.py`
- Current AutoFit implementation: `src/narrative/block3/models/nf_adaptive_champion.py`

## Immediate Next Work

1. ~~Land the first clean V739 results.~~ ✅ V739: 120/160 conditions landed.
2. ~~Finish gap-fill for the remaining 10 partial TSLib models~~ ✅ Covered by accel_v2 scripts.
3. ~~Run and land the real text-enabled reruns~~ ✅ Phase 12 COMPLETE.
4. ~~Integrate 23 new TSLib models into benchmark~~ ✅ Phase 15 submitted. 15 valid + 8 excluded (Finding H).
5. ~~Submit targeted reruns for 5 failed models~~ ✅ Fix11 GPU scripts completed.
6. ~~Complete V739 s2/e2 gap-fill~~ ⏳ V739@132/160, 5R af739. ~3-5 conds per 2d.
7. ~~Complete ml_tabular t1_fu fix~~ ❌ OOM at all memory levels. XGBoost@159, XGBoostPoisson@157 — structural limit.
8. ~~Fix HPC admin complaints~~ ✅ All jobs migrated to 7-8 CPUs.
9. ~~Chronos2+TTM seed2 gap-fill~~ ✅ All 5 gpu_fnd scripts COMPLETED.
10. ~~Finding H: identify broken P15 models~~ ✅ 8 models excluded, accel_v2 scripts created.
11. ~~Cancel old v1 jobs with broken models~~ ✅ 8 old v1 jobs cancelled (5 l40s + 3 hopper). Freed L40S for v2.
12. Wait for 57 active jobs to complete (30R + 27PD across gpu+l40s+hopper).
13. s2 gap-fill: gpu_cos2_t2 running with trimmed 23-model list.
14. e2 gap: ETSformer/LightTS/Pyraformer/Reformer have 0/28 e2 — accel_v2 covers this.
15. g2_ac_v2 auto-requeue: second round resubmitted (2026-03-24). Trap handler NOW active. Will auto-requeue on future TIMEOUT.
16. Only then start any V740+ iteration.
