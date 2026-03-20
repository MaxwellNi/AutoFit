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

## Verified Current State (2026-03-20 12:00)

- Freeze: complete and read-only
- Canonical benchmark: `runs/benchmarks/block3_phase9_fair/`
- Raw benchmark scan (2026-03-20 12:00 rescan):
  - `15212` raw records in metrics.json
  - `137` raw models in benchmark (116 real + 21 retired AutoFit @1 record each)
  - `17` audit-excluded models (see AUDIT_EXCLUDED_MODELS in aggregate_block3_results.py):
    - A: Sundial, TimesFM2, LagLlama, Moirai, MoiraiLarge, Moirai2 (silent fallback)
    - B: AutoCES, xLSTM, TimeLLM, StemGNN, TimeXer (training crash fallback)
    - C: TimeMoE, MOMENT (near-duplicate of Timer)
    - G: MICN, MultiPatchFormer, TimeFilter (constant predictions)
    - Structural: NegativeBinomialGLM (convergence failure, 21 records)
  - `99` active (leaderboard) models = 116 raw - 17 excluded
  - `77` raw models at 160/160, `64` active (leaderboard) models at 160/160
  - `35` incomplete active models
  - `160` max per model: t1(72) + t2(48) + t3(40). task3 has no core_only_seed2.
  - Total unique records: 15015/~18080 = **83.0%** completion
  - `87%` task redundancy: ~72 truly unique evaluation cells
- AutoFit:
  - V734-V738 are retired due to oracle test-set leakage
  - V739 is the only valid current AutoFit baseline
  - V739 landed conditions: `116/160` (missing 44 s2/e2 = seed2+edgar_seed2)
  - V739 s2/e2 gap-fill: af739_t1_s2/t2_s2/t2_e2 RUNNING@2d (5267968-70), af739_t1_e2 RUNNING@1d (5266705, 15h left), af739_t3_e2 PENDING@2d (5267971)
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
- Model completion tiers (2026-03-20):
  - @160 (complete): 64 active + 13 excluded = 77 raw (↑2: Chronos2+TTM DONE)
  - @159: XGBoost (1 missing: t1/full/is_funded — structural OOM, UNFIXABLE)
  - @157: XGBoostPoisson (3 missing: t1/full/h{7,14,30}/is_funded — structural OOM, UNFIXABLE)
  - @116: AutoFitV739 (missing s2/e2 — 5 af739 scripts: 1 running + 4 resubmitted @2d)
  - @113: ETSformer/LightTS/Pyraformer/Reformer (covered by ALL33 accel)
  - @109: Crossformer/MSGNet/MambaSimple/PAttn (covered by ALL33 accel)
  - @54: 21 P15 new models (progressing, ALL33 accel RUNNING)
  - @52: ModernTCN (slower, ALL33 accel RUNNING)
  - NegativeBinomialGLM: excluded (structural failure, 21 records)
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
  - 5 excluded: Koopa (NaN divergence), CycleNet/TQNet (cycle_index), Mamba (mamba_ssm), TiRex (NX-AI tirex not on PyPI)
  - Code commit: `e177f6f` (encoder-only forward fix)
  - Bug-fix commits: `c4d214e` (6 bugs), `1185617` (n_vars fix), `0373037` (fix11)
  - Seed2 harness fix: commit `a9162c2`
  - ALL33 acceleration: 3-partition parallel strategy active (2026-03-20 12:00)
  - Admin Julien approved l40s/hopper usage ("as you have a rationale for this bigger cpu/gpu ratio this is not waste")
  - GPU scripts: 12 accel + 3 fix11 + 3 e2 + 5 cos2/new RUNNING + 4 af739 RUNNING/PENDING
  - L40S scripts: 15 accel (5268065-79), 5 RUNNING + 10 PENDING, QOS=iris-snt
  - Hopper scripts: 15 accel (5268095-109), all PENDING, QOS=besteffort (preemptible)
  - l40s optimization: co/s2→120G(8c), ce/ct/e2/fu→200G(14c). Admin-approved.
  - hopper: co→150G(9c), ce/ct/e2→189G(11c), fu→200G(12c). No iris-hopper QOS available.
  - Total active (2026-03-20 12:00): npin 35R+26PD + cfisch 1R = 62 jobs
  - Chronos2+TTM: COMPLETE @160
  - af739: 3 RUNNING@2d + 1 RUNNING@1d(15h left) + 1 PENDING@2d
  - 4 duplicate af739 cancelled, 15 duplicate l40s cancelled

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

1. ~~Land the first clean V739 results.~~ ✅ V739 COMPLETE: 112/112 conditions landed.
2. ~~Finish gap-fill for the remaining 10 partial TSLib models~~ ✅ Covered by ALL33 accel GPU scripts (20 pending).
3. ~~Run and land the real text-enabled reruns~~ ✅ Phase 12 COMPLETE.
4. ~~Integrate 23 new TSLib models into benchmark~~ ✅ Phase 15 submitted. ALL33 accel + fix11 covering all conditions.
5. ~~Submit targeted reruns for 5 failed models~~ ✅ Fix11 GPU scripts submitted.
6. ~~Complete V739 s2/e2 gap-fill~~ ⏳ Running on gpu (af739_t1_e2 running, 4 resubmitted @2d).
7. ~~Complete ml_tabular t1_fu fix~~ ❌ OOM at all memory levels (200G/320G/640G). XGBoost@159, XGBoostPoisson@157 — structural limit.
8. ~~Fix HPC admin complaints (CPU over-allocation)~~ ✅ All hopper+l40s pending cancelled, all jobs migrated to GPU at 7-8 CPUs.
9. ~~Chronos2+TTM seed2 gap-fill~~ ✅ All 5 gpu_fnd scripts COMPLETED (7-26 min each).
10. Wait for all 62 active jobs to complete (35 running + 26 pending across gpu+l40s+hopper).
11. Only then start any V740+ iteration.
