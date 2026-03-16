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

## Verified Current State (2026-03-16)

- Freeze: complete and read-only
- Canonical benchmark: `runs/benchmarks/block3_phase9_fair/`
- Raw benchmark scan:
  - `132` metrics files
  - `13407` raw records (Phase 12 text reruns nearly complete)
  - `91` raw models
  - `80` raw complete models (`≥104`)
  - `11` raw partial models
- AutoFit:
  - V734-V738 are retired due to oracle test-set leakage
  - V739 is the only valid current AutoFit baseline
  - V739 landed conditions: `112/112` (ALL COMPLETE — 4 ablations × 3 tasks)
  - V739 benchmark ranking: **#13/80** by mean rank (14.38, top 16% across 56 universal conditions)
  - V739 conditions won: 3/56 (1 per task)
  - V739 quality: 0 NaN/Inf, 0 fallback, 100% fairness pass
- Top-5 models by mean rank: NHITS (4.21), PatchTST (4.36), NBEATS (4.77), NBEATSx (5.84), ChronosBolt (7.11)
- Dominant champion model: NBEATS — 24/56 conditions won (43%)
- Gap-fill progress:
  - 10 TSLib gap-fill jobs RUNNING (2 npin/l40s + 8 cfisch/gpu)
  - FM seed2: ALL 5/5 COMPLETED
  - Chronos2: COMPLETE (≥104), TTM: COMPLETE (≥104)
  - TSLib @80: TimeFilter, MultiPatchFormer, MSGNet, PAttn, MambaSimple, Crossformer (need +36 each)
  - TSLib @94: ETSformer, LightTS, Pyraformer, Reformer (need +22 each)
  - NegativeBinomialGLM: structural failure (21/104), cannot complete
- Phase 12 text reruns:
  - 48+1 total scripts (40 original + 8 t3_ct + 1 OOM fix)
  - 42 COMPLETED, 6 RUNNING (cfisch tslib_sota only)
  - OOM: ml_t_t1_fu crashed at 320G AND 640G, but ALL valid records were saved (only NegBinGLM missing)
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
  - 5 excluded: Koopa (NaN divergence), CycleNet/TQNet (cycle_index), Mamba (mamba_ssm), TiRex (tirex)
  - 12 jobs: npin gpu (co/ce × 3 tasks) + cfisch gpu (ct/fu × 3 tasks)
  - Scripts: `.slurm_scripts/phase15/`
  - Code commit: `e177f6f` (encoder-only forward fix)

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
2. Finish gap-fill for the remaining 10 partial TSLib models (NegBinGLM structural failure = excluded).
3. ~~Run and land the real text-enabled reruns~~ ✅ Phase 12 submitted (48 jobs, 2026-03-15). 42/48 COMPLETED, 6 RUNNING.
4. ~~Integrate 23 new TSLib models into benchmark~~ ✅ Phase 15 submitted (12 jobs, 2026-03-16). All PENDING.
5. Wait for Phase 12 + Phase 15 + gap-fill to land, then consolidate full benchmark surface.
6. Only then start any V740+ iteration.
