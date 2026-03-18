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

## Verified Current State (2026-03-18)

- Freeze: complete and read-only
- Canonical benchmark: `runs/benchmarks/block3_phase9_fair/`
- Raw benchmark scan:
  - `14418` raw records
  - `114` raw models (98 after audit exclusion of 16 models)
  - `69` complete models (`=160` records)
  - `45` partial models (various stages)
  - `17` valid conditions (task3 has no core_only_seed2)
  - `160` max records per model: t1(72) + t2(48) + t3(40)
  - `87%` task redundancy: overlapping cells across tasks produce identical MAE
  - `~72` truly unique evaluation cells (not 160)
- AutoFit:
  - V734-V738 are retired due to oracle test-set leakage
  - V739 is the only valid current AutoFit baseline
  - V739 landed conditions: `112/160` (missing s2/e2 = seed2+edgar_seed2)
  - V739 s2/e2 gap-fill: RESUBMITTED after seed2 harness fix (5 gpu jobs, 2026-03-18)
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
- Model registry: `docs/references/MODEL_REGISTRY.md` — 152 registered, 117 active, 114 in benchmark
- Model completion tiers:
  - @160 (complete): 69 models (statistical 15, foundation 14, irregular 4, deep_classical 9, transformer_sota 22, tslib_sota 5)
  - @157: 10 ml_tabular models (missing 3 t1_fu records each, split-model job 5263582 PENDING)
  - @112: AutoFitV739 (missing s2/e2, resubmitted after harness fix)
  - @111: ETSformer/LightTS/Pyraformer/Reformer (covered by ALL33 accel scripts)
  - @105: Crossformer/MSGNet/MambaSimple/MultiPatchFormer/PAttn/TimeFilter (covered by ALL33 accel)
  - @29: ModernTCN (covered by ALL33 accel)
  - @23: 22 P15 new models (running across gpu/hopper/l40s)
  - @21: NegativeBinomialGLM (structural failure, excluded)
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
  - 12 jobs: npin gpu (co/ce × 3 tasks) + cfisch gpu (ct/fu × 3 tasks)
  - Scripts: `.slurm_scripts/phase15/`
  - Code commit: `e177f6f` (encoder-only forward fix)
  - Bug-fix audit (2026-03-16): 7 bugs total — FilterTS filter_type+embedding, DUET noisy_gating/num_experts/k, PathFormer gpu/num_nodes, SEMPO tuple return, DeformableTST timm+n_vars
  - Bug-fix commits: `c4d214e` (6 bugs, 13:53 CET), `1185617` (n_vars fix, 14:24 CET)
  - CFPT: 28176 batch errors per job (conv2d channel mismatch, channels=1 vs 5) — produces MAE but quality suspect
  - Running jobs (3 npin): CARD ✅, CFPT ⚠️, FiLM ✅, FreTS ⏳ (DeformableTST/DUET/FilterTS errored; PathFormer/SEMPO will also fail on old code)
  - Targeted rerun scripts: `.slurm_scripts/phase15/p15_rerun_errors_*.sh` for 5 errored models × 3 conditions
  - Fix11 rerun: all 11 model fixes landed (commit `0373037`), 3 gpu + 2 hopper scripts running
  - ModernTCN freq fix: commit `7e023e4`, 7 dedicated jobs all failed (OOM/TIMEOUT), covered by ALL33 accel
  - ALL33 acceleration: 39 scripts (27 npin + 12 cfisch) across gpu/hopper/l40s
  - Gap-fill round 2: CATS/FITS/KANAD/WPMixer ct/fu COMPLETED ✅ (6 jobs, 1-3h each) → 4 models now @160
  - Seed2 harness fix: commit `a9162c2` — added `core_only_seed2`/`core_edgar_seed2` to ABLATION_NAMES + `_SEED2_BASE` mapping
  - cos2 scripts: 6 new scripts in `.slurm_scripts/phase15/cos2/` covering ALL33 × t1/t2 × gpu/hp/l40
  - Total active: 85 jobs (npin 64 + cfisch 21), ~26 GPUs computing

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
2. ~~Finish gap-fill for the remaining 10 partial TSLib models~~ ⏳ Covered by ALL33 accel scripts (39 jobs running).
3. ~~Run and land the real text-enabled reruns~~ ✅ Phase 12 COMPLETE (48/48 + 6 cfisch TIMEOUT). CATS/FITS/KANAD/WPMixer ct/fu gap-fill resubmitted.
4. ~~Integrate 23 new TSLib models into benchmark~~ ✅ Phase 15 submitted (12 jobs, 2026-03-16). Fix11 + ALL33 accel covering all conditions.
5. ~~Submit targeted reruns for 5 failed models~~ ✅ Fix11 scripts running (commit `0373037`).
6. ~~Complete V739 s2/e2 gap-fill~~ ⏳ Resubmitted after seed2 harness fix (5 jobs, 2026-03-18).
7. ~~Complete ml_tabular t1_fu fix~~ ⏳ Resubmitted on bigmem 640G (job 5262824, running).
8. Wait for all 85 active jobs to land, then consolidate full benchmark surface.
9. Only then start any V740+ iteration.
