# Agent Context

## Mission
Block 3 modeling on the finalized WIDE2 freeze (`TRAIN_WIDE_FINAL`). The freeze is complete with all gates PASS. No feature backfilling is required.

## Current State (as of 2026-02-13)
- **WIDE2 Freeze: COMPLETE** (stamp `20260203_225620`)
- All 5 gates PASS:
  - `pointer_valid`: PASS
  - `column_manifest`: PASS
  - `raw_cardinality_coverage`: PASS
  - `freeze_candidates`: PASS
  - `offer_day_coverage_exact`: PASS
- Block 3 entry verified via `scripts/block3_verify_freeze.py`

### Phase 7 Benchmark — ULHPC Iris HPC (IN PROGRESS)
- **Model Registry**: **67 models across 7 categories**
  - `ml_tabular` (15): LogisticRegression, Ridge, Lasso, ElasticNet, SVR, KNN, RandomForest, ExtraTrees, HistGradientBoosting, LightGBM, XGBoost, CatBoost, QuantileRegressor, SeasonalNaive, MeanPredictor
  - `statistical` (5): AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive
  - `deep_classical` (4): NBEATS, NHITS, TFT, DeepAR
  - `transformer_sota` (20): PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx, BiTCN, KAN, RMoK, SOFTS, StemGNN (shard A: 10, shard B: 10)
  - `foundation` (11): Chronos, ChronosBolt, Chronos2, Moirai, MoiraiLarge, Moirai2, Timer, TimeMoE, MOMENT, LagLlama, TimesFM (3 shards: Chronos/Moirai/HF)
  - `irregular` (2): GRU-D, SAITS
  - `autofit` (10): V1, V2, V2E, V3, V3E, V3Max, V4, V5, V6, V7 (2 shards: af1/af2)
- **Platform**: ULHPC Iris — GPU (4×V100 32GB, 756GB), Batch (28c, 112GB), QOS `iris-*-long` (14-day wall)
- **Ablations**: 4 per task — `core_only`, `core_text`, `core_edgar`, `full`
- **Tasks**: `task1_outcome` (3 targets), `task2_forecast` (2 targets, 4 horizons), `task3_risk_adjust` (2 targets)
- **Output Dir**: `runs/benchmarks/block3_20260203_225620_phase7/`
- **Progress**: 23/121 shards complete, 781 metric records
  - 12 RUNNING, 96 PENDING
  - ml_tabular: 10/12 done (all task1+task2 ablations complete, task3 ce/fu pending)
  - statistical: 2/11 done (core_only: t2, t3 complete; t1_co RUNNING; ct/ce/fu 8 tasks pending with OOM fix applied)
  - deep_classical: 6/12 done (t1: co/ct/ce done, t2: co/ct done, t3: co done; fu pending)
  - transformer_sota A: 4/12 done (t1: co/ct running+done; t2: co/ct running; ce/fu pending)
  - transformer_sota B, foundation×3, irregular: all core_only done in Phase 1; Phase 7 ce/fu pending
  - autofit shard 1: 3/12 done (t1-t3 co done; t1-t3 ct RUNNING; ce/fu pending)
  - autofit shard 2: 0/12 done (t1 co/ct just started RUNNING)
- **Issues Fixed**:
  - EDGAR timezone bug (merge_asof dtype mismatch) — fixed in `ae9626b`
  - Statistical OOM 64G→112G — 6 PENDING jobs cancelled, 8 scripts updated, resubmitted
  - 4 OOM failures (sta_t1_ct, sta_t1_fu, sta_t2_ct, sta_t2_fu) resolved
- **Live Results**: See [docs/BLOCK3_RESULTS.md](docs/BLOCK3_RESULTS.md)
- **Full Status**: See [docs/BLOCK3_MODEL_STATUS.md](docs/BLOCK3_MODEL_STATUS.md)

## Hard Constraints
- Only commit/push: `scripts/`, `src/`, `configs/`, `docs/`, except patterns below.
- Never commit anything under `runs/`.
- Never create/track files whose **names** contain: `cursor`, `prompt`, `recovery`, `runbook`, `decision`, `transcript` (case-insensitive).
- All tracked files must be **English** only.
- Block 3 must read **only** `docs/audits/FULL_SCALE_POINTER.yaml` (via `FreezePointer` class).
- **Freeze artifacts are read-only**: never modify files under `runs/*_20260203_225620/`.

## Mandatory Execution Contract
- Read and comply with `docs/BLOCK3_EXECUTION_CONTRACT.md` before every submission/execution path.
- Run `python3 scripts/assert_block3_execution_contract.py --entrypoint <script>` before benchmark execution or job submission.
- Runtime must be insider-only (`python >= 3.11`, insider interpreter path, no base env execution).
- If contract assertion fails, stop and do not submit jobs.

## Block 3 Architecture

### Data Flow
```
docs/audits/FULL_SCALE_POINTER.yaml
        │
        ▼
FreezePointer (src/narrative/data_preprocessing/block3_dataset.py)
        │
        ├─> offers_core_daily (parquet)
        ├─> offers_text (parquet)
        └─> edgar_feature_store (parquet)
                │
                ▼
        Block3Dataset (lazy loading + explicit joins)
                │
                ▼
        BenchmarkHarness (scripts/run_block3_benchmark_shard.py)
                │
                ├─> Statistical (5):       AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive
                ├─> ML Tabular (15):       LightGBM, XGBoost, CatBoost, RandomForest, ...
                ├─> Deep Classical (4):    NBEATS, NHITS, TFT, DeepAR
                ├─> Transformer SOTA (20): PatchTST, iTransformer, TimesNet, TSMixer, Informer, ...
                ├─> Foundation (11):       Chronos, ChronosBolt, Moirai, MoiraiLarge, Timer, ...
                ├─> Irregular (2):         GRU-D, SAITS
                └─> AutoFit (10):          V1–V7 (ensemble model selectors)
```

### Join Keys
- `entity_id` + `crawled_date_day`: offers_core ↔ offers_text
- `cik` + `crawled_date_day`: offers_core ↔ edgar_store

### AutoFit Meta-Features
From `scripts/block3_profile_data.py`:
- `nonstationarity_score`, `periodicity_score`, `multiscale_score`
- `long_memory_score`, `irregular_score`, `heavy_tail_score`
- `exog_strength`, `edgar_strength`, `text_strength`, `missing_rate`

## Key Scripts (Block 3)
- Freeze verification: `scripts/block3_verify_freeze.py`
- Data profiling: `scripts/block3_profile_data.py`
- Benchmark harness (shard): `scripts/run_block3_benchmark_shard.py`
- SLURM submission (master): `scripts/submit_phase7_full_benchmark.sh`
- Results aggregator: `scripts/aggregate_block3_results.py`
- Results consolidator: `scripts/consolidate_block3_results.py`
- Paper tables: `scripts/make_paper_tables_v2.py`
- Dataset interface: `src/narrative/data_preprocessing/block3_dataset.py`
- AutoFit composer: `src/narrative/auto_fit/rule_based_composer.py`
- Concept bottleneck: `src/narrative/explainability/concept_bottleneck.py`

## Model Source Files (Block 3)
- Deep + Transformer + Foundation: `src/narrative/block3/models/deep_models.py`
- Statistical (StatsForecast): `src/narrative/block3/models/statistical.py`
- Irregular (PyPOTS): `src/narrative/block3/models/irregular_models.py`
- Traditional ML (sklearn): `src/narrative/block3/models/traditional_ml.py`
- AutoFit wrapper: `src/narrative/block3/models/autofit_wrapper.py`
- Unified registry: `src/narrative/block3/models/registry.py`
- Base classes: `src/narrative/block3/models/base.py`

## Block 3 Configuration
- Config file: `configs/block3.yaml`
- Targets: `total_amount_sold`, `number_investors`, `days_to_close`
- Horizons: [7, 14, 30, 60]
- Context lengths: [30, 60, 90]
- Metrics: RMSE, MAE, MAPE, SMAPE, CRPS (probabilistic)

## Commit Trail
- `ae9626b` Fix EDGAR join: strip timezone from datetime64[ns,UTC] for merge_asof compatibility
- `444f376` Phase 7: Fix entity coverage, RobustFallback, hybrid predict, EDGAR covariates, AutoFit target_transform
- `d837828` Phase 7: SLURM submission scripts for full 67-model benchmark
- `faafdcf` AutoFit V7: data-adapted robust ensemble with 6 SOTA innovations
- `ad07032` AutoFit V6: conference-grade stacked generalization (Phase 6)
- `dce0ff9` AutoFit V5: empirical regime-aware ensemble + 5 new foundation models (65 total)
- `c53abf6` Phase 4: +10 SOTA models (59 total), AutoFitV4 w/ target-transform+NCL+full-OOF
- `320c314` Phase 3: Fix 6 critical issues, 42 SLURM jobs submitted
- `87baa13` Phase 2 AutoFit: 5-fold temporal CV + stability penalty, benchmark fixes
- `014ac92` Fix n_series dynamic computation, SLURM mem 128G, 4090 launch scripts
- `fcbe970` Model registry rewrite: 44 models, panel data fix, all 6 categories
- `3ce5509` WIDE2 freeze seal complete, all gates PASS

## Pending Work (Block 3)
1. ✅ Freeze verification script
2. ✅ Unified dataset interface
3. ✅ Data profiling for AutoFit
4. ✅ Benchmark harness with 6 baseline categories
5. ✅ AutoFit rule-based composer
6. ✅ Concept bottleneck for interpretability
7. ✅ Model registry expansion: 67 models across 7 categories (Phase 4-7)
8. ✅ Panel data fix: all categories receive entity-panel kwargs
9. ✅ Deep/Transformer models: 19 NeuralForecast models (panel-aware)
10. ✅ Statistical models: entity-sampled panel via StatsForecast
11. ✅ Foundation models: 11 models (Chronos family + Moirai family + Timer/TimeMoE/MOMENT/LagLlama/TimesFM)
12. ✅ Irregular models: GRU-D + SAITS via PyPOTS
13. ✅ AutoFit V1-V7: 10 ensemble selection strategies
14. ✅ Benchmark harness updated for all 7 categories + 4 ablations
15. ✅ Phase 7 code fixes (5 root causes across 4 files, 57/57 tests pass)
16. ✅ EDGAR timezone fix (merge_asof dtype mismatch)
17. ✅ Statistical OOM fix (64G → 112G memory)
18. ⏳ Phase 7 full benchmark run on Iris HPC — 23/121 shards done, 781 records
19. ⏳ AutoFit model selection based on profile
20. ⏳ TCAV-style concept importance analysis
21. ⏳ Results leaderboard + paper tables
