# Agent Context

## Mission
Block 3 modeling on the finalized WIDE2 freeze (`TRAIN_WIDE_FINAL`). The freeze is complete with all gates PASS. No feature backfilling is required.

## Current State (as of 2026-02-08)
- **WIDE2 Freeze: COMPLETE** (stamp `20260203_225620`)
- All 5 gates PASS:
  - `pointer_valid`: PASS
  - `column_manifest`: PASS
  - `raw_cardinality_coverage`: PASS
  - `freeze_candidates`: PASS
  - `offer_day_coverage_exact`: PASS
- Block 3 entry verified via `scripts/block3_verify_freeze.py`

### Benchmark Progress (44 Models, 6 Categories)
- **Model Registry Rewrite**: COMPLETE — 44 models across 6 categories
  - `ml_tabular` (15): LogisticRegression, Ridge, Lasso, ElasticNet, SVR, KNN, RandomForest, ExtraTrees, HistGradientBoosting, LightGBM, XGBoost, CatBoost, QuantileRegressor, SeasonalNaive, MeanPredictor
  - `statistical` (5): AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive
  - `deep_classical` (4): NBEATS, NHITS, TFT, DeepAR
  - `transformer_sota` (15): PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx, BiTCN, KAN, RMoK, SOFTS, StemGNN
  - `foundation` (3): Chronos, Moirai, TimesFM
  - `irregular` (2): GRU-D, SAITS
- **Panel Data Fix**: ALL categories now receive `train_raw`/`target`/`horizon` kwargs
- **n_series Fix**: Dynamic `n_series=panel["unique_id"].nunique()` for iTransformer/TSMixer/RMoK/SOFTS/StemGNN (was hardcoded to 1)
- **Dependencies**: NeuralForecast 3.1.4, PyTorch 2.7.1+cu128, Chronos, uni2ts, PyPOTS, StatsForecast
- **4090 Benchmark**: IN PROGRESS — tmux `b3_full`, dual GPU, output `runs/benchmarks/block3_20260203_225620_4090_final/`
  - GPU0: deep_classical + transformer_sota + foundation (task3 in progress)
  - GPU1: ml_tabular (task1 in progress, SVR running)
  - 12/36 shards complete, 336 metric records, 21 models evaluated so far
  - n_series bug: iTransformer/TSMixer/RMoK/SOFTS/StemGNN fallback in this run; re-run script ready
- **3090**: SSH unreachable (ift-severn.cege.ucl.ac.uk timeout)
- **Iris SLURM**: All jobs OOM at 100GB; fixed to 128GB, not yet resubmitted
- **Live Results**: See [docs/BLOCK3_RESULTS.md](docs/BLOCK3_RESULTS.md)
- **Full Status**: See [docs/BLOCK3_MODEL_STATUS.md](docs/BLOCK3_MODEL_STATUS.md)

## Hard Constraints
- Only commit/push: `scripts/`, `src/`, `configs/`, `docs/`, except patterns below.
- Never commit anything under `runs/`.
- Never create/track files whose **names** contain: `cursor`, `prompt`, `recovery`, `runbook`, `decision`, `transcript` (case-insensitive).
- All tracked files must be **English** only.
- Block 3 must read **only** `docs/audits/FULL_SCALE_POINTER.yaml` (via `FreezePointer` class).
- **Freeze artifacts are read-only**: never modify files under `runs/*_20260203_225620/`.

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
                ├─> Transformer SOTA (15): PatchTST, iTransformer, TimesNet, TSMixer, Informer, ...
                ├─> Foundation (3):        Chronos, Moirai, TimesFM
                └─> Irregular (2):         GRU-D, SAITS
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
- Results aggregator: `scripts/aggregate_block3_results.py`
- n_series re-run: `scripts/rerun_nseries_models.sh`
- 4090 dual-GPU launcher: `scripts/run_block3_full_4090.sh`
- tmux conda wrapper: `scripts/launch_b3_tmux.sh`
- Dataset interface: `src/narrative/data_preprocessing/block3_dataset.py`
- AutoFit composer: `src/narrative/auto_fit/rule_based_composer.py`
- Concept bottleneck: `src/narrative/explainability/concept_bottleneck.py`

## Model Source Files (Block 3)
- Deep + Transformer + Foundation: `src/narrative/block3/models/deep_models.py`
- Statistical (StatsForecast): `src/narrative/block3/models/statistical.py`
- Irregular (PyPOTS): `src/narrative/block3/models/irregular_models.py`
- Traditional ML (sklearn): `src/narrative/block3/models/traditional_ml.py`
- Unified registry: `src/narrative/block3/models/registry.py`
- Base classes: `src/narrative/block3/models/base.py`

## Block 3 Configuration
- Config file: `configs/block3.yaml`
- Targets: `total_amount_sold`, `number_investors`, `days_to_close`
- Horizons: [7, 14, 30, 60]
- Context lengths: [30, 60, 90]
- Metrics: RMSE, MAE, MAPE, SMAPE, CRPS (probabilistic)

## Commit Trail
- `014ac92` Fix n_series dynamic computation, SLURM mem 128G, 4090 launch scripts.
- `fcbe970` Model registry rewrite: 44 models, panel data fix, all 6 categories.
- `3ce5509` WIDE2 freeze seal complete, all gates PASS.
- `47a8db3` Add 4090 adapters and wide-freeze updates.
- `1a15ea6` Add mac-hop sync scripts for 4090 workflow.

## Pending Work (Block 3)
1. ✅ Freeze verification script
2. ✅ Unified dataset interface
3. ✅ Data profiling for AutoFit
4. ✅ Benchmark harness with 6 baseline categories
5. ✅ AutoFit rule-based composer
6. ✅ Concept bottleneck for interpretability
7. ✅ Model registry rewrite: 44 models across 6 categories
8. ✅ Panel data fix: all categories receive entity-panel kwargs
9. ✅ Deep/Transformer models: 19 NeuralForecast models (panel-aware)
10. ✅ Statistical models: entity-sampled panel via StatsForecast
11. ✅ Foundation models: Chronos + Moirai + TimesFM wrappers
12. ✅ Irregular models: GRU-D + SAITS via PyPOTS
13. ✅ Benchmark harness updated for all 6 categories
14. ⏳ Full benchmark run on 4090 (dual GPU) — 3 tasks × 6 categories × 44 models
15. ⏳ Full benchmark run on 3090 (dual GPU)
16. ⏳ AutoFit model selection based on profile
17. ⏳ TCAV-style concept importance analysis
18. ⏳ Results leaderboard + paper tables
