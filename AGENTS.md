# Agent Context

## Mission
Block 3 modeling on the finalized WIDE2 freeze (`TRAIN_WIDE_FINAL`). The freeze is complete with all gates PASS. No feature backfilling is required.

## Current State (as of 2026-02-07)
- **WIDE2 Freeze: COMPLETE** (stamp `20260203_225620`)
- All 5 gates PASS:
  - `pointer_valid`: PASS
  - `column_manifest`: PASS
  - `raw_cardinality_coverage`: PASS
  - `freeze_candidates`: PASS
  - `offer_day_coverage_exact`: PASS
- Block 3 entry verified via `scripts/block3_verify_freeze.py`

### Benchmark Progress
- **Paper-Ready**: 2 models (XGBoost MAE=274,952, LightGBM MAE=360,834)
- **Blocked**: Deep/Transformer models using fallback (panel data incompatibility)
- **Not Run**: Statistical (data scale overflow), Foundation (missing deps)
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
        BenchmarkHarness (scripts/run_block3_benchmark.py)
                │
                ├─> Statistical baselines
                ├─> ML Tabular (LightGBM, XGBoost)
                ├─> Deep Classical (N-BEATS, TFT)
                ├─> Transformer SOTA (PatchTST, iTransformer, TimeMixer)
                ├─> GluonTS (DeepAR, WaveNet)
                └─> Foundation (TimesFM, Chronos)
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
- Benchmark harness: `scripts/run_block3_benchmark.py`
- Dataset interface: `src/narrative/data_preprocessing/block3_dataset.py`
- AutoFit composer: `src/narrative/auto_fit/rule_based_composer.py`
- Concept bottleneck: `src/narrative/explainability/concept_bottleneck.py`

## Block 3 Configuration
- Config file: `configs/block3.yaml`
- Targets: `total_amount_sold`, `number_investors`, `days_to_close`
- Horizons: [7, 14, 30, 60]
- Context lengths: [30, 60, 90]
- Metrics: RMSE, MAE, MAPE, SMAPE, CRPS (probabilistic)

## Commit Trail
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
7. ✅ GPU benchmark execution (deep_classical, transformer_sota) - fallback mode
8. ✅ ml_tabular partial run (XGBoost, LightGBM paper-ready)
9. ⚠️ **BLOCKED**: Fix panel data compatibility for deep/transformer models
10. ⚠️ **BLOCKED**: Fix statistical models (data scale overflow)
11. ⏳ Complete ml_tabular for all tasks/ablations
12. ⏳ Foundation models (Chronos, TimesFM)
13. ⏳ AutoFit model selection based on profile
14. ⏳ TCAV-style concept importance analysis
