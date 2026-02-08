# Block 3 Model Benchmark Status

> Last Updated: 2026-02-08 UTC
> Freeze Stamp: `20260203_225620`
> Model Registry: **44 models across 6 categories**

## Executive Summary

| Category | Models | Count | Panel-Aware | Status |
|----------|--------|-------|-------------|--------|
| **statistical** | AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive | 5 | ✅ Entity-sampled (50 entities) | 🔧 Ready to run |
| **ml_tabular** | LogisticRegression, Ridge, … MeanPredictor | 15 | N/A (tabular) | 🔧 Ready to run |
| **deep_classical** | NBEATS, NHITS, TFT, DeepAR | 4 | ✅ Entity-sampled (200 entities) | 🔧 Ready to run |
| **transformer_sota** | PatchTST, iTransformer, … StemGNN | 15 | ✅ Entity-sampled (200 entities) | 🔧 Ready to run |
| **foundation** | Chronos, Moirai, TimesFM | 3 | ✅ Entity contexts (200 entities) | 🔧 Ready to run |
| **irregular** | GRU-D, SAITS | 2 | ✅ 3-D masked panel (100 entities) | 🔧 Ready to run |
| **TOTAL** | | **44** | | |

---

## Architecture (Post-Rewrite 2026-02-08)

### Model Source Files
| File | Category | Models | Backend |
|------|----------|--------|---------|
| `src/narrative/block3/models/deep_models.py` | deep_classical + transformer_sota + foundation | 22 | NeuralForecast 3.1.4 / Chronos / Moirai / TimesFM |
| `src/narrative/block3/models/statistical.py` | statistical | 5 | StatsForecast (Nixtla) |
| `src/narrative/block3/models/irregular_models.py` | irregular | 2 | PyPOTS |
| `src/narrative/block3/models/traditional_ml.py` | ml_tabular | 15 | sklearn / LightGBM / XGBoost / CatBoost |
| `src/narrative/block3/models/registry.py` | ALL | 44 | Unified registry |
| `src/narrative/block3/models/base.py` | — | — | ModelBase, ModelConfig |

### Panel Data Strategy
All panel-aware categories use entity-sampled panel construction:
- Filter entities with ≥ 20 observations
- Random sample up to MAX_ENTITIES (200 for deep/transformer, 50 for statistical, 100 for irregular)
- Build NeuralForecast-style panel: `unique_id / ds / y`
- Falls back to synthetic panel from flat y if `train_raw` unavailable

### Benchmark Harness
`scripts/run_block3_benchmark_shard.py` passes `train_raw`, `target`, `horizon` kwargs to **all** panel-aware categories:
`deep_classical`, `transformer_sota`, `foundation`, `statistical`, `irregular`

---

## Dependencies (Verified on 4090, insider env)

| Package | Version | Status |
|---------|---------|--------|
| PyTorch | 2.7.1+cu128 | ✅ (2× GPU detected) |
| NeuralForecast | 3.1.4 | ✅ (19 models) |
| StatsForecast | 2.0.3 | ✅ (5 models) |
| chronos | 2.2.2 | ✅ |
| uni2ts (Moirai) | installed | ✅ |
| timesfm | NOT installed | ❌ (no pip package) |
| pypots | installed | ✅ (GRU-D + SAITS) |
| scikit-learn | 1.8.0 | ✅ |
| lightgbm | 4.6.0 | ✅ |
| xgboost | 2.1.4 | ✅ |
| catboost | 1.2.8 | ✅ |

---

## Detailed Model Registry

### 1. Statistical Models (5) — StatsForecast

| Model | Panel Support | Notes |
|-------|---------------|-------|
| AutoARIMA | ✅ 50 entities | Automatic ARIMA selection |
| AutoETS | ✅ 50 entities | Exponential smoothing |
| AutoTheta | ✅ 50 entities | Theta method |
| MSTL | ✅ 50 entities | Multi-seasonal decomposition |
| SF_SeasonalNaive | ✅ 50 entities | Seasonal baseline |

### 2. ML Tabular Models (15) — sklearn / GBDT

| Model | Notes |
|-------|-------|
| LogisticRegression | Classification |
| Ridge | L2 regression |
| Lasso | L1 regression |
| ElasticNet | L1+L2 |
| SVR | Support vector |
| KNN | K-nearest neighbors |
| RandomForest | Ensemble |
| ExtraTrees | Extremely randomized trees |
| HistGradientBoosting | Native histogram GBM |
| LightGBM | Microsoft GBDT |
| XGBoost | XGBoost GBDT |
| CatBoost | Yandex GBDT |
| QuantileRegressor | Probabilistic |
| SeasonalNaive | Baseline |
| MeanPredictor | Baseline |

### 3. Deep Classical Models (4) — NeuralForecast

| Model | Paper | Panel Support |
|-------|-------|---------------|
| NBEATS | Oreshkin et al., 2019 | ✅ 200 entities |
| NHITS | Challu et al., 2022 | ✅ 200 entities |
| TFT | Lim et al., 2021 | ✅ 200 entities |
| DeepAR | Salinas et al., 2020 | ✅ 200 entities |

### 4. Transformer SOTA Models (15) — NeuralForecast

| Model | Paper | Panel Support |
|-------|-------|---------------|
| PatchTST | Nie et al., ICLR 2023 | ✅ 200 entities |
| iTransformer | Liu et al., ICLR 2024 | ✅ 200 entities |
| TimesNet | Wu et al., ICLR 2023 | ✅ 200 entities |
| TSMixer | Chen et al., TMLR 2023 | ✅ 200 entities |
| Informer | Zhou et al., AAAI 2021 | ✅ 200 entities |
| Autoformer | Wu et al., NeurIPS 2021 | ✅ 200 entities |
| FEDformer | Zhou et al., ICML 2022 | ✅ 200 entities |
| VanillaTransformer | Vaswani et al., 2017 | ✅ 200 entities |
| TiDE | Das et al., TMLR 2023 | ✅ 200 entities |
| NBEATSx | Olivares et al., 2022 | ✅ 200 entities |
| BiTCN | — | ✅ 200 entities |
| KAN | Liu et al., 2024 | ✅ 200 entities |
| RMoK | — | ✅ 200 entities |
| SOFTS | — | ✅ 200 entities |
| StemGNN | Cao et al., NeurIPS 2020 | ✅ 200 entities |

### 5. Foundation Models (3)

| Model | Provider | Status |
|-------|----------|--------|
| Chronos | Amazon | ✅ chronos-t5-small |
| Moirai | Salesforce | ✅ moirai-1.1-R-small |
| TimesFM | Google | ❌ Not installed |

### 6. Irregular Models (2) — PyPOTS

| Model | Paper | Panel Support |
|-------|-------|---------------|
| GRU-D | Che et al., 2018 | ✅ 100 entities, masked |
| SAITS | Du et al., 2023 | ✅ 100 entities, masked |

---

## Data Characteristics

| Metric | Value |
|--------|-------|
| Total rows | 5,553,820 |
| Train split | 4,421,931 |
| Val split | 575,246 |
| Test split | 556,643 |
| Entities | 20,944 |
| Avg time span | ~210 days per entity |
| Panel structure | High N (entities), moderate T (time) |

---

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Targets | `total_amount_sold`, `number_investors`, `days_to_close` |
| Tasks | task1_outcome, task2_forecast, task3_risk_adjust |
| Horizons | [7, 14, 30, 60] |
| Ablations | core_only, core_text, core_edgar, full |
| Metrics | MAE, RMSE, MAPE, SMAPE, CRPS |

---

## Pending

1. ⏳ Full benchmark run on 4090 (2× RTX 4090, 24GB each)
2. ⏳ Full benchmark run on 3090 (2× RTX 3090)
3. ⏳ Results leaderboard + paper LaTeX tables
4. ⏳ AutoFit model selection based on data profile
5. ⏳ TCAV-style concept importance analysis
