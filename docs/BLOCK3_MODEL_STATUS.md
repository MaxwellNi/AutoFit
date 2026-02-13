# Block 3 Model Benchmark Status

> Last Updated: 2026-02-13 02:15 UTC
> Freeze Stamp: `20260203_225620`
> Model Registry: **67 models across 7 categories**
> Phase 7 Benchmark: **IN PROGRESS** (23/121 shards, 781 records, 67 models registered)
> Platform: ULHPC Iris HPC (GPU V100 + Batch 112GB)

## Executive Summary

| Category | Models | Count | Panel-Aware | Phase 7 Status |
|----------|--------|-------|-------------|----------------|
| **ml_tabular** | LogisticRegression, Ridge, … MeanPredictor | 15 | N/A (tabular) | 7/11 done, 4 pending |
| **statistical** | AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive | 5 | ✅ Entity-sampled | 2/11 done, 1 running, 8 pending (OOM fixed) |
| **deep_classical** | NBEATS, NHITS, TFT, DeepAR | 4 | ✅ 2000 entities + Ridge fallback | 6/11 done, 5 pending |
| **transformer_sota A** | PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx | 10 | ✅ 2000 entities + Ridge fallback | 4/11 done, 4 running, 3 pending |
| **transformer_sota B** | BiTCN, KAN, RMoK, SOFTS, StemGNN | 5+(shard) | ✅ 200 entities (n_series) | 0/11 done, 11 pending |
| **foundation Chronos** | Chronos, ChronosBolt, Chronos2 | 3 | ✅ Entity contexts | 0/11 done, 11 pending |
| **foundation Moirai** | Moirai, MoiraiLarge, Moirai2 | 3 | ✅ Entity contexts | 0/11 done, 11 pending |
| **foundation HF** | Timer, TimeMoE, MOMENT, LagLlama, TimesFM | 5 | ✅ Entity contexts | 0/11 done, 11 pending |
| **irregular** | GRU-D, SAITS | 2 | ✅ 3-D masked panel (1000 entities) | 1/11 done, 10 pending |
| **autofit shard1** | V1, V2, V2E, V3, V3E | 5 | Meta-learner | 3/11 done, 5 running, 3 pending |
| **autofit shard2** | V3Max, V4, V5, V6, V7 | 5 | Meta-learner | 0/11 done, 2 running, 9 pending |
| **TOTAL** | | **67** | | **23/121 shards (19%)** |

---

## Architecture (Post-Phase 7 Rewrite 2026-02-12)

### Model Source Files
| File | Category | Models | Backend |
|------|----------|--------|---------|
| `src/narrative/block3/models/deep_models.py` | deep_classical + transformer_sota + foundation | 35 | NeuralForecast 3.1.4 / Chronos / Moirai / Timer / etc |
| `src/narrative/block3/models/statistical.py` | statistical | 5 | StatsForecast (Nixtla) |
| `src/narrative/block3/models/irregular_models.py` | irregular | 2 | PyPOTS |
| `src/narrative/block3/models/traditional_ml.py` | ml_tabular | 15 | sklearn / LightGBM / XGBoost / CatBoost |
| `src/narrative/block3/models/autofit_wrapper.py` | autofit | 10 | Meta-learner ensemble |
| `src/narrative/block3/models/registry.py` | ALL | 67 | Unified registry |
| `src/narrative/block3/models/base.py` | — | — | ModelBase, ModelConfig |

### Panel Data Strategy (Phase 7 — updated from Phase 3)
- **Deep/Transformer (non-n_series)**: 2000 entities, min 10 obs, Ridge fallback for unseen
- **Transformer (n_series models)**: 200 entities, min 10 obs
- **Statistical**: Entity-sampled panel via StatsForecast
- **Irregular**: 1000 entities, 3-D masked panel
- **RobustFallback**: Catches all NeuralForecast exceptions, auto-falls back to Ridge regression
- **Hybrid predict**: Ridge regression on features for test entities not seen during training

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

### 4. Transformer SOTA Models (20) — NeuralForecast (2 shards)

**Shard A (10 models)**:

| Model | Paper | Panel Support |
|-------|-------|---------------|
| PatchTST | Nie et al., ICLR 2023 | ✅ 2000 entities |
| iTransformer | Liu et al., ICLR 2024 | ✅ 200 entities (n_series) |
| TimesNet | Wu et al., ICLR 2023 | ✅ 2000 entities |
| TSMixer | Chen et al., TMLR 2023 | ✅ 200 entities (n_series) |
| Informer | Zhou et al., AAAI 2021 | ✅ 2000 entities |
| Autoformer | Wu et al., NeurIPS 2021 | ✅ 2000 entities |
| FEDformer | Zhou et al., ICML 2022 | ✅ 2000 entities |
| VanillaTransformer | Vaswani et al., 2017 | ✅ 2000 entities |
| TiDE | Das et al., TMLR 2023 | ✅ 2000 entities |
| NBEATSx | Olivares et al., 2022 | ✅ 2000 entities |

**Shard B (10 models)**:

| Model | Paper | Panel Support |
|-------|-------|---------------|
| BiTCN | — | ✅ 2000 entities |
| KAN | Liu et al., 2024 | ✅ 2000 entities |
| RMoK | — | ✅ 200 entities (n_series) |
| SOFTS | — | ✅ 200 entities (n_series) |
| StemGNN | Cao et al., NeurIPS 2020 | ✅ 200 entities (n_series) |

### 5. Foundation Models (11) — 3 shards

**Chronos Shard (3 models)**:

| Model | Provider | Status |
|-------|----------|--------|
| Chronos | Amazon | ✅ chronos-t5-small |
| ChronosBolt | Amazon | ✅ chronos-bolt-small |
| Chronos2 | Amazon | ✅ |

**Moirai Shard (3 models)**:

| Model | Provider | Status |
|-------|----------|--------|
| Moirai | Salesforce | ✅ moirai-1.1-R-small |
| MoiraiLarge | Salesforce | ✅ moirai-1.1-R-large |
| Moirai2 | Salesforce | ✅ moirai-2-R-small |

**HF Shard (5 models)**:

| Model | Provider | Status |
|-------|----------|--------|
| Timer | — | ✅ |
| TimeMoE | — | ✅ |
| MOMENT | CMU | ✅ |
| LagLlama | — | ✅ |
| TimesFM | Google | ✅ |

### 6. Irregular Models (2) — PyPOTS

| Model | Paper | Panel Support |
|-------|-------|---------------|
| GRU-D | Che et al., 2018 | ✅ 1000 entities, masked |
| SAITS | Du et al., 2023 | ✅ 1000 entities, masked |

### 7. AutoFit Models (10) — 2 shards

**Shard 1 (V1-V5)**:

| Model | Strategy |
|-------|----------|
| V1 | Simple best-of-K selection |
| V2 | 5-fold temporal CV + stability penalty |
| V2E | V2 with ElasticNet candidates |
| V3 | Top-K exhaustive search |
| V3E | V3 with ElasticNet candidates |

**Shard 2 (V3Max-V7)**:

| Model | Strategy |
|-------|----------|
| V3Max | Exhaustive search with time budget |
| V4 | Target-transform + NCL + full-OOF |
| V5 | Empirical regime-aware ensemble |
| V6 | Conference-grade stacked generalization |
| V7 | Data-adapted robust ensemble with 6 SOTA innovations |

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
| Platform | ULHPC Iris HPC (GPU V100 + Batch 112GB) |
| QOS | `iris-*-long` (14-day wall times) |
| Total shards | 121 |

---

## Pending

1. ⏳ Phase 7 full benchmark run — 23/121 shards done (19%), 781 records
2. ⏳ transformer_sota B, foundation×3 — all pending (GPU queue)
3. ⏳ Results consolidation + leaderboard
4. ⏳ Paper LaTeX tables
5. ⏳ AutoFit model selection based on data profile
6. ⏳ TCAV-style concept importance analysis

## Issues Fixed (Phase 7)

| # | Issue | Root Cause | Fix | Status |
|---|-------|-----------|-----|--------|
| 1 | Entity coverage <5% | max_entities=200, min_obs=20 | 2000 entities, min_obs=10, Ridge fallback | ✅ `444f376` |
| 2 | RobustFallback silent failure | Only caught specific exceptions | Catch all, auto-fallback | ✅ `444f376` |
| 3 | Unseen entities → global_mean | No fallback for new entities | Ridge on features | ✅ `444f376` |
| 4 | EDGAR features ignored | Not passed to panel builder | Added `futr_exog_list` | ✅ `444f376` |
| 5 | AutoFit no target transform | Regression targets skewed | Auto log1p/expm1 | ✅ `444f376` |
| 6 | EDGAR timezone dtype | `datetime64[ns,UTC]` mismatch | `tz_convert(None)` | ✅ `ae9626b` |
| 7 | Statistical OOM (64G) | Text/EDGAR joins expand memory | 112G + 28 CPUs | ✅ Session fix |
