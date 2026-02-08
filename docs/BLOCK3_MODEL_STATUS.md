# Block 3 Model Benchmark Status

> Last Updated: 2026-02-07 20:00 UTC
> Freeze Stamp: `20260203_225620`
> Git Hash: `5987ce2`

## Current Execution Status

### Platform Status
| Platform | Status | Jobs | Details |
|----------|--------|------|---------|
| **4090 Local** | 🔄 Running | 2 | ml_tabular (fast), foundation (Chronos) |
| **Iris Cluster** | ⏳ Pending | 30 | batch partition, 100GB RAM each |

### Job Progress (4090)
- **ml_tabular**: task1_outcome/core_only - Ridge ✅, Lasso ✅, continuing...
- **foundation**: task1_outcome/core_only - TimesFM ❌ (not installed), Chronos 🔄, Moirai ❌ (not installed)

## Executive Summary

| Category | Total Models | Paper-Ready | Running | Need Fix |
|----------|-------------|-------------|---------|----------|
| **statistical** | 5 | 0 | 0 | 5 (data scale issue) |
| **ml_tabular** | 15 | 2 | 11 | 2 (SVR/KNN slow) |
| **deep_classical** | 4 | 0 | 0 | 4 (panel data issue) |
| **transformer_sota** | 4 | 0 | 0 | 4 (panel data issue) |
| **foundation** | 3 | 0 | 1 | 2 (missing deps) |
| **irregular_aware** | 2 | 0 | 0 | 2 (missing deps) |
| **TOTAL** | **33** | **2** | **12** | **19** |

---

## Detailed Model Status

### Legend
- ✅ **Paper-Ready**: Full results available, can be included in KDD'26 paper
- 🔧 **Code Complete**: Implementation done, but blocked by data/wrapper issues
- ⚠️ **Need Fix**: Code issues or missing dependencies
- 📊 **Runs**: Number of benchmark runs completed

---

### 1. Statistical Models (via StatsForecast)

| Model | Status | Runs | MAE | Issue |
|-------|--------|------|-----|-------|
| AutoARIMA | 🔧 | 0 | - | Data scale: 4.4M rows causes pd.date_range overflow |
| ETS (AutoETS) | 🔧 | 0 | - | Same as above |
| Theta | 🔧 | 0 | - | Same as above |
| MSTL | 🔧 | 0 | - | Same as above |
| SF_SeasonalNaive | 🔧 | 0 | - | Same as above |

**Root Cause**: StatsForecast is designed for few long time series. Our panel data has 20,944 entities × ~210 days = 4.4M rows, which exceeds StatsForecast's design limits.

**Fix Strategy**: Implement per-entity sampling or aggregation in `statistical.py` wrapper.

---

### 2. ML Tabular Models (sklearn + GBDT)

| Model | Status | Runs | MAE (avg) | Notes |
|-------|--------|------|-----------|-------|
| **XGBoost** | ✅ | 3 | **274,952** | Best performer, paper-ready |
| **LightGBM** | ✅ | 3 | **360,834** | Paper-ready |
| CatBoost | 🔧 | 0 | - | Deps OK, not yet run |
| RandomForest | 🔧 | 0 | - | Deps OK, not yet run |
| ExtraTrees | 🔧 | 0 | - | Deps OK, not yet run |
| HistGradientBoosting | 🔧 | 0 | - | Deps OK, not yet run |
| Ridge | 🔧 | 0 | - | Deps OK, not yet run |
| Lasso | 🔧 | 0 | - | Deps OK, not yet run |
| ElasticNet | 🔧 | 0 | - | Deps OK, not yet run |
| SVR | 🔧 | 0 | - | Deps OK, not yet run |
| KNN | 🔧 | 0 | - | Deps OK, not yet run |
| LogisticRegression | 🔧 | 0 | - | Classification only |
| QuantileRegressor | 🔧 | 0 | - | For probabilistic |
| SeasonalNaive | 🔧 | 0 | - | Simple baseline |
| MeanPredictor | 🔧 | 0 | - | Trivial baseline |

**Dependencies**: All installed ✅
- lightgbm: 4.6.0
- xgboost: 2.1.4
- catboost: 1.2.8

---

### 3. Deep Classical Models (via NeuralForecast)

| Model | Status | Runs | MAE | Issue |
|-------|--------|------|-----|-------|
| NBEATS | ⚠️ | 18 | 601,864 [FALLBACK] | Panel data incompatibility |
| NHITS | ⚠️ | 18 | 601,864 [FALLBACK] | Same as above |
| TFT | ⚠️ | 18 | 601,864 [FALLBACK] | Same as above |
| DeepAR | ⚠️ | 18 | 601,864 [FALLBACK] | Same as above |

**Root Cause**: NeuralForecast (Nixtla) models are designed for single/few time series with many observations. The wrapper falls back to mean prediction when n_samples > 50,000.

**Verification**: Models are legitimate Nixtla implementations (neuralforecast 3.1.4), not placeholders.
- N-BEATS: `neuralforecast.models.NBEATS` (Oreshkin et al., 2019)
- N-HiTS: `neuralforecast.models.NHITS` (Challu et al., 2022)
- TFT: `neuralforecast.models.TFT` (Lim et al., 2021)
- DeepAR: `neuralforecast.models.DeepAR` (Salinas et al., 2020)

**Fix Strategy**: 
1. Option A: Sample entities and train per-entity models
2. Option B: Use entity-aware libraries like `pytorch-forecasting` or `tsai`

---

### 4. Transformer SOTA Models (via NeuralForecast)

| Model | Status | Runs | MAE | Paper Reference |
|-------|--------|------|-----|-----------------|
| PatchTST | ⚠️ | 18 | 601,864 [FALLBACK] | Nie et al., ICLR 2023 |
| iTransformer | ⚠️ | 18 | 601,864 [FALLBACK] | Liu et al., ICLR 2024 |
| TimesNet | ⚠️ | 18 | 601,864 [FALLBACK] | Wu et al., ICLR 2023 |
| TSMixer | ⚠️ | 18 | 601,864 [FALLBACK] | Chen et al., TMLR 2023 |

**Root Cause**: Same panel data incompatibility as deep_classical.

**Verification**: All are official Nixtla implementations in neuralforecast 3.1.4:
- PatchTST: `neuralforecast.models.PatchTST`
- iTransformer: `neuralforecast.models.iTransformer`
- TimesNet: `neuralforecast.models.TimesNet`
- TSMixer: `neuralforecast.models.TSMixer`

---

### 5. Foundation Models

| Model | Status | Runs | Issue |
|-------|--------|------|-------|
| Chronos | 🔧 | 0 | chronos 2.2.2 installed, wrapper incomplete |
| TimesFM | ⚠️ | 0 | **NOT INSTALLED** |
| Moirai | ⚠️ | 0 | **NOT INSTALLED** (requires uni2ts) |

**Dependencies**:
- chronos: 2.2.2 ✅
- timesfm: NOT INSTALLED ❌
- uni2ts (Moirai): NOT INSTALLED ❌

**Fix Strategy**: 
1. Install timesfm: `pip install timesfm`
2. Complete Chronos wrapper in `deep_models.py`
3. Install uni2ts for Moirai (optional)

---

### 6. Irregular-Aware Models

| Model | Status | Runs | Issue |
|-------|--------|------|-------|
| GRU-D | 🔧 | 0 | torch 2.7.1 OK, implementation complete |
| SAITS | ⚠️ | 0 | pypots NOT INSTALLED |

**Dependencies**:
- torch: 2.7.1+cu126 ✅
- pypots: NOT INSTALLED ❌

---

## Current Benchmark Coverage

### Runs Completed (by Task × Ablation)

| Task | Category | core_only | full | Total |
|------|----------|-----------|------|-------|
| task1_outcome | ml_tabular | 3 (LightGBM, XGBoost) | 0 | 3 |
| task1_outcome | deep_classical | 12 | 12 | 24 |
| task1_outcome | transformer_sota | 12 | 12 | 24 |
| task2_forecast | deep_classical | 12 | 12 | 24 |
| task2_forecast | transformer_sota | 12 | 12 | 24 |
| task3_risk_adjust | deep_classical | 12 | 12 | 24 |
| task3_risk_adjust | transformer_sota | 12 | 12 | 24 |
| **TOTAL** | - | - | - | **150** |

### Valid Results for Paper

Only **2 models** currently have valid (non-fallback) results:

| Model | Category | Task | Ablation | MAE | RMSE |
|-------|----------|------|----------|-----|------|
| XGBoost | ml_tabular | task1_outcome | core_only | 274,952 | 1,892,455* |
| LightGBM | ml_tabular | task1_outcome | core_only | 360,834 | 2,056,789* |

*RMSE values estimated from metrics.json

---

## Action Items for Paper-Grade Results

### Priority 1: Complete ml_tabular (Immediate)
```bash
# Run all remaining ml_tabular models
for TASK in task1_outcome task2_forecast task3_risk_adjust; do
  for ABL in core_only full; do
    python scripts/run_block3_benchmark_shard.py \
      --task $TASK --category ml_tabular --ablation $ABL \
      --preset standard \
      --output-dir runs/benchmarks/block3_20260203_225620_4090_standard/$TASK/ml_tabular/$ABL
  done
done
```

### Priority 2: Fix Statistical Models
- Edit `src/narrative/block3/models/statistical.py`
- Implement entity-sampling strategy (sample 100 entities, 500 obs each)
- Re-run statistical benchmark

### Priority 3: Fix Deep/Transformer Models
- Option A: Implement entity-aware training loop
- Option B: Switch to `pytorch-forecasting` or `tsai` backends
- Option C: Sample approach similar to statistical fix

### Priority 4: Install Missing Dependencies
```bash
pip install timesfm pypots
```

---

## Paper Table Template (KDD'26)

Once all models are properly run, the paper table should look like:

| Model | Type | MAE | RMSE | MAPE | Time (s) |
|-------|------|-----|------|------|----------|
| MeanPredictor | Baseline | - | - | - | <1 |
| SeasonalNaive | Statistical | - | - | - | - |
| AutoARIMA | Statistical | - | - | - | - |
| XGBoost | ML | **274,952** | - | - | - |
| LightGBM | ML | 360,834 | - | - | - |
| CatBoost | ML | - | - | - | - |
| N-BEATS | Deep | - | - | - | - |
| TFT | Deep | - | - | - | - |
| PatchTST | Transformer | - | - | - | - |
| iTransformer | Transformer | - | - | - | - |
| TimesNet | Transformer | - | - | - | - |
| Chronos | Foundation | - | - | - | - |

---

## Technical Notes

### Data Characteristics
- **Total rows**: 5,553,820 (train: 4,421,931 / val: 575,246 / test: 556,643)
- **Entities**: 20,944 unique entity_ids
- **Time span**: ~210 days per entity on average
- **Panel structure**: High N (entities), moderate T (time)

### Known Wrapper Issues
1. **deep_models.py line 140-145**: Fallback triggered when n_samples > 50,000
2. **statistical.py line 91-93**: pd.date_range overflow for large periods
3. **FoundationModelWrapper**: Chronos predict method incomplete

### Environment
- Python: 3.11.13
- CUDA: Available (torch 2.7.1+cu126)
- neuralforecast: 3.1.4 (official Nixtla)
- statsforecast: 2.0.3 (official Nixtla)

---

## Conclusion

**Current State**: 2/33 models (6%) have paper-ready results.

**Blocking Issue**: Panel data (20K entities × 210 days) incompatible with time-series libraries designed for few long series.

**Path Forward**: 
1. Focus on ml_tabular (already works) for immediate paper results
2. Implement entity-sampling strategy for statistical/deep models
3. Consider switching to panel-aware libraries for deep learning models
