# SOTA Models for Time Series Forecasting

This document catalogs the state-of-the-art models integrated into the Block 3 benchmark harness.

## Model Categories

### 1. Statistical Baselines

| Model | Description | Implementation | Exog Support | Probabilistic |
|-------|-------------|----------------|--------------|---------------|
| SeasonalNaive | Repeat last seasonal period | `statsforecast` | ❌ | ❌ |
| ETS | Exponential smoothing | `statsforecast` | ❌ | ✅ |
| AutoARIMA | Auto ARIMA selection | `statsforecast` | ✅ | ✅ |
| Theta | Theta method | `statsforecast` | ❌ | ✅ |

### 2. ML Tabular

| Model | Description | Implementation | Exog Support | Probabilistic |
|-------|-------------|----------------|--------------|---------------|
| LightGBM | Gradient boosting | `lightgbm` | ✅ | ❌ (quantile) |
| XGBoost | Gradient boosting | `xgboost` | ✅ | ❌ (quantile) |
| CatBoost | Gradient boosting | `catboost` | ✅ | ❌ (quantile) |
| RandomForest | Ensemble trees | `sklearn` | ✅ | ❌ |

### 3. Deep Classical

| Model | Description | Paper | Implementation | Exog Support | Probabilistic |
|-------|-------------|-------|----------------|--------------|---------------|
| N-BEATS | Neural basis expansion | [ICLR 2020](https://arxiv.org/abs/1905.10437) | `pytorch-forecasting` | ❌ | ✅ |
| N-HiTS | Hierarchical N-BEATS | [AAAI 2023](https://arxiv.org/abs/2201.12886) | `neuralforecast` | ❌ | ✅ |
| TFT | Temporal Fusion Transformer | [IJF 2021](https://arxiv.org/abs/1912.09363) | `pytorch-forecasting` | ✅ | ✅ |
| DeepAR | Autoregressive RNN | [IJF 2020](https://arxiv.org/abs/1704.04110) | `gluonts` | ✅ | ✅ |

### 4. Transformer SOTA (THUML Time-Series-Library)

| Model | Description | Paper | Source | Exog Support | Probabilistic |
|-------|-------------|-------|--------|--------------|---------------|
| **PatchTST** | Patched time series transformer | [ICLR 2023](https://arxiv.org/abs/2211.14730) | [thuml/PatchTST](https://github.com/yuqinie98/PatchTST) | ❌ | ❌ |
| **iTransformer** | Inverted transformer | [ICLR 2024](https://arxiv.org/abs/2310.06625) | [thuml/iTransformer](https://github.com/thuml/iTransformer) | ✅ | ❌ |
| **TimeMixer** | Multi-scale mixing | [ICLR 2024](https://arxiv.org/abs/2405.14616) | [thuml/TimeMixer](https://github.com/kwuking/TimeMixer) | ✅ | ❌ |
| **TimesNet** | Temporal 2D variation | [ICLR 2023](https://arxiv.org/abs/2210.02186) | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | ✅ | ❌ |
| Autoformer | Auto-correlation transformer | [NeurIPS 2021](https://arxiv.org/abs/2106.13008) | [thuml/Autoformer](https://github.com/thuml/Autoformer) | ✅ | ❌ |
| FEDformer | Frequency enhanced | [ICML 2022](https://arxiv.org/abs/2201.12740) | [thuml/FEDformer](https://github.com/MAZiqing/FEDformer) | ✅ | ❌ |
| Informer | Efficient transformer | [AAAI 2021](https://arxiv.org/abs/2012.07436) | [thuml/Informer](https://github.com/zhouhaoyi/Informer2020) | ✅ | ❌ |
| DLinear | Simple linear | [AAAI 2023](https://arxiv.org/abs/2205.13504) | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) | ❌ | ❌ |

**Unified Interface**: [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)

### 5. GluonTS Ecosystem

| Model | Description | Paper | Source | Exog Support | Probabilistic |
|-------|-------------|-------|--------|--------------|---------------|
| DeepAR | Autoregressive RNN | [IJF 2020](https://arxiv.org/abs/1704.04110) | `gluonts.torch` | ✅ | ✅ |
| WaveNet | Dilated convolutions | [SSRN 2016](https://arxiv.org/abs/1609.03499) | `gluonts.torch` | ✅ | ✅ |
| SimpleFeedForward | MLP baseline | - | `gluonts.torch` | ✅ | ✅ |
| Transformer | Vanilla transformer | - | `gluonts.torch` | ✅ | ✅ |

**Installation**: `pip install gluonts[torch]`

### 6. Foundation Models

| Model | Description | Paper | Source | Exog Support | Probabilistic | Pre-trained |
|-------|-------------|-------|--------|--------------|---------------|-------------|
| **TimesFM** | Google's foundation model | [arXiv 2024](https://arxiv.org/abs/2310.10688) | [google-research/timesfm](https://github.com/google-research/timesfm) | ❌ | ✅ | ✅ 200M params |
| **Chronos** | Amazon's foundation model | [arXiv 2024](https://arxiv.org/abs/2403.07815) | [amazon-science/chronos](https://github.com/amazon-science/chronos-forecasting) | ❌ | ✅ | ✅ T5-based |
| **Moirai** | Salesforce's foundation | [arXiv 2024](https://arxiv.org/abs/2402.02592) | [salesforce/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | ❌ | ✅ | ✅ |
| Lag-Llama | LLM for time series | [arXiv 2024](https://arxiv.org/abs/2310.08278) | [time-series-foundation/lag-llama](https://github.com/time-series-foundation-models/lag-llama) | ❌ | ✅ | ✅ |
| MOMENT | Moment-based foundation | [ICML 2024](https://arxiv.org/abs/2402.03885) | [moment-timeseries](https://github.com/moment-timeseries-foundation-model/moment) | ❌ | ❌ | ✅ |

---

## AutoFit Model Selection

Based on data profile meta-features, AutoFit selects the optimal backbone:

| Condition | Backbone | Rationale |
|-----------|----------|-----------|
| `multiscale_score > 0.7` | TimeMixer | Multi-resolution decomposition |
| `periodicity_score > 0.6` | PatchTST | Strong seasonal patterns |
| `nonstationarity_score < 0.3` | iTransformer | Stable channel correlations |
| Default | TimesNet | General-purpose temporal 2D |

### Fusion Selection

| Condition | Fusion Type | Rationale |
|-----------|-------------|-----------|
| `exog_strength > 0.3` | Cross-attention | Strong exogenous signal |
| `text_strength > 0.2 OR edgar_strength > 0.2` | FiLM | Moderate auxiliary |
| Default | None | Endogenous only |

---

## Installation Guide

### Core Dependencies
```bash
pip install torch>=2.0 pytorch-lightning>=2.0
pip install gluonts[torch]
pip install statsforecast neuralforecast
pip install lightgbm xgboost catboost
```

### THUML Time-Series-Library
```bash
git clone https://github.com/thuml/Time-Series-Library.git
cd Time-Series-Library
pip install -e .
```

### Foundation Models
```bash
# TimesFM
pip install timesfm

# Chronos
pip install chronos-forecasting

# Moirai
pip install uni2ts
```

---

## Benchmark Configuration

From `configs/block3.yaml`:

```yaml
baselines:
  statistical:
    - SeasonalNaive
    - AutoARIMA
    - ETS

  ml_tabular:
    - LightGBM
    - XGBoost
    - CatBoost

  deep_classical:
    - N-BEATS
    - TFT

  transformer_sota:
    - PatchTST
    - iTransformer
    - TimeMixer

  gluonts:
    - DeepAR
    - WaveNet

  foundation:
    - TimesFM
    - Chronos
```

---

## Performance Notes

### Memory Requirements (single GPU)

| Model | VRAM | Batch Size | Notes |
|-------|------|------------|-------|
| PatchTST | 4GB | 32 | Efficient patching |
| iTransformer | 6GB | 32 | Channel attention |
| TimeMixer | 8GB | 32 | Multi-scale |
| TFT | 8GB | 64 | Attention + LSTM |
| TimesFM | 12GB | 16 | Foundation model |
| Chronos | 16GB | 8 | T5-based |

### Training Time (1000 entities, 365 days)

| Model | Time | Hardware |
|-------|------|----------|
| LightGBM | ~2 min | CPU |
| PatchTST | ~30 min | RTX 4090 |
| TimesFM (zero-shot) | ~5 min | RTX 4090 |
| Chronos (zero-shot) | ~10 min | RTX 4090 |

---

## References

1. **Time-Series-Library**: https://github.com/thuml/Time-Series-Library
2. **GluonTS**: https://ts.gluon.ai/
3. **NeuralForecast**: https://nixtla.github.io/neuralforecast/
4. **Statsforecast**: https://nixtla.github.io/statsforecast/
5. **TimesFM**: https://github.com/google-research/timesfm
6. **Chronos**: https://github.com/amazon-science/chronos-forecasting
