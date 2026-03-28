# SOTA Models for Time Series Forecasting

> 2026-03-23 update:
> This catalog is now best read as a historical integration overview plus a
> reference shelf for benchmarked families. Its old AutoFit selection section
> is **not** the current operational truth for Block 3. The current clean line
> is `AutoFitV739` under `runs/benchmarks/block3_phase9_fair/`, and the active
> V740 direction is a **single-model conditional forecaster**, not a
> hand-written meta-feature router. See:
> - `docs/CURRENT_SOURCE_OF_TRUTH.md`
> - `docs/references/BLOCK3_CHAMPION_COMPONENT_ANALYSIS.md`
> - `docs/references/V740_DESIGN_SPECIFICATION.md`
> - `docs/references/V740_ALPHA_ENGINEERING_SPEC.md`

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

## Historical AutoFit Model Selection (Superseded)

The table below describes an older design idea and should now be interpreted as
historical background only. It is kept for traceability, but it is no longer
the current Block 3 truth.

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

## Current V740-Relevant Additions (2026-03-23)

The most relevant missing-or-newly-added models for the current V740 goal are:

| Model | Venue | Why it matters now | Current local status |
|-------|-------|--------------------|----------------------|
| SAMformer | ICML 2024 Oral | Efficient single-model transformer with SAM + channel-wise attention | **Integrated locally**; synthetic smoke passed, and a generic freeze-backed local smoke now passes on `task1_outcome/core_edgar/is_funded/h=14` (`MAE=0.3863`, non-constant). Still not canonical-benchmark-cleared |
| LightGTS | ICML 2025 | Lightweight general TS model; highly aligned with V740 efficiency target | missing |
| OLinear | NeurIPS 2025 | Strong low-cost orthogonal-domain linear baseline | missing, blocked by matrix artifacts |
| LiPFormer | ICDE 2025 | Lightweight patch-based alternative to PatchTST | missing |
| ElasTST | NeurIPS 2024 | One model for varied horizons; directly relevant to 160-cell coverage | missing |
| DIAN | IJCAI 2024 | Invariant/variant decoupling for heterogeneous panels | missing |
| UniTS | NeurIPS 2024 | Unified multi-task design reference for V740 conditioning | missing |
| Prophet | classic business TS baseline | Cheap reviewer-recognizable sanity line for finance-style forecasting | wrapper exists locally; user-local vendor install works; generic real-data smoke now passes |
| TabPFN-TS | NeurIPS 2024 workshop line | Ultra-fast zero-shot TS baseline; very high information-gain comparator | local wrappers already exist; `GLIBCXX` import blocker is resolved, current blocker is gated HF model access |

### Fresh 2025-2026 Design Signals Worth Tracking

These are not necessarily the next benchmark additions, but they matter for the
V740 methodology direction:

| Signal | Venue | Why it matters |
|-------|-------|----------------|
| CASA | IJCAI 2025 | Efficient score-attention design; relevant to V740's lightweight local-context branch |
| TimeEmb | NeurIPS 2025 | Lightweight static-dynamic disentanglement for mixed panel data |
| DistDF | ICLR 2026 | Direct distributional multistep objective; relevant to long-horizon stability |
| QDF | ICLR 2026 | Hard-structure-aware weighting objective; useful for difficult cells |
| JAPAN | ICLR 2026 | Density-aware conformal prediction sets; useful for calibrated uncertainty |
| PIH / MEW modules | ICLR 2026-under-review | Strong signal that longer horizons must be treated as an effective-window problem, not just a bigger horizon token; evaluated on the standard `Weather/Traffic/Electricity/ETT/Solar/PEMS` family |
| Selective Learning | NeurIPS 2025 | Timestep-selective training to reduce noise-driven overfitting on hard temporal regions; especially relevant because it is examined across non-stationary (`ETTh1`, `Exchange`) and periodic (`Weather`) regimes |
| Decomposition Delivers Both | NeurIPS 2024 | Reinforces the decomposition-first path as an efficiency-preserving backbone choice |
| TimeDiT | 2025 | General-purpose diffusion-style TSFM; interesting but too heavy for alpha |
| FLAIRR-TS | Findings of EMNLP 2025 | Retrieval + iterative refinement; promising as a later auxiliary memory signal |

### Heavy but still important public-comparison references

These models are not the first additions we should prioritize for the
efficiency-first V740 path, but they still matter as public-benchmark
reference points and should stay visible in the literature audit:

| Model | Venue | Why it still matters |
|-------|-------|----------------------|
| TimeMoE | ICLR 2025 | Already in our codebase and registry; important heavy foundation-model reference for zero-shot/full-shot comparisons |
| TimerXL | ICLR 2025 | Already in our codebase and registry; strong long-context baseline for public generalization studies |
| TimeDiT | 2025 | Useful upper-bound reference for "general TSFM" style modeling, even if too heavy for V740-alpha |

### Most relevant mechanism clusters for the next V740 step

The literature pattern is now clear enough that we should stop thinking in
terms of isolated paper names and instead track mechanism clusters:

1. **single-model multi-task / multi-horizon conditioning**
   - `UniTS`, `ElasTST`, `TimerXL`
2. **lightweight long-horizon / local-context modeling**
   - `LightGTS`, `LiPFormer`, `SAMformer`, `CASA`, `PIH`
3. **distribution alignment and calibration**
   - `DistDF`, `QDF`, `JAPAN`, `Time-o1`, `Selective Learning`
4. **heterogeneous covariate disentanglement**
   - `DIAN`, `TimeEmb`
5. **decomposition-first compression**
   - `Decomposition Delivers Both`, `NBEATS`-family signals

This is the most useful reading order for V740 now. It matches both the Block
3 champion analysis and the broader 2024-2026 public-paper comparator pattern.

## Recurring Public Benchmark Families in 2024-2026 Papers

For future V740 generalization testing, we should not pick public datasets
arbitrarily. The recurring families across recent top-venue forecasting papers
and official repos cluster into the following groups.

### 1. Canonical long-horizon multivariate forecasting

- `ETTh1`, `ETTh2`
- `ETTm1`, `ETTm2`
- `Electricity` / `ECL`
- `Traffic`
- `Weather`
- `Exchange`
- `ILI` / `Illness`

These remain the most common direct-comparison datasets for recent long-term
forecasting papers and should form the default external evaluation pack.

### 2. Traffic / graph-like temporal structure

- `PEMS03`
- `PEMS04`
- `PEMS07`
- `PEMS08`

These are especially useful when auditing whether a mechanism transfers beyond
our entrepreneurial-finance panel into strongly structured multivariate
dynamics.

### 3. Covariate-sensitive / exogenous forecasting

- `Solar`
- energy or load datasets with exogenous variables
- market or electricity-price datasets used in covariate-aware forecasting

These are the most relevant public tests for the EDGAR/text side of V740,
because they stress exogenous usefulness and availability rather than purely
endogenous trend fitting.

### 4. Broader robustness suites

- `M4`
- `M5`
- selected Monash Forecasting Repository datasets

These are not close analogues of Block 3, but they remain useful for checking
whether a candidate V740 line is becoming over-specialized to one proprietary
panel domain.

## Common Comparator Families in 2024-2026 Papers

The latest top-venue forecasting papers also keep returning to a relatively
stable comparison pack. For our eventual public generalization benchmark, these
families should be treated as the default external comparator set:

### 1. Linear / decomposition baselines

- `DLinear`
- `RLinear`
- `OLinear`

These remain essential because they frequently stay competitive on stable
continuous/count regimes and provide the strongest cheap sanity anchors.

### 2. Transformer / mixer baselines

- `PatchTST`
- `iTransformer`
- `TimesNet`
- `TimeMixer`
- `FEDformer`
- `Crossformer`
- `Informer`
- `Autoformer`

These are still the default modern long-horizon comparison family in many
papers, even when the new method is not itself a Transformer.

### 3. Deep classical / direct forecasters

- `NBEATS`
- `NHITS`
- `TFT`
- `TiDE`
- `TCN`

These matter especially for our data because Block 3 already shows that
`NBEATS` and `NHITS` remain among the hardest champions to beat.

### 4. Foundation / pretrained references

- `Chronos`
- `Moirai`
- `TimesFM`
- `TimerXL`
- `TimeMoE`
- `Sundial`

These models are not always the most efficient local additions, but they are
important for public-paper credibility because they recur in zero-shot /
full-shot comparison packs across recent foundation-model papers.

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
