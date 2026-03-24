# Block 3 Model Registry

> Last updated: 2026-03-23
> Source: `src/narrative/block3/models/registry.py` + current Phase 9 fair benchmark scan
> Registered in code: **153** models | Raw benchmark materialized: **137** | Active leaderboard models: **92** | Active complete models @160: **64**

## Summary

The table below is intentionally registry-centric. Exact benchmark completeness
and exclusion status continue to evolve as Phase 15+H jobs land, so the current
authoritative benchmark counts should always be read from:

- `docs/CURRENT_SOURCE_OF_TRUTH.md`
- `docs/benchmarks/phase9_current_snapshot.md`
- `runs/benchmarks/block3_phase9_fair/all_results.csv`

| Category | Registered in code | Primary library / wrapper |
|---|---:|---|
| statistical | 15 | StatsForecast |
| deep_classical | 9 | NeuralForecast |
| transformer_sota | 24 | NeuralForecast |
| foundation | 15 | Custom wrappers / HF |
| irregular | 4 | PyPOTS |
| ml_tabular | 20 | scikit-learn, XGBoost, LightGBM, CatBoost |
| tslib_sota | 42 | TSLib (vendored) |
| autofit | 24 | Custom AutoFit lines (historical retired + current V739) |
| **Total** | **153** | |

## 1. Statistical Models (15 models — all @160 COMPLETE)

| Model | Library | Method | Paper/Reference | Year | Status |
|---|---|---|---|---|---|
| AutoARIMA | StatsForecast | Auto ARIMA(p,d,q) selection | Hyndman & Khandakar (2008) | 2008 | ✅ @160 |
| AutoETS | StatsForecast | Auto Error-Trend-Seasonality | Hyndman et al. (2002) | 2002 | ✅ @160 |
| AutoTheta | StatsForecast | Auto Theta method | Assimakopoulos & Nikolopoulos (2000) | 2000 | ✅ @160 |
| AutoCES | StatsForecast | Complex Exponential Smoothing | Svetunkov & Boylan (2019) | 2019 | ✅ @160 |
| DynamicOptimizedTheta | StatsForecast | Dynamic Optimized Theta | Fiorucci et al. (2016) | 2016 | ✅ @160 |
| MSTL | StatsForecast | Multi-Seasonal Trend Decomp | Bandara et al. (2021) | 2021 | ✅ @160 |
| Holt | StatsForecast | Holt's linear trend | Holt (1957) | 1957 | ✅ @160 |
| HoltWinters | StatsForecast | Holt-Winters seasonal | Winters (1960) | 1960 | ✅ @160 |
| CrostonClassic | StatsForecast | Intermittent demand (classic) | Croston (1972) | 1972 | ✅ @160 |
| CrostonOptimized | StatsForecast | Optimized Croston | Syntetos & Boylan (2005) | 2005 | ✅ @160 |
| CrostonSBA | StatsForecast | Bias-adjusted Croston | Syntetos & Boylan (2001) | 2001 | ✅ @160 |
| HistoricAverage | StatsForecast | Simple historic mean | Baseline | — | ✅ @160 |
| Naive | StatsForecast | Last-value persistence | Baseline | — | ✅ @160 |
| SF_SeasonalNaive | StatsForecast | Seasonal naive | Baseline | — | ✅ @160 |
| WindowAverage | StatsForecast | Moving window average | Baseline | — | ✅ @160 |

**Seed stability**: All deterministic — exact 0.000% seed delta.

## 2. Deep Classical Models (9 models — all @160 COMPLETE)

| Model | Library | Architecture | Paper/Reference | Year | Status | Wins |
|---|---|---|---|---|---|---|
| NBEATS | NeuralForecast | Basis expansion + double residual | Oreshkin et al. (ICLR 2020) | 2020 | ✅ @160 | **65** |
| NHITS | NeuralForecast | Hierarchical interpolation | Challu et al. (AAAI 2023) | 2023 | ✅ @160 | **21** |
| GRU | NeuralForecast | Gated recurrent unit | Cho et al. (EMNLP 2014) | 2014 | ✅ @160 | **11** |
| DeepAR | NeuralForecast | Autoregressive probabilistic | Salinas et al. (IJoF 2020) | 2020 | ✅ @160 | 0 |
| LSTM | NeuralForecast | Long short-term memory | Hochreiter & Schmidhuber (1997) | 1997 | ✅ @160 | 0 |
| MLP | NeuralForecast | Multi-layer perceptron | Baseline | — | ✅ @160 | 0 |
| TCN | NeuralForecast | Temporal convolution network | Bai et al. (2018) | 2018 | ✅ @160 | 0 |
| TFT | NeuralForecast | Temporal Fusion Transformer | Lim et al. (IJoF 2021) | 2021 | ✅ @160 | 0 |
| DilatedRNN | NeuralForecast | Dilated recurrent network | Chang et al. (NeurIPS 2017) | 2017 | ✅ @160 | 0 |

**Category champion**: NBEATS — 65/160 wins (40.6%), mean rank 5.01.
**Category total wins**: 97/160 (60.6%) — dominant category.

## 3. Transformer / SOTA Models (23 active — all @160 COMPLETE)

| Model | Library | Architecture | Paper/Reference | Year | Status | Wins |
|---|---|---|---|---|---|---|
| KAN | NeuralForecast | Kolmogorov-Arnold Networks | Liu et al. (2024) | 2024 | ✅ @160 | **16** |
| DeepNPTS | NeuralForecast | Non-parametric TS | Rangapuram et al. (2023) | 2023 | ✅ @160 | **16** |
| PatchTST | NeuralForecast | Patching + Transformer | Nie et al. (ICLR 2023) | 2023 | ✅ @160 | **4** |
| NBEATSx | NeuralForecast | NBEATS + exogenous | Olivares et al. (IJoF 2023) | 2023 | ✅ @160 | **3** |
| DLinear | NeuralForecast | Decomposition + Linear | Zeng et al. (AAAI 2023) | 2023 | ✅ @160 | **1** |
| NLinear | NeuralForecast | Normalization + Linear | Zeng et al. (AAAI 2023) | 2023 | ✅ @160 | 0 |
| TimesNet | NeuralForecast | 2D temporal variation | Wu et al. (ICLR 2023) | 2023 | ✅ @160 | 0 |
| Informer | NeuralForecast | ProbSparse attention | Zhou et al. (AAAI 2021) | 2021 | ✅ @160 | 0 |
| Autoformer | NeuralForecast | Auto-correlation decomp | Wu et al. (NeurIPS 2021) | 2021 | ✅ @160 | 0 |
| FEDformer | NeuralForecast | Frequency attention | Zhou et al. (ICML 2022) | 2022 | ✅ @160 | 0 |
| iTransformer | NeuralForecast | Inverted Transformer | Liu et al. (ICLR 2024) | 2024 | ✅ @160 | 0 |
| TiDE | NeuralForecast | Time-series Dense Encoder | Das et al. (TMLR 2024) | 2024 | ✅ @160 | 0 |
| TSMixer | NeuralForecast | Time-Series MLP-Mixer | Chen et al. (2023) | 2023 | ✅ @160 | 0 |
| TSMixerx | NeuralForecast | TSMixer + exogenous | Chen et al. (2023) | 2023 | ✅ @160 | 0 |
| TimeMixer | NeuralForecast | Multi-scale mixing | Wang et al. (ICLR 2024) | 2024 | ✅ @160 | 0 |
| TimeLLM | NeuralForecast | LLM-reprogrammed TS | Jin et al. (ICLR 2024) | 2024 | ✅ @160 | 0 |
| TimeXer | NeuralForecast | Cross-variable Transformer | Wang et al. (2024) | 2024 | ✅ @160 | 0 |
| xLSTM | NeuralForecast | Extended LSTM | Beck et al. (2024) | 2024 | ✅ @160 | 0 |
| BiTCN | NeuralForecast | Bidirectional TCN | NeuralForecast contrib | 2023 | ✅ @160 | 0 |
| RMoK | NeuralForecast | Recurrent MoK | NeuralForecast contrib | 2024 | ✅ @160 | 0 |
| SOFTS | NeuralForecast | Self-Organizing TS | NeuralForecast contrib | 2024 | ✅ @160 | 0 |
| StemGNN | NeuralForecast | Spectral Temporal GNN | Cao et al. (NeurIPS 2020) | 2020 | ✅ @160 | 0 |
| VanillaTransformer | NeuralForecast | Standard Transformer | Vaswani et al. (2017) | 2017 | ✅ @160 | 0 |

**Category champion**: KAN and DeepNPTS tied at 16 wins each.
**Category total wins**: 40/160 (25.0%).

**Excluded**: Koopa (NaN divergence) — registered but not active.

## 4. Foundation Models (14 models — all @160 COMPLETE)

| Model | Library | Architecture | Paper/Reference | Year | Status | Wins |
|---|---|---|---|---|---|---|
| Chronos | Custom (HF) | T5 tokenized decoder | Ansari et al. (Amazon, 2024) | 2024 | ✅ @160 | **23** |
| ChronosBolt | Custom (HF) | Efficient Chronos variant | Amazon (2025) | 2025 | ✅ @160 | 0 |
| Chronos2 | Custom (HF) | Chronos v2 | Amazon (2025) | 2025 | ✅ @160 | 0 |
| Moirai | Custom (HF) | Universal forecaster | Salesforce (2024) | 2024 | ✅ @160 | 0 |
| Moirai2 | Custom (HF) | Moirai v2 | Salesforce (2025) | 2025 | ✅ @160 | 0 |
| MoiraiLarge | Custom (HF) | Moirai large variant | Salesforce (2024) | 2024 | ✅ @160 | 0 |
| LagLlama | Custom (HF) | Lag-based LLM TS | Rasul et al. (2024) | 2024 | ✅ @160 | 0 |
| MOMENT | Custom (HF) | Multi-task foundation | Goswami et al. (ICML 2024) | 2024 | ✅ @160 | 0 |
| Sundial | Custom (HF) | Solar/temporal foundation | Alibaba (2025) | 2025 | ✅ @160 | 0 |
| TTM | Custom (HF) | Tiny Time Mixer | IBM (2024) | 2024 | ✅ @160 | 0 |
| TimeMoE | Custom (HF) | MoE time series | TimeMoE team (2024) | 2024 | ✅ @160 | 0 |
| Timer | Custom (HF) | General TS foundation | Thuml (2024) | 2024 | ✅ @160 | 0 |
| TimesFM | Custom (HF) | Google foundation | Das et al. (Google, 2024) | 2024 | ✅ @160 | 0 |
| TimesFM2 | Custom (HF) | TimesFM v2 | Google (2025) | 2025 | ✅ @160 | 0 |

**Category champion**: Chronos — 23/160 wins (14.4%), all on funding_raised_usd.
**Category total wins**: 23/160 (14.4%).
**Notable**: ChronosBolt is best mean rank in category (7.42 overall).

## 5. Irregular / Missing-Data Models (4 models — all @160 COMPLETE)

| Model | Library | Architecture | Paper/Reference | Year | Status | Wins |
|---|---|---|---|---|---|---|
| BRITS | PyPOTS | Bidirectional RNN Imputation | Cao et al. (NeurIPS 2018) | 2018 | ✅ @160 | 0 |
| CSDI | PyPOTS | Score-based Diffusion | Tashiro et al. (NeurIPS 2021) | 2021 | ✅ @160 | 0 |
| GRU-D | PyPOTS | GRU with decay | Che et al. (Scientific Reports 2018) | 2018 | ✅ @160 | 0 |
| SAITS | PyPOTS | Self-Attention Imputation TS | Du et al. (Expert Systems 2023) | 2023 | ✅ @160 | 0 |

**Seed stability warning**: CSDI avg seed delta = 4.42% (most unstable model in benchmark).

## 6. ML / Tabular Models (11 active — 0 complete @160)

| Model | Library | Method | Status | Missing |
|---|---|---|---|---|
| CatBoost | CatBoost | Gradient boosting (ordered) | @157 | 3 cells: t1/fu/is_funded/{h7,h14,h30} |
| XGBoost | XGBoost | Gradient boosting (tree) | @157 | same |
| XGBoostPoisson | XGBoost | Poisson objective | @157 | same |
| LightGBM | LightGBM | Gradient boosting (leaf-wise) | @157 | same |
| LightGBMTweedie | LightGBM | Tweedie objective | @157 | same |
| HistGradientBoosting | scikit-learn | Histogram-based GB | @157 | same |
| ExtraTrees | scikit-learn | Extremely randomized trees | @157 | same |
| RandomForest | scikit-learn | Random forest regressor | @157 | same |
| MeanPredictor | Custom | Global mean baseline | @157 | same |
| SeasonalNaive | Custom | Seasonal naive baseline | @157 | same |
| NegativeBinomialGLM | statsmodels | Negative Binomial GLM | @21 | Structural failure (convergence) |

**Gap-fill**: All 10 @157 models missing exactly 3 cells each (task1_outcome/full/is_funded/{h7,h14,h30}).
Root cause: harness ran all 10 simultaneously → OOM at 640GB. Fix: split script running models one-by-one (job 5263582 PENDING).

**Excluded from registry (not active)**:
- Ridge, Lasso — redundant with HistGradientBoosting
- SVR, KNN — O(n²) complexity
- LogisticRegression — classification-only

## 7. TSLib Vendored Models (40 active — 4 complete @160)

### 7.1 Complete TSLib Models (@160)

| Model | Paper/Reference | Year | Venue | Status | Wins |
|---|---|---|---|---|---|
| CATS | Li et al. | 2025 | NeurIPS 2025 | ✅ @160 | 0 |
| FITS | Xu et al. | 2024 | ICLR 2024 | ✅ @160 | 0 |
| KANAD | Yang et al. | 2025 | 2025 | ✅ @160 | 0 |
| WPMixer | Deng et al. | 2025 | AAAI 2025 | ✅ @160 | 0 |

### 7.2 Partial TSLib Models (@105–111)

| Model | Records | Paper/Reference | Year | Venue | Gap |
|---|---|---|---|---|---|
| ETSformer | 111 | Woo et al. | 2022 | ICML 2022 | Missing t3 seed2 |
| LightTS | 111 | Zhang et al. | 2022 | 2022 | Missing t3 seed2 |
| Pyraformer | 111 | Liu et al. | 2022 | ICLR 2022 | Missing t3 seed2 |
| Reformer | 111 | Kitaev et al. | 2020 | ICLR 2020 | Missing t3 seed2 |
| Crossformer | 105 | Zhang & Yan | 2023 | ICLR 2023 | Missing seed2 + t3 cells |
| MSGNet | 105 | Cai et al. | 2024 | AAAI 2024 | same |
| MambaSimple | 105 | Gu & Dao | 2024 | 2024 | same |
| MultiPatchFormer | 105 | Hwang et al. | 2025 | 2025 | same |
| PAttn | 105 | Liang et al. | 2024 | NeurIPS 2024 | same |
| TimeFilter | 105 | Luo et al. | 2025 | 2025 | same |
| ModernTCN | 29 | Donghao & Xue | 2024 | ICLR 2024 | OOM/TIMEOUT, covered by ALL33 |

### 7.3 Phase 15 New Models (@26 each — all missing seed2)

| Model | Paper/Reference | Year | Venue | Status |
|---|---|---|---|---|
| CARD | Wang et al. | 2024 | ICLR 2024 | @26 |
| CFPT | Chen et al. | 2025 | 2025 | @26 ⚠️ conv2d channel bug |
| DeformableTST | Li et al. | 2025 | 2025 | @26 |
| DUET | Chen et al. | 2024 | NeurIPS 2024 | @26 |
| FiLM | Zhou et al. | 2022 | NeurIPS 2022 | @26 |
| FilterTS | Luo et al. | 2025 | 2025 | @26 |
| FreTS | Yi et al. | 2024 | NeurIPS 2024 | @26 |
| Fredformer | Piao et al. | 2024 | 2024 | @26 |
| MICN | Wang et al. | 2023 | ICLR 2023 | @26 |
| NonstationaryTransformer | Liu et al. | 2022 | NeurIPS 2022 | @26 |
| PDF | Dai et al. | 2024 | 2024 | @26 |
| PIR | Gao et al. | 2025 | 2025 | @26 |
| PathFormer | Chen et al. | 2024 | ICLR 2024 | @26 |
| SCINet | Liu et al. | 2022 | NeurIPS 2022 | @26 |
| SEMPO | Wei et al. | 2025 | 2025 | @26 |
| SRSNet | Li et al. | 2025 | 2025 | @26 |
| SegRNN | Lin et al. | 2023 | 2023 | @26 |
| SparseTSF | Lin et al. | 2024 | ICML 2024 | @26 |
| TimeBridge | Liu et al. | 2025 | 2025 | @26 |
| TimePerceiver | Oreshkin et al. | 2025 | 2025 | @26 |
| TimeRecipe | Chen et al. | 2025 | 2025 | @26 |
| xPatch | Lee et al. | 2024 | 2024 | @26 |

**All 22 P15 models** have 4 ablations per task (co, ct, ce, fu) but zero seed2 ablations.
Covered by cos2 scripts (6 scripts in `.slurm_scripts/phase15/cos2/`).

## 8. AutoFit Models (1 active — V739)

| Version | Status | Records | Notes |
|---|---|---|---|
| V1–V733 | Retired | — | Historical iterations |
| V734–V738 | **INVALID** | — | Oracle test-set leakage — NEVER use |
| **V739** | Active | 112 | Missing seed2/edgar_seed2 (resubmitted) |

## 9. Excluded / Disabled Models

| Model | Category | Reason | Status |
|---|---|---|---|
| Ridge | ml_tabular | Redundant with HistGradientBoosting | Excluded from registry |
| Lasso | ml_tabular | Redundant with HistGradientBoosting | Excluded from registry |
| SVR | ml_tabular | O(n²) complexity | Excluded from registry |
| KNN | ml_tabular | O(n²) complexity | Excluded from registry |
| LogisticRegression | ml_tabular | Classification only, not regression | Excluded from registry |
| Koopa | tslib_sota | NaN divergence | Registered but disabled |
| CycleNet | tslib_sota | Requires cycle_index feature | Registered but disabled |
| TQNet | tslib_sota | Requires cycle_index feature | Registered but disabled |
| Mamba | tslib_sota | Requires mamba_ssm (not installed) | Not registered |
| TiRex | tslib_sota | Requires NX-AI tirex (not on PyPI) | Not registered |
| NegativeBinomialGLM | ml_tabular | Structural convergence failure | @21 records, effectively excluded |
| AutoFitV734–V738 | autofit | Oracle test-set leakage | INVALID — removed from all comparisons |

## 10. Key Statistics

- **Wins concentrated**: Top 7 models hold 157/160 wins (98.1%)
- **Mean rank leaders**: PatchTST (4.28), NHITS (4.38), NBEATS (5.01), NBEATSx (5.81), ChronosBolt (7.42)
- **Category dominance**: deep_classical wins 97/160 (60.6%), transformer_sota 40/160 (25.0%), foundation 23/160 (14.4%)
- **Horizon specialization**: KAN@h=1/ NHITS@h=7 / NBEATS@h=14 / Chronos+NBEATS@h=30
- **Target specialization**: NBEATS@investors_count (75%), DeepNPTS@is_funded (66.7%), Chronos@funding_raised (33.8%)
- **Text effect**: HARMFUL (core_text wins only 11.7% of pairs)
- **EDGAR effect**: MIXED (wins 34.7% of pairs, target-dependent)
- **Seed stability**: avg |delta| = 0.138%, median = 0.000%

## 11. SOTA Coverage Audit (2026-03-23)

Cross-referenced against TSLib, NeuralForecast, and top venues (NeurIPS/ICML/ICLR/IJCAI/AAAI/KDD/ICDE/ACL/EMNLP 2024-2026).

### 11.1 Library Coverage
| Library | Total Models | We Have | Excluded | Missing | Coverage |
|---------|:-----------:|:-------:|:--------:|:-------:|:--------:|
| TSLib (thuml) | 41 | 37 | 3 | 1 | 90% |
| NeuralForecast (Nixtla) | 35 | 31 | 3 | 1 | 89% |

### 11.2 Already Excluded (with justification)
| Model | Reason |
|-------|--------|
| Koopa | NaN divergence on our data |
| Mamba | Requires `mamba_ssm` C extension (not on PyPI) |
| TiRex | Requires `NX-AI tirex` (not on PyPI) |
| CycleNet / TQNet | Requires `cycle_index` feature engineering |
| NegativeBinomialGLM | Structural OOM >640G |

### 11.3 Potentially Missing Models (updated 2026-03-23b)
| Priority | Model | Venue | Year | Status | Why it now matters |
|----------|-------|-------|------|--------|--------------------|
| **HIGH** | LightGTS | ICML | 2025 | Has code | Lightweight general TS model; unusually well aligned with the V740 single-model efficiency goal |
| **HIGH** | OLinear | NeurIPS | 2025 | Has code | Very strong low-cost orthogonal-domain linear baseline; could cover easy continuous/count cells cheaply |
| **HIGH** | LiPFormer | ICDE | 2025 | Has code | Lightweight patch-wise transformer with weak data enriching; attractive efficient alternative to PatchTST |
| **MEDIUM** | ElasTST | NeurIPS | 2024 | Has code | Elastic varied-horizon model; highly relevant to the ``one model for all horizons'' V740 constraint |
| **MEDIUM** | DIAN | IJCAI | 2024 | Has code | Decoupled invariant/variant attention; directly relevant to heterogeneous panel forecasting |
| **MEDIUM** | UniTS | NeurIPS | 2024 | Has code | Unified multi-task time-series model; important architectural reference for V740's condition-tokenized single-model design |
| **MEDIUM** | CASA | IJCAI | 2025 | Has code | Efficient CNN-autoencoder + score-based attention; attractive efficiency-first benchmark candidate |
| **MEDIUM** | Peri-midFormer | NeurIPS | 2024 | Has code | Period-informed decomposition; potentially useful decomposition-first baseline |
| **LOW** | QuantileFormer | IJCAI | 2025 | Has code | Useful if V740 grows a serious probabilistic / calibrated quantile branch |
| **LOW** | DERITS | IJCAI | 2024 | Has code | Non-stationarity via derivative-frequency transformation; more useful as a V740 module than as a must-have benchmark baseline |
| **LOW** | RAFT | ICML | 2025 | Has code | Retrieval-augmented forecasting; useful scientifically but less aligned with single-model simplicity |
| **LOW** | TimeDART | ICML | 2025 | Has code | Diffusion autoregressive transformer; likely too heavy for the current efficiency target |
| **LOW** | XLinear | NeuralForecast | 2026 | In NF | Trivial to add, but expected incremental value is limited |
| **LOW** | MambaSingleLayer | TSLib | 2026 | In TSLib | Still likely blocked by `mamba_ssm` dependency issues |

### 11.3a No Longer Missing

- **SAMformer (ICML 2024 oral)** is now locally integrated as a first custom lightweight wrapper:
  - implementation path: `src/narrative/block3/models/samformer_model.py`
  - registry path: `src/narrative/block3/models/deep_models.py`
  - current status:
    - import + synthetic `fit/predict` smoke passed
    - direct harness smoke currently exits `137` in local interactive runs
    - freeze-backed micro-smoke runs end-to-end but still shows constant-output risk on a tiny real slice

### 11.3b Emerging But Not Yet Registry-Priority

These directions are worth tracking for V740 methodology, but are not yet top benchmark-addition priorities:

- **TimeEmb** (NeurIPS 2025): static-dynamic disentanglement, strong design signal for V740
- **QDF** (ICLR 2026): quadratic-form weighted learning objective for harder long-horizon structure
- **DistDF** (ICLR 2026): distribution-level alignment for multistep forecasting
- **JAPAN** (ICLR 2026): density-aware conformal prediction regions, strong uncertainty/calibration signal
- **TimeDiT** (2025): diffusion-style general TSFM, scientifically relevant but too heavy for alpha
- **FLAIRR-TS** (Findings of EMNLP 2025): retrieval + iterative refinement, useful later as auxiliary memory rather than first-wave benchmark addition
- **LightGTS-Cov** (2026 arXiv): promising covariate-aware extension, but not yet benchmark-priority until the paper/release stabilizes

### 11.4 Coverage Verdict
**优秀覆盖，但不再意味着“可以停止扩展”**。当前主流库覆盖仍然很强，现有 benchmark 已足够支撑可靠结论；但从 V740 的单模型冠军目标来看，仍有几类高价值缺失基线值得补充，尤其是 `LightGTS`、`OLinear`、`LiPFormer`、`ElasTST`、`DIAN` 和 `UniTS`。`SAMformer` 已经从“缺失模型”变成了“已接入、待正式 benchmark”的第一批新增效率型基线。这些模型的价值不在于“再堆几个名字”，而在于它们更贴近 V740 想要达到的效率-精度 Pareto 前沿，或者直接为 V740 的单模型条件化设计提供结构启发。

## Code References

- Registry: `src/narrative/block3/models/registry.py`
- Statistical: `src/narrative/block3/models/statistical.py`
- Deep classical: `src/narrative/block3/models/deep_models.py`
- Transformer SOTA: `src/narrative/block3/models/deep_models.py`
- Foundation: `src/narrative/block3/models/foundation_wrapper.py`
- Irregular: `src/narrative/block3/models/irregular_models.py`
- ML tabular: `src/narrative/block3/models/traditional_ml.py`
- TSLib: `src/narrative/block3/models/tslib_models.py`
- AutoFit: `src/narrative/block3/models/nf_adaptive_champion.py`
- Memory requirements: `docs/references/MEMORY_REQUIREMENTS.md`
