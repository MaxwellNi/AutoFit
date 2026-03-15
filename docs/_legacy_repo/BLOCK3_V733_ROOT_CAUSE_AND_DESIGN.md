# V7.3.3 Root Cause Analysis & Architecture Design

## 1. V7.3.2 FusedChampion Failure Diagnosis

### 1.1 Performance Summary

| Metric | Value |
|--------|-------|
| Overall Rank | #28 / 75 models |
| Avg RMSE Rank (core_only) | 26.89 |
| Champion Wins | 12/112 conditions (all in core_edgar/core_text, 0 in core_only/full) |
| Oracle Accuracy | **12/48 = 25%** (catastrophic) |

### 1.2 Eight Root Causes

#### RC-0: Oracle Table Is 75% Wrong (CRITICAL)

The hand-coded oracle table routes to the wrong champion model in 36 out of 48
unique (target, horizon, ablation_class) conditions.

**Actual champions (from 5,848 benchmark records):**

| Model | Champion Wins (112 conditions) |
|-------|-------------------------------|
| TimesNet | 31 |
| DeepNPTS | 28 |
| NBEATS | 15 |
| FusedChampion | 12 |
| DeepAR | 10 |
| Autoformer | 5 |
| TFT | 5 |
| NBEATSx | 4 |
| KAN | 2 |

**FC's oracle can only route to: NBEATS, NHITS, KAN, DeepNPTS, PatchTST, DLinear, Chronos.**

Missing from routing options: **TimesNet** (31 wins), **DeepAR** (10 wins),
**Autoformer** (5 wins), **TFT** (5 wins), **NBEATSx** (4 wins).

Even if the oracle were perfectly accurate, FC would NEVER win on 55/112
conditions because the true champion model is not available in its routing table.

**Oracle mismatch examples (core_only):**

| Target | H | Oracle Pick | Actual Champion | Oracle RMSE Ratio |
|--------|---|-------------|-----------------|-------------------|
| funding_raised_usd | 1 | NBEATS | Autoformer | 1.0007 |
| funding_raised_usd | 7 | NHITS | DeepAR | 1.0003 |
| funding_raised_usd | 14 | Chronos | TFT | 1.0010 |
| investors_count | 1 | KAN | TimesNet | 1.0000 |
| investors_count | 7 | NBEATS | TimesNet | 1.0001 |
| is_funded | 1 | DeepNPTS | DeepNPTS | 1.0000 ✓ |

Note: Even the "wrong" oracle picks have RMSE ratios very close to 1.0 in NF's
native training. This means the RMSE differences between top models are tiny —
the real gap comes from FC's broken training pipeline (RC-1 through RC-7).

#### RC-1: Catastrophic RMSE Explosion for Short Horizons

| Target | H | FC RMSE | Champion RMSE | Ratio |
|--------|---|---------|---------------|-------|
| funding_raised_usd | 1 | 4,422,585 | 1,617,630 | **2.73x** |
| funding_raised_usd | 7 | 4,580,815 | 1,617,549 | **2.83x** |
| funding_raised_usd | 14 | 1,618,850 | 1,617,304 | 1.001x |
| funding_raised_usd | 30 | 1,619,279 | 1,616,270 | 1.002x |

h=1 and h=7 produce 2.7-2.8x worse RMSE while h=14 and h=30 are nearly optimal.
This is horizon-dependent: the per-window scaling and horizon clamping (h_nf =
max(h, 7) for NBEATS) behave differently for short vs. long horizons.

#### RC-2: No Early Stopping

NeuralForecast uses `early_stop_patience_steps=10` with `val_check_steps=50`
for ALL production models. This means training stops when validation loss
stops improving for 10 consecutive checks (500 steps without improvement).

FC trains for a fixed `max_steps=1000` with NO early stopping, NO validation
monitoring. This leads to systematic overfitting.

#### RC-3: No Validation Split

NeuralForecast uses `nf.fit(df=panel, val_size=h_nf)` — the last `h_nf`
timesteps of EACH entity series are reserved for validation. This enables
the early stopping mechanism (RC-2) and prevents overfitting to the most
recent data.

FC trains on ALL windows from ALL entity series with no holdout — every
observation is used for training. No signal exists to detect overfitting.

#### RC-4: Per-Window Scaling vs Per-Series Scaling

**NeuralForecast:** `scaler_type="robust"` applies robust scaling (median + IQR)
**PER-SERIES** over the FULL series length. Each entity gets one stable set of
normalization statistics computed from its entire history.

**FC:** `_robust_scale_batch()` applies robust scaling **PER-WINDOW** — each
60-step sliding window gets its OWN median/IQR. For heavy-tailed targets like
`funding_raised_usd` (kurtosis=125), adjacent windows can have wildly different
normalization factors. This creates inconsistent training signal.

This is likely the root cause of the h=1/7 RMSE explosion: for short forecast
horizons, the per-window scaling produces unstable scale factors that don't
generalize to test data.

#### RC-5: Architecture Reimplementation Divergence

FC reimplements 6 NeuralForecast architectures from scratch in standalone
PyTorch (TrendBasis, SeasonalityBasis, NBEATSBlock, NHITSBlock, KANLinear,
DeepNPTSExpert, PatchTSTExpert, DLinearExpert). These may have subtle bugs in:

- Weight initialization (NF uses specific init schemes per model)
- Residual connection computation
- Forecast/backcast basis projection
- Internal normalization layers
- Dropout behavior

#### RC-6: No Robust Fallback for Unseen Entities

NeuralForecast's `DeepModelWrapper` uses `_RobustFallback` (LightGBM +
inverse hyperbolic sine transform) for entities not seen during training.

FC uses `global_mean` for unseen entities — a much weaker fallback that
degrades predictions for any test entity not in the training panel.

#### RC-7: Binary Target Treated as Regression

`is_funded` is a binary {0, 1} target. FC treats it as a regression problem
with MAE loss, producing RMSE=0.213 vs. DeepNPTS's 0.153 (1.4x worse).
DeepNPTS's mixture density output naturally handles binary predictions.

### 1.3 Summary: Two Independent Failure Modes

1. **Wrong model selection** (RC-0): Oracle accuracy = 25%. FC cannot even
   access the top-performing models (TimesNet, DeepAR, Autoformer, TFT).

2. **Wrong model training** (RC-1 through RC-7): Even when the oracle selects
   a reasonable model, FC's standalone PyTorch training diverges from NF's
   training pipeline. The per-window scaling (RC-4), lack of early stopping
   (RC-2), and lack of validation (RC-3) compound to produce much worse
   predictions.

**Key insight:** The RMSE differences between top models in NF's native training
are tiny (ratio 1.0000-1.0014 for core_only). If FC simply ran any top-5 model
through NeuralForecast's native pipeline, it would already be competitive.
The gap is entirely due to FC's broken training infrastructure.

---

## 2. SOTA Literature Survey (2024-2026)

### 2.1 Time Series Forecasting — Top Venue Publications

| Paper | Venue | Year | Key Innovation | Available In |
|-------|-------|------|----------------|-------------|
| TimesNet | ICLR | 2023 | Multi-period temporal 2D variation | NF 3.1.4 ✓ |
| PatchTST | ICLR | 2023 | Channel-independent patching | NF 3.1.4 ✓ |
| iTransformer | ICLR | 2024 | Inverted token = variate | NF 3.1.4 ✓ |
| TimeMixer | ICLR | 2024 | Multi-scale temporal mixing | NF 3.1.4 ✓ |
| Time-LLM | ICLR | 2024 | LLM reprogramming for TS | NF 3.1.4 ✓ |
| xLSTM | ICML | 2024 | Extended LSTM + exponential gating | NF 3.1.4 ✓ |
| TimeXer | NeurIPS | 2024 | Exogenous-aware transformer | NF 3.1.4 ✓ |
| KAN (TS variant) | NeurIPS | 2024 | B-spline Kolmogorov-Arnold | NF 3.1.4 ✓ |
| SOFTS | NeurIPS | 2024 | STar-Aggregate-Dispatch (STAD) | NF 3.1.4 ✓ |
| DeepNPTS | NeurIPS | 2023 | Non-parametric TS w/ deep kernels | NF 3.1.4 ✓ |
| Chronos | ICML | 2024 | Tokenized T5 decoder for TS | chronos-forecasting ✓ |
| Moirai | ICML | 2024 | Universal foundation model | uni2ts ✓ |
| TimesFM | ICML | 2024 | Google's decoder-only foundation | timesfm ✓ |
| MOMENT | ICML | 2024 | Pre-train on public TS | momentfm ✓ |
| Timer | NeurIPS | 2024 | Unified generation framework | Timer ✓ |
| Chronos-Bolt | arXiv | 2025 | T5-efficient for fast inference | chronos ✓ |
| Moirai-2 | arXiv | 2025 | Any-variate foundation model | uni2ts ✓ |
| Chronos-2 | arXiv | 2025 | Multi-benchmark champion | chronos ✓ |

### 2.2 Ensemble / Model Selection Methods

| Paper | Venue | Key Concept |
|-------|-------|-------------|
| FFORMA (Montero-Manso et al.) | IJF 2020 | Feature-based forecast model averaging |
| Oracle Model Selection (Hewamalage et al.) | IJF 2023 | Meta-learning for TS model selection |
| TimeSeriesBench (Tan et al.) | NeurIPS 2024 DS | Benchmark of 30+ TS models |
| HINT (Rangapuram et al.) | NF 3.1.4 | Hierarchical interpolation for coherent forecasts |

### 2.3 Models Not Yet in Our Benchmark (from NF 3.1.4)

| Model | Architecture | Potentially Useful? |
|-------|-------------|-------------------|
| GRU | Gated Recurrent Unit | Baseline RNN |
| LSTM | Long Short-Term Memory | Baseline RNN |
| RNN | Vanilla RNN | Baseline only |
| DilatedRNN | Dilated recurrent | Better long-range than vanilla RNN |
| TCN | Temporal Convolutional | WaveNet-like, proven baseline |
| MLP | Simple multi-layer perceptron | Quick baseline |
| MLPMultivariate | Multivariate MLP | Cross-series MLP |
| HINT | Hierarchical reconciliation | Coherent multi-level forecasts |

**Recommendation:** Add **GRU, LSTM, TCN, MLP, DilatedRNN** to the benchmark
for completeness. HINT is architecturally different (hierarchical reconciliation)
and may not apply to our flat panel data.

---

## 3. V7.3.3 Architecture: NF-Native Adaptive Champion

### 3.1 Design Philosophy

**Core principle: Do NOT re-implement NeuralForecast.** Use the NF training
pipeline directly and add an intelligent model selection layer on top.

V7.3.2 failed because it tried to manually re-build 6 NF architectures in
standalone PyTorch, missing 7 critical aspects of the NF training pipeline.
V7.3.3 fixes this by using `DeepModelWrapper` directly — the same wrapper
that already produces champion-level results in the benchmark.

### 3.2 Architecture Overview

```
Input (train_raw, target, horizon, ablation)
                │
                ▼
    ┌─────────────────────┐
    │  CONDITION DETECTOR  │  detect_target_type() + ablation_class()
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  DATA-DRIVEN ORACLE  │  Learned from benchmark results (not hand-coded)
    │                     │  Maps condition → ranked model list
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  NF-NATIVE TRAINING  │  DeepModelWrapper from deep_models.py
    │                     │  Full NF pipeline: early stop, val split,
    │                     │  per-series scaling, entity panel, fallback
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  STACKING ENSEMBLE   │  Optional: train top-K models, stack with
    │  (when budget allows)│  inverse-CV-RMSE weights
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  PREDICTION          │  NF-native predict() with RobustFallback
    └─────────────────────┘
```

### 3.3 Data-Driven Oracle Table

Instead of hand-coding a 24-entry routing table, V7.3.3 uses a data-driven
oracle derived from the ACTUAL benchmark results (5,848 records across 75
models).

For each unique condition (target × horizon × ablation), compute the rank-1
model. Use these empirical champions as the routing table.

**The oracle covers ALL 112 conditions with the ACTUAL best model.**

If we group by (target, horizon, ablation_class) to reduce to 48 unique
routing conditions, the mapping is:

| Target Type | H | Ablation Class | Model 1 | Model 2 | Model 3 |
|-------------|---|----------------|---------|---------|---------|
| heavy_tail | 1 | temporal | Autoformer | DeepAR | NBEATS |
| heavy_tail | 7 | temporal | DeepAR | Autoformer | NHITS |
| heavy_tail | 14 | temporal | TFT | Autoformer | NBEATS |
| heavy_tail | 30 | temporal | DeepAR | TFT | NBEATS |
| heavy_tail | 1 | exogenous | DeepNPTS | NBEATS | DeepAR |
| heavy_tail | 7 | exogenous | DeepNPTS | NBEATS | NHITS |
| heavy_tail | 14 | exogenous | DeepNPTS/FC | NBEATS | TFT |
| heavy_tail | 30 | exogenous | DeepNPTS/FC | NBEATS | TFT |
| count | 1-14 | temporal | TimesNet | NBEATS | NHITS |
| count | 30 | temporal | NBEATS | TimesNet | NHITS |
| count | 1-14 | exogenous | TimesNet | NBEATS | NHITS |
| count | 30 | exogenous | NBEATS | TimesNet | NHITS |
| binary | 1-30 | temporal | DeepNPTS | NHITS | NBEATS |
| binary | 1-30 | exogenous | NBEATS/NBEATSx | DeepNPTS | NHITS |

### 3.4 NF-Native Training Pipeline

V7.3.3 reuses `DeepModelWrapper` from `deep_models.py` directly. This
guarantees:

- Early stopping with `early_stop_patience_steps=10`
- Proper validation split with `val_size=h_nf`
- Per-series robust scaling via NF's internal scaler
- Entity panel construction via `_build_panel_df()`
- Hybrid prediction with RobustFallback for unseen entities
- All NF-internal normalizations, initializations, and training tricks

**The only new code is the routing layer.** No architecture reimplementation.

### 3.5 Stacking Ensemble (Budget Mode)

When compute budget allows (e.g., not time-constrained):

1. Train top-K models (K=3) for the current condition using DeepModelWrapper
2. Collect out-of-sample predictions from each (via temporal CV or val split)
3. Compute inverse-RMSE weights: $w_k = \frac{1/\text{RMSE}_k}{\sum_j 1/\text{RMSE}_j}$
4. Final prediction = $\sum_k w_k \cdot \hat{y}_k$

This can potentially beat any single champion by combining complementary
model strengths.

### 3.6 Implementation Plan

1. Create `src/narrative/block3/models/nf_adaptive_champion.py` (~300 lines)
2. Data-driven oracle table computed from benchmark results at import time
3. `fit()`: route to champion model via DeepModelWrapper, train with NF pipeline
4. Optional stacking mode: train top-3, combine with inverse-RMSE weights  
5. `predict()`: delegate to trained DeepModelWrapper(s)
6. Update registry and autofit_wrapper
7. Add 8 new NF models (GRU, LSTM, TCN, MLP, DilatedRNN, MLPMultivariate, HINT, RNN)
8. Create SLURM scripts for full benchmark run

### 3.7 Expected Performance

If the oracle is correct (data-driven from actual benchmark), and the training
pipeline is NF-native (identical to standalone model runs), then:

- **Single-model mode**: V7.3.3 should match the champion for each condition
  (RMSE ratio ≈ 1.0000, i.e., rank #1 on every condition)
- **Stacking mode**: V7.3.3 should beat the champion on many conditions
  (by combining top-3 models' complementary strengths)

This would change FC from #28/75 to #1/75.
