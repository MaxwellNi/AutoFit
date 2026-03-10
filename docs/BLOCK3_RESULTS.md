# Block 3 Benchmark Results

> Last updated: 2026-03-10 (Phase 9 — deep re-audit, 4 new bugs found, V736 ranking analysis)
> Canonical results dir: `runs/benchmarks/block3_phase9_fair/`
> Phase 7/8 results are **DEPRECATED** (4 critical bugs fixed)
> **Data integrity audit**: RE-AUDITED 2026-03-10 — 4 new critical findings (see §Deep Audit below)

**Generated**: 2026-03-09
**Benchmark Dir**: `block3_phase9_fair`
**Total Records**: 8198
**Complete Models (104/104 conditions)**: 76
**Partial Models**: 17

## Overview

| Metric | Value |
|--------|-------|
| Raw records | 8198 |
| Complete models (104/104) | 76 |
| Partial models (<104) | 17 |
| Categories | autofit, deep_classical, foundation, irregular, ml_tabular, statistical, transformer_sota, tslib_sota |
| Tasks | task1_outcome, task2_forecast, task3_risk_adjust |

## Full Leaderboard (76 complete models, ranked by average MAE)

| Rank | Model | Category | Avg MAE | Integrity |
|-----:|-------|----------|--------:|-----------|
| 1 | Chronos | foundation | 159,732 | ✅ genuine |
| 2 | PatchTST | transformer_sota | 159,829 | ✅ genuine |
| 3 | ChronosBolt | foundation | 159,873 | ✅ genuine |
| 4 | NHITS | deep_classical | 159,983 | ✅ genuine |
| 5 | TFT | deep_classical | 160,056 | ✅ genuine |
| 6 | NBEATS | deep_classical | 160,176 | ✅ genuine |
| 7 | NBEATSx | transformer_sota | 160,176 | ⚠️ near-duplicate of NBEATS (73.1% identical) |
| 8 | DeepAR | deep_classical | 160,218 | ✅ genuine |
| 9 | TimesNet | transformer_sota | 160,396 | ✅ genuine |
| 10 | Informer | transformer_sota | 160,400 | ✅ genuine |
| 11 | KAN | transformer_sota | 160,663 | ✅ genuine |
| 12 | TimeMixer | transformer_sota | 160,691 | ✅ genuine |
| 13 | BiTCN | transformer_sota | 160,758 | ✅ genuine |
| 14 | RMoK | transformer_sota | 160,926 | ✅ genuine |
| 15 | NLinear | transformer_sota | 161,032 | ✅ genuine |
| 16 | FEDformer | transformer_sota | 161,649 | ✅ genuine |
| 17 | TiDE | transformer_sota | 161,729 | ✅ genuine |
| 18 | DLinear | transformer_sota | 161,856 | ✅ genuine |
| 19 | GRU | deep_classical | 162,046 | ✅ genuine |
| 20 | TSMixer | transformer_sota | 162,064 | ✅ genuine |
| 21 | LSTM | deep_classical | 162,114 | ✅ genuine |
| 22 | iTransformer | transformer_sota | 162,189 | ✅ genuine |
| 23 | MLP | deep_classical | 162,387 | ✅ genuine |
| 24 | TCN | deep_classical | 162,412 | ✅ genuine |
| 25 | DilatedRNN | deep_classical | 162,594 | ✅ genuine |
| 26 | SOFTS | transformer_sota | 162,891 | ✅ genuine |
| **27** | **AutoFitV736** | **autofit** | **163,465** | **✅ genuine** |
| 28 | TimeMoE | foundation | 163,486 | ⚠️ 98-99% identical to Timer/MOMENT |
| 29 | Timer | foundation | 163,486 | ⚠️ 98-99% identical to TimeMoE/MOMENT |
| 30 | MOMENT | foundation | 163,486 | ⚠️ 98-99% identical to Timer/TimeMoE |
| **31** | **AutoFitV734** | **autofit** | **165,162** | **✅ genuine** |
| 32 | LightGBMTweedie | ml_tabular | 165,728 | ✅ genuine (4/26 horizon-invariant — VARIES by horizon) |
| 33 | Sundial | foundation | 165,729 | ❌ silent context-mean fallback |
| 34 | TimesFM2 | foundation | 165,729 | ❌ silent context-mean fallback |
| 35 | LagLlama | foundation | 165,729 | ❌ silent context-mean fallback |
| 36 | Moirai | foundation | 165,729 | ❌ silent context-mean fallback |
| 37 | MoiraiLarge | foundation | 165,729 | ❌ silent context-mean fallback |
| 38 | Moirai2 | foundation | 165,729 | ❌ silent context-mean fallback |
| 39 | Autoformer | transformer_sota | 166,652 | ✅ genuine |
| **40** | **AutoFitV735** | **autofit** | **167,038** | **✅ genuine** |
| 41 | DeepNPTS | transformer_sota | 167,862 | ✅ genuine |
| 42 | RandomForest | ml_tabular | 173,229 | ✅ genuine (horizon-invariant by design) |
| 43 | XGBoostPoisson | ml_tabular | 178,934 | ✅ genuine (horizon-invariant by design) |
| 44 | TSMixerx | transformer_sota | 185,554 | ✅ genuine |
| 45 | TimesFM | foundation | 189,242 | ⚠️ horizon-invariant (23/26) |
| 46 | LightGBM | ml_tabular | 196,848 | ✅ genuine (5/26 horizon-invariant — VARIES by horizon) |
| 47 | XGBoost | ml_tabular | 201,999 | ✅ genuine (horizon-invariant by design) |
| 48 | ExtraTrees | ml_tabular | 206,963 | ✅ genuine (horizon-invariant by design) |
| 49 | HistGradientBoosting | ml_tabular | 208,779 | ✅ genuine (horizon-invariant by design) |
| 50 | CatBoost | ml_tabular | 242,324 | ✅ genuine (horizon-invariant by design) |
| 51 | VanillaTransformer | transformer_sota | 260,812 | ✅ genuine |
| 52 | BRITS | irregular | 343,295 | ✅ genuine (horizon-invariant) |
| 53 | GRU-D | irregular | 343,728 | ✅ genuine (horizon-invariant) |
| 54 | SAITS | irregular | 343,739 | ✅ genuine (horizon-invariant) |
| 55 | HistoricAverage | statistical | 640,493 | ✅ baseline |
| 56 | CSDI | irregular | 902,624 | ⚠️ 44/104 fallback |
| 57 | xLSTM | transformer_sota | 902,705 | ❌ 100% fallback (training crash) |
| 58 | TimeLLM | transformer_sota | 902,705 | ❌ 100% fallback (training crash) |
| 59 | AutoCES | statistical | 902,705 | ❌ 100% fallback (training crash) |
| 60 | MeanPredictor | ml_tabular | 902,705 | ✅ baseline (mean by design) |
| 61 | StemGNN | transformer_sota | 902,705 | ❌ 40/104 fallback (training crash) |
| 62 | TimeXer | transformer_sota | 902,705 | ❌ 40/104 fallback (training crash) |
| 63-76 | *(statistical models)* | statistical | 1.58M-13.1M | ✅ genuine |

### AutoFit Rankings Summary

| Variant | Overall | task1 | task2 | task3 |
|---------|---------|-------|-------|-------|
| **V736** | **#27/76** (163,465) | #26 (128,663) | #27 (192,994) | #30 (193,694) |
| V734 | #31/76 (165,162) | #31 (129,913) | #31 (194,869) | #31 (196,053) |
| V735 | #40/76 (167,038) | #39 (131,293) | #39 (196,940) | #41 (198,656) |

### Partial Models (still running)

| Model | Records | Category |
|-------|--------:|----------|
| TimeFilter | 32/104 | tslib_sota |
| MultiPatchFormer | 32/104 | tslib_sota |
| MSGNet | 32/104 | tslib_sota |
| PAttn | 32/104 | tslib_sota |
| MambaSimple | 32/104 | tslib_sota |
| Crossformer | 32/104 | tslib_sota |
| ETSformer | 20/104 | tslib_sota |
| LightTS | 20/104 | tslib_sota |
| Pyraformer | 20/104 | tslib_sota |
| Reformer | 20/104 | tslib_sota |
| NegativeBinomialGLM | 16/104 | ml_tabular |
| FreTS | 1/104 | tslib_sota |
| MICN | 1/104 | tslib_sota |
| SegRNN | 1/104 | tslib_sota |
| NonstationaryTransformer | 1/104 | tslib_sota |
| FiLM | 1/104 | tslib_sota |
| SCINet | 1/104 | tslib_sota |

## task1_outcome

### autofit

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| AutoFitV734 | 384.2K | 384.2K | 384.4K | 384.3K | ✅ |
| AutoFitV735 | 384.2K | 384.2K | 384.3K | 384.3K | ✅ |
| AutoFitV736 | 389.3K | 389.2K | 394.6K | 395.5K | ✅ |

### deep_classical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATS | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| NHITS | 374.6K | 374.4K | 376.7K | 375.5K | ✅ |
| TFT | 374.8K | 374.7K | 375.4K | 377.0K | ✅ |
| DeepAR | 374.9K | 375.8K | 375.7K | 377.1K | ✅ |
| GRU | 384.3K | 384.6K | 384.2K | 385.3K | ✅ |
| LSTM | 384.7K | 384.3K | 384.4K | 385.7K | ✅ |
| DilatedRNN | 385.0K | 385.2K | 386.3K | 387.1K | ✅ |
| MLP | 385.2K | 385.2K | 385.9K | 385.2K | ✅ |
| TCN | 385.8K | 384.7K | 385.0K | 386.3K | ✅ |

### foundation

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| Chronos | 374.8K | 374.8K | 374.7K | 374.6K | ✅ |
| ChronosBolt | 375.0K | 375.0K | 375.0K | 375.0K | ✅ |
| MOMENT | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| TimeMoE | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Timer | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Sundial | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM2 | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM | 447.4K | 447.4K | 447.4K | 447.4K | ✅ |

### irregular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| GRU-D | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| SAITS | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |

### statistical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| SF_SeasonalNaive | 4.06M | 4.08M | 4.08M | 4.08M | ✅ |
| AutoETS | 4.08M | 4.09M | 4.09M | 4.10M | ✅ |
| AutoARIMA | 4.13M | 4.15M | 4.18M | 4.24M | ✅ |
| MSTL | 4.14M | 4.23M | 4.13M | 4.13M | ✅ |
| AutoTheta | 4.16M | 4.27M | 4.39M | 4.67M | ✅ |

### transformer_sota

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATSx | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| PatchTST | 374.6K | 374.7K | 375.1K | 375.5K | ✅ |
| TimesNet | 374.9K | 375.3K | 376.2K | 378.7K | ✅ |
| Informer | 375.2K | 375.5K | 377.7K | 376.8K | ✅ |
| iTransformer | 376.3K | 376.6K | 376.0K | 376.2K | ✅ |
| FEDformer | 377.7K | 378.8K | 379.0K | 381.4K | ✅ |
| Autoformer | 378.4K | 395.0K | 402.7K | 388.2K | ✅ |
| TiDE | 378.4K | 384.0K | 377.9K | 377.4K | ✅ |
| TSMixer | 379.5K | 380.2K | 379.4K | 381.5K | ✅ |
| KAN | 384.5K | 385.1K | 386.0K | 387.3K | ✅ |
| TimeMixer | 385.3K | 385.2K | 386.2K | 386.2K | ✅ |
| NLinear | 385.6K | 387.3K | 386.9K | 386.6K | ✅ |
| RMoK | 386.1K | 386.0K | 386.8K | 386.2K | ✅ |
| BiTCN | 386.2K | 386.1K | 386.2K | 385.4K | ✅ |
| SOFTS | 386.4K | 386.5K | 387.8K | 403.1K | ✅ |
| DLinear | 388.1K | 388.9K | 388.3K | 389.0K | ✅ |
| DeepNPTS | 395.3K | 393.1K | 394.7K | 403.8K | ✅ |
| TSMixerx | 397.9K | 406.5K | 496.8K | 476.8K | ✅ |
| VanillaTransformer | 612.9K | 612.9K | 612.9K | 612.9K | ⚠️ fallback |
| TimeLLM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| xLSTM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| StemGNN | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| TimeXer | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |

## task2_forecast

### autofit

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| AutoFitV734 | 384.2K | 384.2K | 384.4K | 384.3K | ✅ |
| AutoFitV735 | 384.2K | 384.2K | 384.3K | 384.3K | ✅ |
| AutoFitV736 | 389.3K | 389.2K | 394.6K | 395.5K | ✅ |

### deep_classical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATS | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| NHITS | 374.6K | 374.4K | 376.7K | 375.5K | ✅ |
| TFT | 374.8K | 374.7K | 375.4K | 377.0K | ✅ |
| DeepAR | 374.9K | 375.8K | 375.7K | 377.1K | ✅ |
| GRU | 384.3K | 384.6K | 384.2K | 385.3K | ✅ |
| LSTM | 384.7K | 384.3K | 384.4K | 385.7K | ✅ |
| DilatedRNN | 385.0K | 385.2K | 386.3K | 387.1K | ✅ |
| MLP | 385.2K | 385.2K | 385.9K | 385.2K | ✅ |
| TCN | 385.8K | 384.7K | 385.0K | 386.3K | ✅ |

### foundation

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| Chronos | 374.8K | 374.8K | 374.7K | 374.6K | ✅ |
| ChronosBolt | 375.0K | 375.0K | 375.0K | 375.0K | ✅ |
| TimeMoE | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Timer | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| MOMENT | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Sundial | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM2 | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM | 447.4K | 447.4K | 447.4K | 447.4K | ✅ |

### irregular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| GRU-D | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| SAITS | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |

### statistical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| SF_SeasonalNaive | 4.06M | 4.08M | 4.08M | 4.08M | ✅ |
| AutoETS | 4.08M | 4.09M | 4.09M | 4.10M | ✅ |
| AutoARIMA | 4.13M | 4.15M | 4.18M | 4.24M | ✅ |
| MSTL | 4.14M | 4.23M | 4.13M | 4.13M | ✅ |
| AutoTheta | 4.16M | 4.27M | 4.39M | 4.67M | ✅ |

### transformer_sota

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATSx | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| PatchTST | 374.6K | 374.7K | 375.1K | 375.5K | ✅ |
| KAN | 374.8K | 375.4K | 376.3K | 377.6K | ✅ |
| TimesNet | 374.9K | 375.3K | 376.2K | 378.7K | ✅ |
| Informer | 375.2K | 375.5K | 377.7K | 376.8K | ✅ |
| TimeMixer | 375.6K | 375.5K | 376.5K | 376.5K | ✅ |
| NLinear | 375.9K | 377.6K | 377.2K | 376.9K | ✅ |
| RMoK | 376.4K | 376.3K | 377.1K | 376.5K | ✅ |
| BiTCN | 376.5K | 376.4K | 376.5K | 375.7K | ✅ |
| iTransformer | 376.5K | 377.1K | 390.1K | 384.1K | ✅ |
| SOFTS | 376.6K | 376.8K | 378.0K | 393.4K | ✅ |
| FEDformer | 377.7K | 378.8K | 379.0K | 381.4K | ✅ |
| Autoformer | 378.4K | 395.0K | 402.7K | 388.2K | ✅ |
| DLinear | 378.4K | 379.2K | 378.6K | 379.2K | ✅ |
| TiDE | 378.4K | 384.0K | 377.9K | 377.4K | ✅ |
| TSMixer | 379.5K | 380.2K | 379.4K | 381.5K | ✅ |
| TSMixerx | 388.2K | 396.8K | 487.1K | 467.0K | ✅ |
| DeepNPTS | 395.3K | 393.1K | 394.7K | 403.8K | ✅ |
| VanillaTransformer | 612.9K | 612.9K | 612.9K | 612.9K | ⚠️ fallback |
| TimeLLM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| xLSTM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| TimeXer | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| StemGNN | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |

## task3_risk_adjust

### autofit

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| AutoFitV734 | 384.2K | 384.2K | 384.4K | 384.3K | ✅ |
| AutoFitV735 | 384.2K | 384.2K | 384.3K | 384.3K | ✅ |
| AutoFitV736 | 389.3K | 389.0K | 394.8K | 395.5K | ✅ |

### deep_classical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATS | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| NHITS | 374.6K | 374.4K | 376.7K | 375.5K | ✅ |
| TFT | 374.8K | 374.7K | 375.4K | 377.0K | ✅ |
| DeepAR | 374.9K | 375.8K | 375.7K | 377.1K | ✅ |
| GRU | 384.3K | 384.6K | 384.2K | 385.3K | ✅ |
| LSTM | 384.7K | 384.3K | 384.4K | 385.7K | ✅ |
| DilatedRNN | 385.0K | 385.2K | 386.3K | 387.1K | ✅ |
| MLP | 385.2K | 385.2K | 385.9K | 385.2K | ✅ |
| TCN | 385.8K | 384.7K | 385.0K | 386.3K | ✅ |

### foundation

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| Chronos | 374.8K | 374.8K | 374.7K | 374.6K | ✅ |
| ChronosBolt | 375.0K | 375.0K | 375.0K | 375.0K | ✅ |
| MOMENT | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| TimeMoE | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Timer | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Sundial | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM2 | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM | 447.4K | 447.4K | 447.4K | 447.4K | ✅ |

### irregular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| GRU-D | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| SAITS | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |

### statistical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| SF_SeasonalNaive | 4.06M | 4.08M | 4.08M | 4.08M | ✅ |
| AutoETS | 4.08M | 4.09M | 4.09M | 4.10M | ✅ |
| AutoARIMA | 4.13M | 4.15M | 4.18M | 4.24M | ✅ |
| MSTL | 4.14M | 4.23M | 4.13M | 4.13M | ✅ |
| AutoTheta | 4.16M | 4.27M | 4.39M | 4.67M | ✅ |

### transformer_sota

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATSx | 374.5K | 374.8K | 375.4K | 378.3K | ✅ |
| PatchTST | 374.6K | 374.7K | 375.1K | 375.5K | ✅ |
| KAN | 374.8K | 375.4K | 376.3K | 377.6K | ✅ |
| TimesNet | 374.9K | 375.3K | 376.2K | 378.7K | ✅ |
| Informer | 375.2K | 375.5K | 377.7K | 376.8K | ✅ |
| TimeMixer | 375.6K | 375.5K | 376.5K | 376.5K | ✅ |
| NLinear | 375.9K | 377.6K | 377.2K | 376.9K | ✅ |
| RMoK | 376.4K | 376.3K | 377.1K | 376.5K | ✅ |
| BiTCN | 376.5K | 376.4K | 376.5K | 375.7K | ✅ |
| SOFTS | 376.6K | 376.8K | 378.0K | 393.4K | ✅ |
| FEDformer | 377.7K | 378.8K | 379.0K | 381.4K | ✅ |
| iTransformer | 377.8K | 375.9K | 410.4K | 380.4K | ✅ |
| Autoformer | 378.4K | 395.0K | 402.7K | 388.2K | ✅ |
| DLinear | 378.4K | 379.2K | 378.6K | 379.2K | ✅ |
| TiDE | 378.4K | 384.0K | 377.9K | 377.4K | ✅ |
| TSMixer | 379.5K | 380.2K | 379.4K | 381.5K | ✅ |
| TSMixerx | 388.2K | 396.8K | 487.1K | 467.0K | ✅ |
| DeepNPTS | 395.3K | 393.1K | 394.7K | 403.8K | ✅ |
| VanillaTransformer | 612.9K | 612.9K | 612.9K | 612.9K | ⚠️ fallback |
| TimeLLM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| xLSTM | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| TimeXer | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| StemGNN | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |

## Training Time Summary

| Model | Avg (s) | Min (s) | Max (s) |
|-------|--------:|--------:|--------:|
| AutoFitV736 | 3140.9 | 52.9 | 39074.7 |
| iTransformer | 1452.0 | 927.9 | 1763.2 |
| MSTL | 1175.3 | 112.5 | 1981.8 |
| AutoARIMA | 667.5 | 284.4 | 1133.3 |
| SOFTS | 459.7 | 329.9 | 708.1 |
| GRU-D | 355.4 | 179.6 | 525.6 |
| FEDformer | 342.2 | 171.1 | 585.0 |
| RMoK | 278.3 | 165.1 | 426.0 |
| Autoformer | 265.4 | 152.3 | 429.0 |
| Informer | 251.4 | 113.5 | 332.0 |
| TimesNet | 237.0 | 125.8 | 386.1 |
| TimeMixer | 175.7 | 140.6 | 232.6 |
| TSMixer | 155.2 | 107.6 | 228.3 |
| TSMixerx | 144.3 | 113.0 | 223.4 |
| AutoTheta | 114.3 | 60.1 | 200.9 |
| TFT | 91.6 | 71.1 | 129.2 |
| AutoETS | 80.1 | 10.2 | 157.5 |
| VanillaTransformer | 73.6 | 50.2 | 178.6 |
| PatchTST | 70.6 | 35.7 | 141.3 |
| SAITS | 63.2 | 24.2 | 104.5 |
| DeepAR | 61.8 | 30.6 | 171.3 |
| DilatedRNN | 47.7 | 24.5 | 82.9 |
| AutoFitV735 | 46.7 | 8.3 | 138.1 |
| LSTM | 39.7 | 16.2 | 76.7 |
| TCN | 38.9 | 12.1 | 76.4 |
| GRU | 35.3 | 11.8 | 79.9 |
| TiDE | 34.6 | 20.3 | 96.6 |
| MLP | 33.8 | 9.2 | 70.3 |
| BiTCN | 31.8 | 21.8 | 79.7 |
| AutoFitV734 | 28.5 | 3.2 | 94.8 |
| KAN | 28.0 | 10.7 | 69.5 |
| Sundial | 27.6 | 8.1 | 48.6 |
| TimesFM2 | 24.9 | 7.0 | 45.6 |
| NHITS | 24.5 | 10.5 | 37.6 |
| NBEATS | 24.1 | 9.3 | 70.5 |
| NBEATSx | 24.0 | 9.3 | 73.8 |
| NLinear | 22.2 | 11.8 | 62.2 |
| DLinear | 21.5 | 7.8 | 63.9 |
| DeepNPTS | 21.3 | 14.1 | 36.0 |
| TimeXer | 15.0 | 7.0 | 56.2 |
| StemGNN | 11.5 | 4.7 | 48.1 |
| MOMENT | 11.0 | 5.5 | 54.5 |
| Timer | 8.3 | 2.9 | 50.9 |
| TimeMoE | 8.1 | 2.9 | 48.5 |
| Chronos | 7.9 | 2.7 | 30.9 |
| ChronosBolt | 7.2 | 2.6 | 12.5 |
| TimesFM | 7.1 | 2.0 | 46.7 |
| xLSTM | 6.8 | 1.8 | 46.2 |
| TimeLLM | 6.0 | 3.0 | 40.9 |
| SF_SeasonalNaive | 1.3 | 0.4 | 2.2 |

## Completion Matrix

Shows which task/category/ablation combinations have results.

| Task | Category | Ablation | Models |
|------|----------|----------|-------:|
| task1_outcome | autofit | core_edgar | 3 |
| task1_outcome | autofit | core_only | 2 |
| task1_outcome | autofit | core_text | 3 |
| task1_outcome | autofit | full | 3 |
| task1_outcome | deep_classical | core_edgar | 9 |
| task1_outcome | deep_classical | core_only | 9 |
| task1_outcome | deep_classical | core_text | 9 |
| task1_outcome | deep_classical | full | 9 |
| task1_outcome | foundation | core_edgar | 8 |
| task1_outcome | foundation | core_only | 8 |
| task1_outcome | foundation | core_text | 8 |
| task1_outcome | foundation | full | 8 |
| task1_outcome | irregular | core_edgar | 2 |
| task1_outcome | irregular | full | 2 |
| task1_outcome | statistical | core_edgar | 5 |
| task1_outcome | statistical | core_text | 5 |
| task1_outcome | statistical | full | 5 |
| task1_outcome | transformer_sota | core_edgar | 23 |
| task1_outcome | transformer_sota | core_only | 23 |
| task1_outcome | transformer_sota | core_text | 23 |
| task1_outcome | transformer_sota | full | 23 |
| task2_forecast | autofit | core_edgar | 3 |
| task2_forecast | autofit | core_only | 3 |
| task2_forecast | autofit | core_text | 3 |
| task2_forecast | autofit | full | 3 |
| task2_forecast | deep_classical | core_edgar | 9 |
| task2_forecast | deep_classical | core_only | 9 |
| task2_forecast | deep_classical | core_text | 9 |
| task2_forecast | deep_classical | full | 9 |
| task2_forecast | foundation | core_edgar | 8 |
| task2_forecast | foundation | core_only | 8 |
| task2_forecast | foundation | core_text | 8 |
| task2_forecast | foundation | full | 8 |
| task2_forecast | irregular | core_edgar | 2 |
| task2_forecast | irregular | full | 2 |
| task2_forecast | statistical | core_edgar | 5 |
| task2_forecast | statistical | core_text | 5 |
| task2_forecast | statistical | full | 5 |
| task2_forecast | transformer_sota | core_edgar | 23 |
| task2_forecast | transformer_sota | core_only | 23 |
| task2_forecast | transformer_sota | core_text | 23 |
| task2_forecast | transformer_sota | full | 23 |
| task3_risk_adjust | autofit | core_edgar | 3 |
| task3_risk_adjust | autofit | core_only | 3 |
| task3_risk_adjust | autofit | full | 2 |
| task3_risk_adjust | deep_classical | core_edgar | 9 |
| task3_risk_adjust | deep_classical | core_only | 9 |
| task3_risk_adjust | deep_classical | full | 9 |
| task3_risk_adjust | foundation | core_edgar | 8 |
| task3_risk_adjust | foundation | core_only | 8 |
| task3_risk_adjust | foundation | full | 8 |
| task3_risk_adjust | irregular | core_edgar | 2 |
| task3_risk_adjust | irregular | full | 2 |
| task3_risk_adjust | statistical | core_edgar | 5 |
| task3_risk_adjust | statistical | full | 5 |
| task3_risk_adjust | transformer_sota | core_edgar | 23 |
| task3_risk_adjust | transformer_sota | core_only | 23 |
| task3_risk_adjust | transformer_sota | full | 23 |

---

## Data Integrity Audit (2026-03-09) — VERIFIED

> All findings below verified empirically via `/tmp/verify_audit.py` against actual metrics data.
> Two earlier claims corrected: C6 (73.1% not 86.5%), C7 (LightGBM IS horizon-variant).

### Critical Finding A: 6 Foundation Models — Silent Context-Mean Fallback

**Models**: Sundial, TimesFM2, LagLlama, Moirai, MoiraiLarge, Moirai2

**Evidence**:
- All 6 produce IDENTICAL avg MAE = 165,728.53
- 26/26 condition groups have identical MAE across all horizons (h=1,7,14,30)
- Only 6 unique MAE values out of 104 records
- `fallback_fraction=0.0` (incorrectly reported — harness doesn't detect model-internal fallback)

**Root Cause**: Each model's `predict()` has `except Exception: preds.append(float(np.mean(ctx)))` — when model inference fails (import errors, GPU OOM, API incompatibility), the exception is silently caught and replaced with per-entity context mean. All 6 models share the same entity contexts → identical predictions.

**Additional bugs in Moirai/Moirai2**: `entry.samples.mean(axis=1)` averages across the prediction_length dimension (should be `axis=0`), collapsing horizon information even if models succeed. Plus MOMENT has hardcoded `forecast_horizon=7` in `_load_moment()`.

**Impact**: These 6 models' rankings (#33-38) are **invalid** — they represent per-entity context mean, not model predictions.

### Critical Finding B: 5 Models — 100% Training Crash → Global Mean Fallback

**Models**: AutoCES, StemGNN, TimeXer, xLSTM, TimeLLM

**Evidence**:
- All produce IDENTICAL MAE = 902,704.71 (same as MeanPredictor)
- `fallback_fraction=1.0` (correctly reported) for AutoCES, xLSTM, TimeLLM (104/104)
- StemGNN, TimeXer: `fallback_fraction=0.38` reported, but still 104/104 MAE identical to MeanPredictor

**Root Cause**: NeuralForecast training crashes → `_use_fallback=True` → predict returns `np.mean(self._last_y)` (global mean, not per-entity). This bypasses the fitted RobustFallback (LightGBM).

**Impact**: These 5 models' rankings (#57-62) are meaningless — identical to MeanPredictor baseline.

### Finding C: Timer ≈ TimeMoE ≈ MOMENT (98-99% identical)

- Timer/TimeMoE: 102/104 identical (98.1%)
- Timer/MOMENT: 103/104 identical (99.0%)
- All three show 23-24/26 horizon-invariant groups

**Root Cause**: These models partially succeed but mostly produce near-identical predictions. The `generate()` API may not properly interface with the time series models. Avg MAE = 163,486 (slightly better than Group A due to some genuine predictions).

### Finding D: NBEATSx ≈ NBEATS (73.1% identical) — CORRECTED

- 76/104 conditions produce identical MAE (corrected from earlier 86.5% claim)
- Remaining 28 differ by <1%

**Root Cause**: NBEATSx is NOT in `_NF_SUPPORTS_STATIC_EXOG` set → never receives exogenous EDGAR features → trains identically to NBEATS. This is a misconfiguration, not a model failure.

### Finding E: ml_tabular Horizon Invariance — CORRECTED

**Mostly invariant**: CatBoost, ExtraTrees, HistGradientBoosting, RandomForest, XGBoost, XGBoostPoisson — 26/26 horizon-invariant. This is by design (tree models predict static features without horizon dependency).

**NOT fully invariant**: LightGBM (5/26 invariant) and LightGBMTweedie (4/26 invariant) **DO vary by horizon**. Earlier claim that ALL ml_tabular are horizon-invariant was WRONG. LightGBM's native objective interacts with data partitioning to produce horizon-sensitive splits.

### Effective Leaderboard (excluding invalid/duplicate)

Removing 6 silent-fallback models (C1), 5 crash-fallback models (C2), 2 near-duplicates (TimeMoE, MOMENT ≈ Timer):

| Clean Rank | Model | Avg MAE |
|---:|-------|--------:|
| 1 | Chronos | 159,732 |
| 2 | PatchTST | 159,829 |
| 3 | ChronosBolt | 159,873 |
| 4 | NHITS | 159,983 |
| 5 | TFT | 160,056 |
| 6 | NBEATS | 160,176 |
| 7 | NBEATSx | 160,176 |
| 8 | DeepAR | 160,218 |
| 9 | TimesNet | 160,396 |
| 10 | Informer | 160,400 |
| 11 | KAN | 160,663 |
| 12 | TimeMixer | 160,691 |
| 13 | BiTCN | 160,758 |
| 14 | RMoK | 160,926 |
| 15 | NLinear | 161,032 |
| 16 | FEDformer | 161,649 |
| 17 | TiDE | 161,729 |
| 18 | DLinear | 161,856 |
| 19 | GRU | 162,046 |
| 20 | TSMixer | 162,064 |
| 21 | LSTM | 162,114 |
| 22 | iTransformer | 162,189 |
| 23 | MLP | 162,387 |
| 24 | TCN | 162,412 |
| 25 | DilatedRNN | 162,594 |
| 26 | SOFTS | 162,891 |
| **27** | **AutoFitV736** | **163,465** |
| 28 | Timer | 163,486 |
| **29** | **AutoFitV734** | **165,162** |
| 30 | LightGBMTweedie | 165,728 |
| ... | ... | ... |
| **37** | **AutoFitV735** | **167,038** |

### CRITICAL: Normalized Ranking (per-target min-max) — V736 is #8

The raw MAE ranking is **dominated by `funding_raised_usd` scale** (MAE ~380K) vs `investors_count` (~45) vs `is_funded` (~0.03). When normalizing each target to [0,1] before averaging:

| Norm Rank | Model | Norm Score |
|---:|-------|----------:|
| 1 | PatchTST | 0.0001 |
| 2 | NHITS | 0.0002 |
| 3 | DeepNPTS | 0.0008 |
| 4 | NBEATSx | 0.0011 |
| 5 | NBEATS | 0.0012 |
| 6 | DLinear | 0.0018 |
| 7 | MLP | 0.0019 |
| **8** | **AutoFitV736** | **0.0021** |
| 9 | AutoFitV734 | 0.0024 |
| 10 | ChronosBolt | 0.0025 |
| 11 | AutoFitV735 | 0.0028 |
| ... | ... | ... |
| 17 | Chronos | 0.0066 |

**Key insight**: V736 is **#8 normalized** (vs #27 raw). Chronos drops from #1 to #17. V736 excels on `investors_count` (#8) and `is_funded` (#8) but loses on `funding_raised_usd` (#27) due to EDGAR feature degradation in `core_edgar`/`full` ablations.

### V736 vs Chronos Head-to-Head

| Metric | V736 | Chronos |
|--------|------:|--------:|
| **Condition wins** | **65** | 39 |
| Per-target wins | **2/3** (investors, is_funded) | 1/3 (funding_raised) |
| core_only rank | #8 | #1 |
| core_text rank | #8 | #1 |
| core_edgar rank | #30 | #1 |
| full rank | #30 | #1 |

**Root cause of V736's avg MAE loss**: The EDGAR feature integration causes a +4.1% degradation on `core_edgar`/`full` ablations for `funding_raised_usd`. Without EDGAR (`core_only`/`core_text`), V736 trails Chronos by only +0.2%. V736's ensemble assigns weight to EDGAR-derived features that overfit on funding amounts, while Chronos (zero-shot foundation model) ignores exogenous features entirely.

---

## Deep Re-Audit (2026-03-10) — 4 NEW CRITICAL FINDINGS

### NEW Finding F: Text Ablation Completely Broken 🔴

**Severity**: CRITICAL — invalidates entire ablation study

**Root Cause**: `offers_text.parquet` contains 19 columns that are ALL `string` dtype (headline, title, description_text, company_description, etc.). The `_prepare_features()` function at L572 of `run_block3_benchmark_shard.py` uses:
```python
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
```
This **silently drops ALL text string columns**. The join works correctly (`join_core_with_text()` adds 15 text columns), but they are removed before reaching any model.

**Impact**:
- `core_text ≡ core_only` for **ALL** models (0 numeric difference)
- `full ≡ core_edgar` for **ALL** models (in terms of feature matrix)
- 50% of benchmark compute was wasted on redundant ablations (core_text and full)
- Paper ablation study showing "text contribution" is completely invalid

**Fix Required**: Text→embedding pipeline before `_prepare_features()`:
- Option A: `sentence-transformers/all-MiniLM-L6-v2` → 384-dim embeddings per text column
- Option B: TF-IDF (top-1000) + PCA (→50-dim) — lightweight version

### NEW Finding G: 3 TSLib Models = 100% Constant Predictions 🔴

| Model | Conditions | fairness_pass=False | Status |
|-------|-----------|---------------------|--------|
| MICN | 15/15 | 100% | Must exclude |
| MultiPatchFormer | 31/31 | 100% | Must exclude |
| TimeFilter | 31/31 | 100% | Must exclude |

These models produce identical predictions for ALL test samples, making their MAE meaningless.

### NEW Finding H: NF Model Training Non-Determinism 🟡

8 NF models (BiTCN, DLinear, KAN, NLinear, RMoK, SOFTS, TimeMixer, TSMixerx) show ~1388 MAE units difference between `core_edgar` and `full` ablations, despite having IDENTICAL feature matrices (text columns dropped). Verified:
- Text join keys are unique (0 duplicates in 5.77M text rows)
- All 8 models produce DIFFERENT predictions (8 distinct MAE values)
- Difference is from GPU non-deterministic training in separate SLURM jobs

**Recommendation**: Run 3 random seeds and report mean ± std for NF models.

### NEW Finding I: Foundation Model Ablation Contaminated by RobustFallback 🟡

Foundation models (Chronos, ChronosBolt, Moirai, etc.) show MAE differences across ablations despite being univariate zero-shot models. Root cause: the `_RobustFallback` LightGBM handles entities with <10 training observations. This fallback model IS feature-dependent:
- `core_only`: LightGBM trained on core numeric features only
- `core_edgar`: LightGBM trained on core + EDGAR features → better fallback → lower MAE

Example: Chronos funding_raised_usd drops -1.61% from core_only to core_edgar — this is the fallback effect, NOT Chronos itself.

**Impact**: Foundation model ablation results reflect the RobustFallback sensitivity, not the foundation model's sensitivity to features.

---

## V736 Ranking Analysis (2026-03-10) — Definitive

### Multiple Ranking Standards Compared

| Standard | V736 Rank | Best at this standard |
|----------|-----------|----------------------|
| Raw MAE (avg across all conditions) | **#27/71** | Chronos (159,732) |
| Normalized Mean Rank (per-condition rank avg) | **#10/71** | NHITS (11.47) |
| investors_count (per-target MAE) | **#8/71** | NBEATS (44.78) |
| is_funded (per-target MAE) | **#8/71** | PatchTST (0.0327) |
| funding_raised_usd (per-target MAE) | **#27/71** | Chronos (377,504) |

### V736 vs Top Models (Head-to-Head)

| Rival | V736 Wins | Win Rate | Avg Gap |
|-------|-----------|----------|---------|
| Chronos | 153/264 | **58.0%** | +1.18% |
| TFT | 157/264 | **59.5%** | -0.08% |
| KAN | 127/264 | 48.1% | +0.56% |
| TimesNet | 119/264 | 45.1% | +0.57% |
| ChronosBolt | 73/264 | 27.7% | +1.16% |
| NBEATS | 73/264 | 27.7% | +1.35% |
| PatchTST | 68/264 | 25.8% | +1.42% |
| NHITS | 56/264 | 21.2% | +1.47% |

**Chronos granularity**: V736 beats Chronos on investors_count (124/124, 100%) and is_funded (16/16, 100%), but loses on funding_raised_usd (13/124, 10.5%).

### NeurIPS 2026 Standard Recommendation

The paper should use **Normalized Mean Rank** (standard in TSLib/Monash/M4 benchmarks) as the primary metric, with per-target MAE tables and condition-level win-rates as secondary evidence. Raw MAE average is misleading for multi-scale targets.

V736 at Normalized Rank #10 is **competitive but not oral-level** (needs #1-3). Key improvement area: funding_raised_usd target (+2.34% gap to Chronos).

_Last updated: 2026-03-10_