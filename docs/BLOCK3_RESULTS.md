# Block 3 Benchmark Results

**Generated**: 2026-03-13 12:37:14
**Benchmark Dir**: `block3_phase9_fair`
**Total Records (post-filter)**: 6668

## Overview

| Metric | Value |
|--------|-------|
| Raw records | 8660 |
| Filtered records | 6668 |
| Comparability filter | fairness_only=True, min_coverage=0.98 |
| Models evaluated | 69 |
| Categories | deep_classical, foundation, irregular, ml_tabular, statistical, transformer_sota, tslib_sota |
| Tasks | task1_outcome, task2_forecast, task3_risk_adjust |
| Total evaluations | 6668 |
| Real results | 6617 |
| Fallback (mean) | 51 |

## task1_outcome

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
| Timer | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Chronos2 | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TTM | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM | 447.4K | 447.4K | 447.4K | 447.4K | ✅ |

### irregular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| BRITS | 811.8K | 811.8K | 811.8K | 811.8K | ✅ |
| GRU-D | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| SAITS | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| CSDI | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |

### ml_tabular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NegativeBinomialGLM | 0.43 | 0.43 | 0.43 | 0.43 | ⚠️ fallback |
| LightGBMTweedie | 394.1K | 394.1K | 394.1K | 394.0K | ✅ |
| RandomForest | 414.3K | 414.3K | 414.3K | 414.3K | ✅ |
| XGBoostPoisson | 424.1K | 424.1K | 424.1K | 424.1K | ✅ |
| LightGBM | 455.6K | 455.6K | 455.6K | 455.6K | ✅ |
| XGBoost | 483.2K | 483.2K | 483.2K | 483.2K | ✅ |
| ExtraTrees | 487.4K | 487.4K | 487.4K | 487.4K | ✅ |
| HistGradientBoosting | 495.2K | 495.2K | 495.2K | 495.2K | ✅ |
| CatBoost | 567.9K | 567.9K | 567.9K | 567.9K | ✅ |
| MeanPredictor | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| SeasonalNaive | 31.03M | 31.03M | 31.03M | 31.03M | ✅ |

### statistical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| HistoricAverage | 1.51M | 1.51M | 1.51M | 1.51M | ✅ |
| CrostonSBA | 3.74M | 3.74M | 3.74M | 3.74M | ✅ |
| CrostonClassic | 3.94M | 3.94M | 3.94M | 3.94M | ✅ |
| SF_SeasonalNaive | 4.06M | 4.08M | 4.08M | 4.08M | ✅ |
| AutoETS | 4.08M | 4.09M | 4.09M | 4.10M | ✅ |
| WindowAverage | 4.08M | 4.08M | 4.08M | 4.08M | ✅ |
| CrostonOptimized | 4.09M | 4.09M | 4.09M | 4.09M | ✅ |
| HoltWinters | 4.13M | 4.31M | 4.42M | 4.68M | ✅ |
| Naive | 4.13M | 4.13M | 4.13M | 4.13M | ✅ |
| DynamicOptimizedTheta | 4.13M | 4.15M | 4.17M | 4.22M | ✅ |
| AutoARIMA | 4.13M | 4.15M | 4.18M | 4.24M | ✅ |
| MSTL | 4.14M | 4.23M | 4.13M | 4.13M | ✅ |
| Holt | 4.16M | 4.26M | 4.37M | 4.63M | ✅ |
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

### tslib_sota

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| Crossformer | 674.4K | 679.1K | 695.5K | 678.4K | ✅ |
| Reformer | 777.2K | 681.9K | 1.14M | 685.2K | ⚠️ fallback |
| MambaSimple | 800.2K | 990.1K | 1.48M | 1.09M | ✅ |
| MSGNet | 989.2K | 712.6K | 926.7K | 1.05M | ✅ |
| PAttn | 1.01M | 1.03M | 1.02M | 1.00M | ✅ |
| LightTS | 1.40M | 2.02M | 745.8K | 2.72M | ⚠️ fallback |

## task2_forecast

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
| Timer | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Chronos2 | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TTM | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM | 447.4K | 447.4K | 447.4K | 447.4K | ✅ |

### irregular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| BRITS | 811.8K | 811.8K | 811.8K | 811.8K | ✅ |
| GRU-D | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| SAITS | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| CSDI | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |

### ml_tabular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| LightGBMTweedie | 394.1K | 394.1K | 394.0K | 394.0K | ✅ |
| RandomForest | 414.3K | 414.3K | 414.3K | 414.3K | ✅ |
| XGBoostPoisson | 424.1K | 424.1K | 424.1K | 424.1K | ✅ |
| LightGBM | 455.6K | 455.6K | 455.7K | 455.6K | ✅ |
| XGBoost | 483.2K | 483.2K | 483.2K | 483.2K | ✅ |
| ExtraTrees | 487.4K | 487.4K | 487.4K | 487.4K | ✅ |
| HistGradientBoosting | 495.2K | 495.2K | 495.2K | 495.2K | ✅ |
| CatBoost | 567.9K | 567.9K | 567.9K | 567.9K | ✅ |
| MeanPredictor | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| SeasonalNaive | 31.03M | 31.03M | 31.03M | 31.03M | ✅ |

### statistical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| HistoricAverage | 1.51M | 1.51M | 1.51M | 1.51M | ✅ |
| CrostonSBA | 3.74M | 3.74M | 3.74M | 3.74M | ✅ |
| CrostonClassic | 3.94M | 3.94M | 3.94M | 3.94M | ✅ |
| SF_SeasonalNaive | 4.06M | 4.08M | 4.08M | 4.08M | ✅ |
| AutoETS | 4.08M | 4.09M | 4.09M | 4.10M | ✅ |
| WindowAverage | 4.08M | 4.08M | 4.08M | 4.08M | ✅ |
| CrostonOptimized | 4.09M | 4.09M | 4.09M | 4.09M | ✅ |
| HoltWinters | 4.13M | 4.31M | 4.42M | 4.68M | ✅ |
| Naive | 4.13M | 4.13M | 4.13M | 4.13M | ✅ |
| DynamicOptimizedTheta | 4.13M | 4.15M | 4.17M | 4.22M | ✅ |
| AutoARIMA | 4.13M | 4.15M | 4.18M | 4.24M | ✅ |
| MSTL | 4.14M | 4.23M | 4.13M | 4.13M | ✅ |
| Holt | 4.16M | 4.26M | 4.37M | 4.63M | ✅ |
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

### tslib_sota

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| Crossformer | 674.4K | 679.1K | 695.5K | 678.4K | ✅ |
| MambaSimple | 800.2K | 990.1K | 1.48M | 1.09M | ✅ |
| MSGNet | 989.2K | 712.6K | 926.7K | 1.05M | ✅ |
| PAttn | 1.01M | 1.03M | 1.02M | 1.00M | ✅ |

## task3_risk_adjust

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
| Timer | 383.6K | 383.6K | 383.6K | 383.6K | ✅ |
| Chronos2 | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TTM | 393.3K | 393.3K | 393.3K | 393.3K | ✅ |
| TimesFM | 447.4K | 447.4K | 447.4K | 447.4K | ✅ |

### irregular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| BRITS | 811.8K | 811.8K | 811.8K | 811.8K | ✅ |
| GRU-D | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| SAITS | 813.7K | 813.7K | 813.7K | 813.7K | ✅ |
| CSDI | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |

### ml_tabular

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| LightGBMTweedie | 394.1K | 394.1K | 394.1K | 394.1K | ✅ |
| RandomForest | 414.3K | 414.3K | 414.3K | 414.3K | ✅ |
| XGBoostPoisson | 424.1K | 424.1K | 424.1K | 424.1K | ✅ |
| LightGBM | 455.6K | 455.6K | 455.6K | 455.6K | ✅ |
| XGBoost | 483.2K | 483.2K | 483.2K | 483.2K | ✅ |
| ExtraTrees | 487.4K | 487.4K | 487.4K | 487.4K | ✅ |
| HistGradientBoosting | 495.2K | 495.2K | 495.2K | 495.2K | ✅ |
| CatBoost | 567.9K | 567.9K | 567.9K | 567.9K | ✅ |
| MeanPredictor | 2.13M | 2.13M | 2.13M | 2.13M | ✅ |
| SeasonalNaive | 31.03M | 31.03M | 31.03M | 31.03M | ✅ |

### statistical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| HistoricAverage | 1.51M | 1.51M | 1.51M | 1.51M | ✅ |
| CrostonSBA | 3.74M | 3.74M | 3.74M | 3.74M | ✅ |
| CrostonClassic | 3.94M | 3.94M | 3.94M | 3.94M | ✅ |
| SF_SeasonalNaive | 4.06M | 4.08M | 4.08M | 4.08M | ✅ |
| AutoETS | 4.08M | 4.09M | 4.09M | 4.10M | ✅ |
| WindowAverage | 4.08M | 4.08M | 4.08M | 4.08M | ✅ |
| CrostonOptimized | 4.09M | 4.09M | 4.09M | 4.09M | ✅ |
| HoltWinters | 4.13M | 4.31M | 4.42M | 4.68M | ✅ |
| Naive | 4.13M | 4.13M | 4.13M | 4.13M | ✅ |
| DynamicOptimizedTheta | 4.13M | 4.15M | 4.17M | 4.22M | ✅ |
| AutoARIMA | 4.13M | 4.15M | 4.18M | 4.24M | ✅ |
| MSTL | 4.14M | 4.23M | 4.13M | 4.13M | ✅ |
| Holt | 4.16M | 4.26M | 4.37M | 4.63M | ✅ |
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

### tslib_sota

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| Crossformer | 674.4K | 679.1K | 695.5K | 678.4K | ✅ |
| Reformer | 777.2K | 681.9K | 1.14M | 685.2K | ⚠️ fallback |
| MambaSimple | 800.2K | 990.1K | 1.48M | 1.09M | ✅ |
| MSGNet | 989.2K | 712.6K | 926.7K | 1.05M | ✅ |
| PAttn | 1.01M | 1.03M | 1.02M | 1.00M | ✅ |
| LightTS | 1.40M | 2.02M | 745.8K | 2.72M | ⚠️ fallback |

## Training Time Summary

| Model | Avg (s) | Min (s) | Max (s) |
|-------|--------:|--------:|--------:|
| MambaSimple | 11865.8 | 865.8 | 49536.0 |
| MSGNet | 11516.7 | 1130.1 | 30456.4 |
| Reformer | 5066.1 | 926.5 | 17614.9 |
| Crossformer | 2584.6 | 277.7 | 4364.2 |
| PAttn | 1976.9 | 330.2 | 4472.5 |
| NegativeBinomialGLM | 1457.4 | 1029.1 | 1947.6 |
| iTransformer | 1454.7 | 927.9 | 1763.2 |
| BRITS | 1399.2 | 349.2 | 2770.7 |
| MSTL | 1176.3 | 112.5 | 1982.0 |
| LightTS | 1020.2 | 137.8 | 3222.8 |
| AutoARIMA | 692.1 | 284.4 | 1179.2 |
| RandomForest | 642.1 | 224.6 | 1563.0 |
| SOFTS | 459.7 | 329.9 | 708.1 |
| ExtraTrees | 439.4 | 71.5 | 1186.1 |
| GRU-D | 354.0 | 179.6 | 525.6 |
| FEDformer | 344.4 | 171.1 | 585.0 |
| RMoK | 278.3 | 165.1 | 426.0 |
| HistGradientBoosting | 275.2 | 6.8 | 1640.3 |
| Autoformer | 273.9 | 152.3 | 465.4 |
| Informer | 255.7 | 113.5 | 368.5 |
| TimesNet | 244.5 | 125.8 | 463.3 |
| TimeMixer | 175.7 | 140.6 | 232.6 |
| TSMixer | 158.0 | 107.6 | 262.7 |
| TSMixerx | 144.3 | 113.0 | 223.4 |
| HoltWinters | 143.6 | 41.8 | 278.8 |
| XGBoostPoisson | 120.8 | 30.8 | 305.6 |
| XGBoost | 119.5 | 24.5 | 733.0 |
| AutoTheta | 116.5 | 60.1 | 201.1 |
| TFT | 91.6 | 71.1 | 129.2 |
| AutoETS | 80.8 | 10.2 | 157.5 |
| VanillaTransformer | 79.6 | 50.2 | 217.3 |
| PatchTST | 72.9 | 35.7 | 141.3 |
| SAITS | 62.4 | 23.6 | 106.0 |
| DeepAR | 61.8 | 30.6 | 171.3 |
| CSDI | 56.4 | 14.7 | 149.8 |
| Holt | 54.5 | 8.2 | 125.3 |
| DilatedRNN | 47.7 | 24.5 | 82.9 |
| LSTM | 39.7 | 16.2 | 76.7 |
| TCN | 38.9 | 12.1 | 76.4 |
| TiDE | 36.9 | 20.3 | 104.3 |
| DynamicOptimizedTheta | 35.8 | 16.8 | 68.6 |
| GRU | 35.3 | 11.8 | 79.9 |
| MLP | 33.8 | 9.2 | 70.3 |
| BiTCN | 31.8 | 21.8 | 79.7 |
| KAN | 28.0 | 10.7 | 69.5 |
| Chronos2 | 27.0 | 7.9 | 46.3 |
| LightGBMTweedie | 26.4 | 4.7 | 415.7 |
| LightGBM | 26.3 | 4.7 | 562.0 |
| NBEATSx | 25.4 | 9.3 | 73.8 |
| CatBoost | 25.2 | 2.2 | 99.3 |
| NHITS | 24.5 | 10.5 | 37.6 |
| TTM | 24.4 | 7.2 | 45.4 |
| NBEATS | 24.1 | 9.3 | 70.5 |
| NLinear | 22.2 | 11.8 | 62.2 |
| DLinear | 21.5 | 7.8 | 63.9 |
| DeepNPTS | 21.3 | 14.1 | 36.0 |
| Timer | 8.3 | 2.9 | 50.9 |
| Chronos | 7.9 | 2.7 | 30.9 |
| ChronosBolt | 7.2 | 2.6 | 12.5 |
| TimesFM | 7.1 | 2.0 | 46.7 |
| CrostonOptimized | 4.0 | 1.9 | 6.5 |
| SF_SeasonalNaive | 1.3 | 0.4 | 2.2 |
| CrostonClassic | 1.2 | 0.4 | 2.1 |
| CrostonSBA | 1.2 | 0.4 | 2.1 |
| Naive | 1.0 | 0.3 | 2.0 |
| HistoricAverage | 1.0 | 0.3 | 1.8 |
| WindowAverage | 0.8 | 0.2 | 1.5 |
| MeanPredictor | 0.0 | 0.0 | 0.0 |
| SeasonalNaive | 0.0 | 0.0 | 0.0 |

## Completion Matrix

Shows which task/category/ablation combinations have results.

| Task | Category | Ablation | Models |
|------|----------|----------|-------:|
| task1_outcome | deep_classical | core_edgar | 9 |
| task1_outcome | deep_classical | core_edgar_seed2 | 9 |
| task1_outcome | deep_classical | core_only | 9 |
| task1_outcome | deep_classical | core_only_seed2 | 9 |
| task1_outcome | foundation | core_edgar | 6 |
| task1_outcome | foundation | core_edgar_seed2 | 4 |
| task1_outcome | foundation | core_only | 6 |
| task1_outcome | foundation | core_only_seed2 | 6 |
| task1_outcome | irregular | core_edgar | 4 |
| task1_outcome | irregular | core_edgar_seed2 | 4 |
| task1_outcome | irregular | core_only | 4 |
| task1_outcome | irregular | core_only_seed2 | 4 |
| task1_outcome | ml_tabular | core_edgar | 11 |
| task1_outcome | ml_tabular | core_edgar_seed2 | 11 |
| task1_outcome | ml_tabular | core_only | 11 |
| task1_outcome | ml_tabular | core_only_seed2 | 11 |
| task1_outcome | statistical | core_edgar | 14 |
| task1_outcome | statistical | core_edgar_seed2 | 14 |
| task1_outcome | statistical | core_only | 14 |
| task1_outcome | statistical | core_only_seed2 | 14 |
| task1_outcome | transformer_sota | core_edgar | 19 |
| task1_outcome | transformer_sota | core_edgar_seed2 | 19 |
| task1_outcome | transformer_sota | core_only | 19 |
| task1_outcome | transformer_sota | core_only_seed2 | 19 |
| task1_outcome | tslib_sota | core_edgar | 6 |
| task1_outcome | tslib_sota | core_edgar_seed2 | 4 |
| task1_outcome | tslib_sota | core_only | 6 |
| task1_outcome | tslib_sota | core_only_seed2 | 6 |
| task2_forecast | deep_classical | core_edgar | 9 |
| task2_forecast | deep_classical | core_edgar_seed2 | 9 |
| task2_forecast | deep_classical | core_only | 9 |
| task2_forecast | deep_classical | core_only_seed2 | 9 |
| task2_forecast | foundation | core_edgar | 6 |
| task2_forecast | foundation | core_edgar_seed2 | 4 |
| task2_forecast | foundation | core_only | 6 |
| task2_forecast | foundation | core_only_seed2 | 6 |
| task2_forecast | irregular | core_edgar | 4 |
| task2_forecast | irregular | core_edgar_seed2 | 4 |
| task2_forecast | irregular | core_only | 4 |
| task2_forecast | irregular | core_only_seed2 | 4 |
| task2_forecast | ml_tabular | core_edgar | 10 |
| task2_forecast | ml_tabular | core_edgar_seed2 | 10 |
| task2_forecast | ml_tabular | core_only | 10 |
| task2_forecast | ml_tabular | core_only_seed2 | 10 |
| task2_forecast | statistical | core_edgar | 14 |
| task2_forecast | statistical | core_edgar_seed2 | 14 |
| task2_forecast | statistical | core_only | 14 |
| task2_forecast | statistical | core_only_seed2 | 14 |
| task2_forecast | transformer_sota | core_edgar | 19 |
| task2_forecast | transformer_sota | core_edgar_seed2 | 19 |
| task2_forecast | transformer_sota | core_only | 19 |
| task2_forecast | transformer_sota | core_only_seed2 | 19 |
| task2_forecast | tslib_sota | core_edgar | 4 |
| task2_forecast | tslib_sota | core_edgar_seed2 | 4 |
| task2_forecast | tslib_sota | core_only | 4 |
| task2_forecast | tslib_sota | core_only_seed2 | 4 |
| task3_risk_adjust | deep_classical | core_edgar | 9 |
| task3_risk_adjust | deep_classical | core_edgar_seed2 | 9 |
| task3_risk_adjust | deep_classical | core_only | 9 |
| task3_risk_adjust | foundation | core_edgar | 6 |
| task3_risk_adjust | foundation | core_edgar_seed2 | 4 |
| task3_risk_adjust | foundation | core_only | 6 |
| task3_risk_adjust | irregular | core_edgar | 4 |
| task3_risk_adjust | irregular | core_edgar_seed2 | 4 |
| task3_risk_adjust | irregular | core_only | 4 |
| task3_risk_adjust | ml_tabular | core_edgar | 10 |
| task3_risk_adjust | ml_tabular | core_edgar_seed2 | 10 |
| task3_risk_adjust | ml_tabular | core_only | 10 |
| task3_risk_adjust | statistical | core_edgar | 14 |
| task3_risk_adjust | statistical | core_edgar_seed2 | 14 |
| task3_risk_adjust | statistical | core_only | 14 |
| task3_risk_adjust | transformer_sota | core_edgar | 19 |
| task3_risk_adjust | transformer_sota | core_edgar_seed2 | 19 |
| task3_risk_adjust | transformer_sota | core_only | 19 |
| task3_risk_adjust | tslib_sota | core_edgar | 6 |
| task3_risk_adjust | tslib_sota | core_edgar_seed2 | 4 |
| task3_risk_adjust | tslib_sota | core_only | 6 |

---
_Last updated: 2026-03-13 12:37:14_