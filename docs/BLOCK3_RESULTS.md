# Block 3 Benchmark Results

**Generated**: 2026-03-28 16:56:06
**Benchmark Dir**: `block3_phase9_fair`
**Total Records (post-filter)**: 12272

## Overview

> Note: the post-filter CSV still contains **21 retired AutoFit legacy lines** (`AutoFitV1, AutoFitV2, AutoFitV2E, AutoFitV3, AutoFitV3E` ...). For current clean benchmark interpretation, treat `AutoFitV739` as the only valid AutoFit baseline and use the `non-retired models` count below when comparing against the active frontier.

| Metric | Value |
|--------|-------|
| Raw records | 16122 |
| Filtered records | 12272 |
| Comparability filter | fairness_only=True, min_coverage=0.98 |
| Models evaluated | 107 |
| Non-retired models evaluated | 86 |
| Retired AutoFit legacy lines still present | 21 |
| Categories | autofit, deep_classical, foundation, irregular, ml_tabular, statistical, transformer_sota, tslib_sota |
| Tasks | task1_outcome, task2_forecast, task3_risk_adjust |
| Total evaluations | 12272 |
| Real results | 12195 |
| Fallback (mean) | 77 |

## task1_outcome

### autofit

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| AutoFitV738 | 380.4K | 380.7K | 381.3K | 384.2K | ✅ |
| AutoFitV737 | 380.4K | 380.5K | 381.4K | 383.5K | ✅ |
| AutoFitV739 | 380.8K | 380.9K | 381.0K | 380.3K | ✅ |
| AutoFitV732 | 384.2K | 384.2K | 384.5K | 384.7K | ✅ |
| AutoFitV736 | 384.2K | 384.4K | 385.2K | 387.3K | ✅ |
| NFAdaptiveChampion | 385.0K | 384.6K | 385.5K | 386.9K | ✅ |
| AutoFitV734 | 386.7K | 384.9K | 384.0K | 386.1K | ✅ |
| AutoFitV735 | 388.1K | 385.5K | 384.2K | 386.9K | ✅ |
| AutoFitV733 | 388.1K | 385.5K | 385.1K | 386.9K | ✅ |
| AutoFitV1 | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV6 | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV3Max | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV3 | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV73 | 396.1K | 396.1K | 396.1K | 396.1K | ✅ |
| AutoFitV731 | 396.1K | 396.1K | 396.1K | 396.1K | ✅ |
| AutoFitV4 | 426.8K | 426.8K | 426.8K | 426.8K | ✅ |
| AutoFitV3E | 429.6K | 429.6K | 429.6K | 429.6K | ✅ |
| AutoFitV2 | 430.4K | 430.4K | 430.4K | 430.4K | ✅ |
| AutoFitV2E | 430.4K | 430.4K | 430.4K | 430.4K | ✅ |
| AutoFitV5 | 475.6K | 475.6K | 475.6K | 475.6K | ✅ |
| AutoFitV71 | 482.9K | 483.5K | 483.5K | 483.5K | ✅ |
| AutoFitV72 | 482.9K | 483.5K | 482.9K | 482.9K | ✅ |
| AutoFitV7 | 488.4K | 488.4K | 488.4K | 488.4K | ✅ |
| FusedChampion | 3.76M | 3.83M | 384.5K | 384.7K | ✅ |

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
| LightGBMTweedie | 394.1K | 394.1K | 394.1K | 394.0K | ✅ |
| RandomForest | 414.3K | 414.3K | 414.3K | 414.3K | ✅ |
| XGBoostPoisson | 424.1K | 424.1K | 424.1K | 424.1K | ✅ |
| LightGBM | 455.6K | 455.6K | 455.6K | 455.6K | ✅ |
| XGBoost | 483.2K | 483.2K | 483.2K | 483.2K | ✅ |
| ExtraTrees | 487.4K | 487.4K | 487.4K | 487.4K | ✅ |
| HistGradientBoosting | 495.2K | 495.2K | 495.2K | 495.2K | ✅ |
| CatBoost | 567.9K | 567.9K | 567.9K | 567.9K | ⚠️ fallback |
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
| Crossformer | 674.4K | 679.1K | 695.5K | 678.4K | ⚠️ fallback |
| FreTS | 734.2K | 3.52M | 4.41M | 4.46M | ⚠️ fallback |
| Reformer | 777.2K | 681.9K | 1.14M | 685.2K | ⚠️ fallback |
| MambaSimple | 800.2K | 990.1K | 1.48M | 1.09M | ⚠️ fallback |
| MSGNet | 989.2K | 712.6K | 926.7K | 1.05M | ✅ |
| xPatch | 1.00M | 695.9K | 1.15M | 3.09M | ✅ |
| PAttn | 1.01M | 1.03M | 1.02M | 1.00M | ✅ |
| TimeRecipe | 1.01M | 1.02M | 1.01M | 680.0K | ✅ |
| SRSNet | 1.02M | 1.01M | 1.02M | 1.01M | ✅ |
| PIR | 1.02M | 1.02M | 1.02M | 1.02M | ✅ |
| CARD | 1.02M | 1.02M | 1.02M | 1.02M | ✅ |
| PDF | 1.02M | 1.02M | 1.04M | 1.05M | ✅ |
| DUET | 1.03M | 1.03M | 1.03M | 1.02M | ✅ |
| ModernTCN | 1.04M | 1.04M | 1.06M | 1.14M | ✅ |
| SCINet | 1.05M | 985.7K | 1.02M | 1.04M | ✅ |
| FiLM | 1.06M | 1.06M | 1.06M | 1.06M | ✅ |
| NonstationaryTransformer | 1.07M | 1.05M | 1.04M | 1.86M | ✅ |
| Fredformer | 1.08M | 1.07M | 986.3K | 1.05M | ✅ |
| LightTS | 1.40M | 2.02M | 745.8K | 2.72M | ⚠️ fallback |
| FilterTS | 5.32M | 2.60M | 4.13M | 8.89M | ✅ |
| SegRNN | 8.83M | 6.36M | 4.20M | 7.15M | ✅ |

## task2_forecast

### autofit

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| AutoFitV738 | 380.4K | 380.7K | 381.3K | 384.2K | ✅ |
| AutoFitV737 | 380.4K | 380.5K | 381.4K | 383.5K | ✅ |
| AutoFitV739 | 380.8K | 380.9K | 381.0K | 380.3K | ✅ |
| AutoFitV732 | 384.2K | 384.2K | 384.5K | 384.7K | ✅ |
| AutoFitV736 | 384.2K | 384.4K | 385.2K | 387.3K | ✅ |
| NFAdaptiveChampion | 385.0K | 384.6K | 385.5K | 386.9K | ✅ |
| AutoFitV734 | 386.7K | 384.9K | 384.0K | 386.1K | ✅ |
| AutoFitV735 | 388.1K | 385.5K | 384.2K | 386.9K | ✅ |
| AutoFitV733 | 388.1K | 385.5K | 385.1K | 386.9K | ✅ |
| AutoFitV1 | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV6 | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV3Max | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV3 | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV73 | 396.1K | 396.1K | 396.1K | 396.1K | ✅ |
| AutoFitV731 | 396.1K | 396.1K | 396.1K | 396.1K | ✅ |
| AutoFitV4 | 426.8K | 426.8K | 426.8K | 426.8K | ✅ |
| AutoFitV3E | 430.0K | 429.6K | 429.6K | 429.6K | ✅ |
| AutoFitV2 | 430.4K | 430.4K | 430.4K | 430.4K | ✅ |
| AutoFitV2E | 430.4K | 430.4K | 430.4K | 430.4K | ✅ |
| AutoFitV5 | 475.6K | 491.0K | 475.6K | 475.6K | ✅ |
| AutoFitV71 | 482.9K | 483.5K | 483.5K | 482.9K | ✅ |
| AutoFitV72 | 483.0K | 482.9K | 482.9K | 483.5K | ✅ |
| AutoFitV7 | 488.4K | 488.4K | 488.4K | 488.4K | ✅ |
| FusedChampion | 3.76M | 3.83M | 384.5K | 384.7K | ✅ |

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
| CatBoost | 567.9K | 567.9K | 567.9K | 567.9K | ⚠️ fallback |
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
| Crossformer | 674.4K | 679.1K | 695.5K | 678.4K | ⚠️ fallback |
| xPatch | 678.9K | 700.8K | 1.30M | 1.73M | ✅ |
| Reformer | 777.2K | 681.9K | 1.14M | 685.2K | ⚠️ fallback |
| MambaSimple | 800.2K | 990.1K | 1.48M | 1.09M | ✅ |
| FreTS | 805.7K | 2.32M | 742.1K | 918.1K | ✅ |
| MSGNet | 989.2K | 712.6K | 926.7K | 1.05M | ✅ |
| SRSNet | 994.8K | 1.01M | 1.01M | 1.01M | ✅ |
| PAttn | 1.01M | 1.03M | 1.02M | 1.00M | ✅ |
| PIR | 1.02M | 1.02M | 1.02M | 1.02M | ✅ |
| CARD | 1.02M | 1.02M | 1.02M | 1.02M | ✅ |
| DUET | 1.03M | 1.03M | 1.03M | 1.01M | ✅ |
| ModernTCN | 1.04M | 1.04M | 968.4K | 700.3K | ✅ |
| PDF | 1.04M | 1.03M | 1.03M | 1.10M | ✅ |
| FiLM | 1.06M | 1.06M | 1.06M | 1.06M | ✅ |
| SCINet | 1.08M | 1.08M | 1.04M | 1.11M | ✅ |
| Fredformer | 1.08M | 1.05M | 1.07M | 1.09M | ✅ |
| NonstationaryTransformer | 1.09M | 1.04M | 1.04M | 1.04M | ✅ |
| FilterTS | 1.40M | 1.23M | 5.24M | 9.23M | ✅ |
| LightTS | 1.40M | 2.02M | 745.8K | 2.72M | ✅ |
| SegRNN | 4.11M | 2.99M | 8.42M | 6.38M | ✅ |
| TimeRecipe | 13.33M | 1.02M | 10.88M | 1.01M | ✅ |

## task3_risk_adjust

### autofit

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| AutoFitV738 | 380.4K | 380.7K | 381.3K | 384.2K | ✅ |
| AutoFitV737 | 380.4K | 380.5K | 381.4K | 383.5K | ✅ |
| AutoFitV739 | 380.8K | 380.9K | 381.0K | 380.3K | ✅ |
| AutoFitV732 | 384.2K | 384.2K | 384.5K | 384.7K | ✅ |
| AutoFitV736 | 384.2K | 384.4K | 385.2K | 387.3K | ✅ |
| NFAdaptiveChampion | 385.0K | 384.6K | 385.5K | 386.9K | ✅ |
| AutoFitV734 | 386.7K | 384.9K | 384.0K | 386.1K | ✅ |
| AutoFitV735 | 388.1K | 385.5K | 384.2K | 386.9K | ✅ |
| AutoFitV733 | 388.1K | 385.5K | 385.1K | 386.9K | ✅ |
| AutoFitV1 | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV6 | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV3Max | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV3 | 393.2K | 393.2K | 393.2K | 393.2K | ✅ |
| AutoFitV73 | 396.1K | 396.1K | 396.1K | 396.1K | ✅ |
| AutoFitV731 | 396.1K | 396.1K | 396.1K | 396.1K | ✅ |
| AutoFitV4 | 426.8K | 426.8K | 426.8K | 426.8K | ✅ |
| AutoFitV3E | 429.6K | 429.6K | 429.6K | 429.6K | ✅ |
| AutoFitV2 | 430.4K | 430.4K | 430.4K | 430.4K | ✅ |
| AutoFitV2E | 430.4K | 430.4K | 430.4K | 430.4K | ✅ |
| AutoFitV5 | 475.6K | 475.6K | 475.6K | 475.6K | ✅ |
| AutoFitV71 | 482.9K | 482.9K | 483.0K | 482.9K | ✅ |
| AutoFitV72 | 482.9K | 482.9K | 483.0K | 482.9K | ✅ |
| AutoFitV7 | 488.4K | 488.4K | 488.4K | 488.4K | ✅ |
| FusedChampion | 3.76M | 3.83M | 384.5K | 384.7K | ✅ |

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
| CatBoost | 567.9K | 567.9K | 567.9K | 567.9K | ⚠️ fallback |
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
| xPatch | 678.9K | 700.8K | 1.30M | 5.78M | ✅ |
| Reformer | 777.2K | 681.9K | 1.14M | 685.2K | ⚠️ fallback |
| MambaSimple | 800.2K | 990.1K | 1.48M | 1.09M | ✅ |
| FreTS | 805.7K | 2.32M | 742.1K | 1.05M | ✅ |
| MSGNet | 989.2K | 712.6K | 926.7K | 1.05M | ✅ |
| SRSNet | 994.8K | 1.01M | 1.01M | 1.01M | ✅ |
| PAttn | 1.01M | 1.03M | 1.02M | 1.00M | ✅ |
| PIR | 1.02M | 1.02M | 1.02M | 1.02M | ✅ |
| CARD | 1.02M | 1.02M | 1.02M | 1.02M | ✅ |
| DUET | 1.03M | 1.03M | 1.03M | 1.03M | ✅ |
| ModernTCN | 1.04M | 1.04M | 1.04M | 674.4K | ✅ |
| PDF | 1.04M | 1.03M | 1.03M | 1.07M | ✅ |
| FiLM | 1.06M | 1.06M | 1.06M | 1.06M | ✅ |
| SCINet | 1.08M | 1.08M | 1.04M | 1.03M | ✅ |
| Fredformer | 1.08M | 1.05M | 1.07M | 1.05M | ✅ |
| NonstationaryTransformer | 1.09M | 1.04M | 1.04M | 1.35M | ✅ |
| FilterTS | 1.40M | 1.23M | 5.24M | 8.36M | ✅ |
| LightTS | 1.40M | 2.02M | 745.8K | 2.72M | ⚠️ fallback |
| SegRNN | 4.11M | 2.99M | 8.42M | 2.42M | ✅ |
| TimeRecipe | 13.33M | 1.02M | 10.88M | 999.9K | ✅ |

## Training Time Summary

| Model | Avg (s) | Min (s) | Max (s) |
|-------|--------:|--------:|--------:|
| MSGNet | 14154.5 | 1121.9 | 105725.4 |
| MambaSimple | 12549.3 | 865.8 | 49536.0 |
| AutoFitV73 | 9873.9 | 8622.1 | 11439.7 |
| AutoFitV731 | 9801.8 | 8645.2 | 11468.4 |
| ModernTCN | 8008.6 | 788.0 | 31210.8 |
| CARD | 7096.1 | 570.7 | 23557.2 |
| FiLM | 6570.7 | 1584.7 | 22418.7 |
| Fredformer | 6263.6 | 695.6 | 18666.2 |
| AutoFitV2 | 5703.1 | 5289.7 | 6442.9 |
| AutoFitV3E | 5694.0 | 5297.2 | 6440.6 |
| AutoFitV2E | 5661.1 | 5312.1 | 6142.5 |
| AutoFitV4 | 5634.5 | 5287.4 | 6116.8 |
| AutoFitV1 | 5327.1 | 4909.0 | 5738.7 |
| AutoFitV3 | 5303.4 | 5035.0 | 5686.4 |
| AutoFitV3Max | 5265.3 | 5008.3 | 5585.4 |
| SCINet | 5231.4 | 274.2 | 25886.3 |
| AutoFitV71 | 4852.1 | 4100.9 | 6149.3 |
| AutoFitV72 | 4803.3 | 4169.5 | 5792.6 |
| Reformer | 4124.2 | 926.5 | 17614.9 |
| AutoFitV7 | 4062.8 | 3402.6 | 4780.2 |
| AutoFitV6 | 3940.7 | 3597.3 | 4253.1 |
| PIR | 3847.2 | 335.9 | 16062.2 |
| FreTS | 3792.5 | 307.8 | 13755.7 |
| Crossformer | 3135.9 | 277.7 | 9227.3 |
| PAttn | 2812.8 | 283.1 | 14803.5 |
| xPatch | 1956.5 | 207.0 | 11008.1 |
| AutoFitV5 | 1824.4 | 1396.0 | 2267.0 |
| BRITS | 1526.4 | 349.2 | 2770.7 |
| iTransformer | 1466.2 | 927.9 | 1806.3 |
| PDF | 1364.9 | 78.1 | 3909.5 |
| DUET | 1343.8 | 79.7 | 4508.6 |
| MSTL | 1157.4 | 112.0 | 1982.0 |
| NonstationaryTransformer | 1111.6 | 61.9 | 7899.8 |
| FilterTS | 955.0 | 163.7 | 3535.8 |
| LightTS | 866.9 | 137.8 | 3222.8 |
| SegRNN | 828.9 | 98.5 | 4057.0 |
| SRSNet | 785.8 | 136.4 | 2852.6 |
| RandomForest | 776.4 | 224.6 | 4432.2 |
| TimeRecipe | 669.4 | 31.0 | 2343.5 |
| AutoARIMA | 639.3 | 213.1 | 1179.2 |
| ExtraTrees | 613.7 | 71.5 | 3940.6 |
| SOFTS | 466.8 | 329.9 | 708.1 |
| AutoFitV739 | 385.6 | 144.1 | 837.6 |
| GRU-D | 362.7 | 169.4 | 552.8 |
| FEDformer | 351.0 | 171.1 | 585.0 |
| RMoK | 287.0 | 165.1 | 470.1 |
| Autoformer | 279.9 | 152.3 | 465.4 |
| Informer | 262.6 | 113.5 | 368.5 |
| NFAdaptiveChampion | 256.5 | 77.3 | 407.6 |
| TimesNet | 250.7 | 125.8 | 463.3 |
| HistGradientBoosting | 216.7 | 6.8 | 1640.3 |
| TimeMixer | 185.2 | 140.6 | 292.3 |
| AutoFitV734 | 175.2 | 72.4 | 379.9 |
| TSMixer | 165.9 | 107.6 | 270.0 |
| TSMixerx | 153.5 | 113.0 | 228.4 |
| HoltWinters | 144.3 | 41.8 | 278.8 |
| XGBoost | 137.7 | 24.5 | 733.0 |
| XGBoostPoisson | 135.5 | 30.8 | 336.4 |
| AutoFitV733 | 122.7 | 35.7 | 295.0 |
| AutoTheta | 112.1 | 50.0 | 201.1 |
| AutoFitV735 | 106.7 | 25.5 | 293.8 |
| TFT | 103.6 | 71.1 | 167.4 |
| VanillaTransformer | 86.5 | 50.2 | 217.3 |
| AutoETS | 80.3 | 8.6 | 157.5 |
| PatchTST | 79.5 | 35.7 | 141.3 |
| SAITS | 71.7 | 23.6 | 143.5 |
| DeepAR | 70.7 | 30.6 | 171.3 |
| CSDI | 66.7 | 14.7 | 166.9 |
| Holt | 54.9 | 8.2 | 125.3 |
| DilatedRNN | 51.1 | 24.5 | 85.7 |
| TiDE | 47.2 | 20.3 | 104.3 |
| TCN | 42.3 | 12.1 | 77.7 |
| LSTM | 42.2 | 16.2 | 76.7 |
| AutoFitV736 | 41.6 | 34.0 | 50.6 |
| AutoFitV737 | 40.9 | 33.1 | 51.2 |
| GRU | 40.6 | 11.8 | 80.1 |
| BiTCN | 39.1 | 21.8 | 83.5 |
| MLP | 37.2 | 9.2 | 71.6 |
| DynamicOptimizedTheta | 36.0 | 16.8 | 68.6 |
| KAN | 35.5 | 10.7 | 83.4 |
| NHITS | 33.0 | 10.5 | 79.4 |
| NBEATSx | 32.9 | 9.3 | 79.2 |
| NBEATS | 31.9 | 9.3 | 74.5 |
| CatBoost | 31.4 | 2.2 | 288.7 |
| FusedChampion | 30.9 | 7.8 | 59.8 |
| Chronos2 | 30.4 | 7.9 | 57.1 |
| DeepNPTS | 30.2 | 14.1 | 75.4 |
| NLinear | 29.9 | 11.8 | 73.6 |
| DLinear | 29.1 | 7.8 | 71.8 |
| TTM | 28.9 | 7.2 | 56.7 |
| LightGBMTweedie | 25.7 | 4.7 | 415.7 |
| LightGBM | 24.8 | 4.7 | 562.0 |
| AutoFitV738 | 19.5 | 14.7 | 25.1 |
| AutoFitV732 | 16.3 | 8.0 | 26.0 |
| Timer | 16.3 | 2.9 | 56.5 |
| Chronos | 16.1 | 2.7 | 55.8 |
| ChronosBolt | 15.4 | 2.6 | 55.3 |
| TimesFM | 15.2 | 2.0 | 55.0 |
| CrostonOptimized | 4.1 | 1.9 | 6.5 |
| CrostonClassic | 1.2 | 0.4 | 2.1 |
| CrostonSBA | 1.2 | 0.4 | 2.1 |
| SF_SeasonalNaive | 1.2 | 0.3 | 2.2 |
| Naive | 1.0 | 0.3 | 2.0 |
| HistoricAverage | 1.0 | 0.3 | 1.8 |
| WindowAverage | 0.8 | 0.2 | 1.5 |
| MeanPredictor | 0.0 | 0.0 | 0.0 |
| SeasonalNaive | 0.0 | 0.0 | 0.0 |

## Completion Matrix

Shows which task/category/ablation combinations have results.

| Task | Category | Ablation | Models |
|------|----------|----------|-------:|
| task1_outcome | autofit | core_edgar | 1 |
| task1_outcome | autofit | core_edgar_seed2 | 24 |
| task1_outcome | autofit | core_only | 1 |
| task1_outcome | autofit | core_only_seed2 | 24 |
| task1_outcome | autofit | core_text | 1 |
| task1_outcome | autofit | full | 1 |
| task1_outcome | deep_classical | core_edgar | 9 |
| task1_outcome | deep_classical | core_edgar_seed2 | 9 |
| task1_outcome | deep_classical | core_only | 9 |
| task1_outcome | deep_classical | core_only_seed2 | 9 |
| task1_outcome | deep_classical | core_text | 9 |
| task1_outcome | deep_classical | full | 9 |
| task1_outcome | foundation | core_edgar | 6 |
| task1_outcome | foundation | core_edgar_seed2 | 4 |
| task1_outcome | foundation | core_only | 6 |
| task1_outcome | foundation | core_only_seed2 | 6 |
| task1_outcome | foundation | core_text | 6 |
| task1_outcome | foundation | full | 6 |
| task1_outcome | irregular | core_edgar | 4 |
| task1_outcome | irregular | core_edgar_seed2 | 4 |
| task1_outcome | irregular | core_only | 4 |
| task1_outcome | irregular | core_only_seed2 | 4 |
| task1_outcome | irregular | core_text | 4 |
| task1_outcome | irregular | full | 4 |
| task1_outcome | ml_tabular | core_edgar | 10 |
| task1_outcome | ml_tabular | core_edgar_seed2 | 10 |
| task1_outcome | ml_tabular | core_only | 10 |
| task1_outcome | ml_tabular | core_only_seed2 | 10 |
| task1_outcome | ml_tabular | core_text | 9 |
| task1_outcome | ml_tabular | full | 9 |
| task1_outcome | statistical | core_edgar | 14 |
| task1_outcome | statistical | core_edgar_seed2 | 14 |
| task1_outcome | statistical | core_only | 14 |
| task1_outcome | statistical | core_only_seed2 | 14 |
| task1_outcome | statistical | core_text | 14 |
| task1_outcome | statistical | full | 14 |
| task1_outcome | transformer_sota | core_edgar | 19 |
| task1_outcome | transformer_sota | core_edgar_seed2 | 19 |
| task1_outcome | transformer_sota | core_only | 19 |
| task1_outcome | transformer_sota | core_only_seed2 | 19 |
| task1_outcome | transformer_sota | core_text | 19 |
| task1_outcome | transformer_sota | full | 19 |
| task1_outcome | tslib_sota | core_edgar | 21 |
| task1_outcome | tslib_sota | core_edgar_seed2 | 19 |
| task1_outcome | tslib_sota | core_only | 21 |
| task1_outcome | tslib_sota | core_only_seed2 | 21 |
| task1_outcome | tslib_sota | core_text | 21 |
| task1_outcome | tslib_sota | full | 21 |
| task2_forecast | autofit | core_edgar | 1 |
| task2_forecast | autofit | core_edgar_seed2 | 24 |
| task2_forecast | autofit | core_only | 1 |
| task2_forecast | autofit | core_only_seed2 | 24 |
| task2_forecast | autofit | core_text | 1 |
| task2_forecast | autofit | full | 1 |
| task2_forecast | deep_classical | core_edgar | 9 |
| task2_forecast | deep_classical | core_edgar_seed2 | 9 |
| task2_forecast | deep_classical | core_only | 9 |
| task2_forecast | deep_classical | core_only_seed2 | 9 |
| task2_forecast | deep_classical | core_text | 9 |
| task2_forecast | deep_classical | full | 9 |
| task2_forecast | foundation | core_edgar | 6 |
| task2_forecast | foundation | core_edgar_seed2 | 4 |
| task2_forecast | foundation | core_only | 6 |
| task2_forecast | foundation | core_only_seed2 | 6 |
| task2_forecast | foundation | core_text | 6 |
| task2_forecast | foundation | full | 6 |
| task2_forecast | irregular | core_edgar | 4 |
| task2_forecast | irregular | core_edgar_seed2 | 4 |
| task2_forecast | irregular | core_only | 4 |
| task2_forecast | irregular | core_only_seed2 | 4 |
| task2_forecast | irregular | core_text | 4 |
| task2_forecast | irregular | full | 4 |
| task2_forecast | ml_tabular | core_edgar | 10 |
| task2_forecast | ml_tabular | core_edgar_seed2 | 10 |
| task2_forecast | ml_tabular | core_only | 10 |
| task2_forecast | ml_tabular | core_only_seed2 | 10 |
| task2_forecast | ml_tabular | core_text | 9 |
| task2_forecast | ml_tabular | full | 9 |
| task2_forecast | statistical | core_edgar | 14 |
| task2_forecast | statistical | core_edgar_seed2 | 14 |
| task2_forecast | statistical | core_only | 14 |
| task2_forecast | statistical | core_only_seed2 | 14 |
| task2_forecast | statistical | core_text | 14 |
| task2_forecast | statistical | full | 14 |
| task2_forecast | transformer_sota | core_edgar | 19 |
| task2_forecast | transformer_sota | core_edgar_seed2 | 19 |
| task2_forecast | transformer_sota | core_only | 19 |
| task2_forecast | transformer_sota | core_only_seed2 | 19 |
| task2_forecast | transformer_sota | core_text | 19 |
| task2_forecast | transformer_sota | full | 19 |
| task2_forecast | tslib_sota | core_edgar | 21 |
| task2_forecast | tslib_sota | core_edgar_seed2 | 19 |
| task2_forecast | tslib_sota | core_only | 21 |
| task2_forecast | tslib_sota | core_only_seed2 | 19 |
| task2_forecast | tslib_sota | core_text | 21 |
| task2_forecast | tslib_sota | full | 21 |
| task3_risk_adjust | autofit | core_edgar | 1 |
| task3_risk_adjust | autofit | core_edgar_seed2 | 24 |
| task3_risk_adjust | autofit | core_only | 1 |
| task3_risk_adjust | autofit | core_text | 1 |
| task3_risk_adjust | autofit | full | 1 |
| task3_risk_adjust | deep_classical | core_edgar | 9 |
| task3_risk_adjust | deep_classical | core_edgar_seed2 | 9 |
| task3_risk_adjust | deep_classical | core_only | 9 |
| task3_risk_adjust | deep_classical | core_text | 9 |
| task3_risk_adjust | deep_classical | full | 9 |
| task3_risk_adjust | foundation | core_edgar | 6 |
| task3_risk_adjust | foundation | core_edgar_seed2 | 4 |
| task3_risk_adjust | foundation | core_only | 6 |
| task3_risk_adjust | foundation | core_text | 6 |
| task3_risk_adjust | foundation | full | 6 |
| task3_risk_adjust | irregular | core_edgar | 4 |
| task3_risk_adjust | irregular | core_edgar_seed2 | 4 |
| task3_risk_adjust | irregular | core_only | 4 |
| task3_risk_adjust | irregular | core_text | 4 |
| task3_risk_adjust | irregular | full | 4 |
| task3_risk_adjust | ml_tabular | core_edgar | 10 |
| task3_risk_adjust | ml_tabular | core_edgar_seed2 | 10 |
| task3_risk_adjust | ml_tabular | core_only | 10 |
| task3_risk_adjust | ml_tabular | core_text | 9 |
| task3_risk_adjust | ml_tabular | full | 9 |
| task3_risk_adjust | statistical | core_edgar | 14 |
| task3_risk_adjust | statistical | core_edgar_seed2 | 14 |
| task3_risk_adjust | statistical | core_only | 14 |
| task3_risk_adjust | statistical | core_text | 14 |
| task3_risk_adjust | statistical | full | 14 |
| task3_risk_adjust | transformer_sota | core_edgar | 19 |
| task3_risk_adjust | transformer_sota | core_edgar_seed2 | 19 |
| task3_risk_adjust | transformer_sota | core_only | 19 |
| task3_risk_adjust | transformer_sota | core_text | 19 |
| task3_risk_adjust | transformer_sota | full | 19 |
| task3_risk_adjust | tslib_sota | core_edgar | 21 |
| task3_risk_adjust | tslib_sota | core_edgar_seed2 | 19 |
| task3_risk_adjust | tslib_sota | core_only | 21 |
| task3_risk_adjust | tslib_sota | core_only_seed2 | 15 |
| task3_risk_adjust | tslib_sota | core_text | 21 |
| task3_risk_adjust | tslib_sota | full | 21 |

---
_Last updated: 2026-03-28 16:56:06_