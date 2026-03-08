# Block 3 Benchmark Results

> Last updated: 2026-03-08 (Phase 9 fair benchmark)
> Canonical results dir: `runs/benchmarks/block3_phase9_fair/`
> Phase 7/8 results are **DEPRECATED** (4 critical bugs fixed)

**Generated**: 2026-03-08 02:20:13
**Benchmark Dir**: `block3_phase9_fair`
**Total Records (post-filter)**: 4847

## Overview

| Metric | Value |
|--------|-------|
| Raw records | 5083 |
| Filtered records | 4847 |
| Comparability filter | fairness_only=True, min_coverage=0.98 |
| Models evaluated | 50 |
| Categories | autofit, deep_classical, foundation, irregular, statistical, transformer_sota |
| Tasks | task1_outcome, task2_forecast, task3_risk_adjust |
| Total evaluations | 4847 |
| Real results | 4835 |
| Fallback (mean) | 12 |

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
_Last updated: 2026-03-08 02:20:14_