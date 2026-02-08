# Block 3 Benchmark Results

**Generated**: 2026-02-08 11:04:34
**Benchmark Dir**: `block3_20260203_225620_4090_final`
**Total Records**: 336

## Overview

| Metric | Value |
|--------|-------|
| Models evaluated | 21 |
| Categories | deep_classical, foundation, transformer_sota |
| Tasks | task1_outcome, task2_forecast |
| Total evaluations | 336 |
| Real results | 292 |
| Fallback (mean) | 44 |

## task1_outcome

### deep_classical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATS | 601.9K | 8.01M | 8.03M | 8.09M | ⚠️ fallback |
| DeepAR | 619.8K | 619.8K | 619.8K | 619.8K | ✅ |
| NHITS | 7.96M | 8.01M | 8.03M | 8.09M | ✅ |
| TFT | 7.98M | 7.98M | 8.00M | 8.01M | ✅ |

### foundation

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| TimesFM | 601.9K | 601.9K | 601.9K | 601.9K | ⚠️ fallback |
| Moirai | 637.8K | 637.8K | 637.8K | 637.8K | ✅ |
| Chronos | 7.96M | 7.97M | 7.96M | 7.97M | ✅ |

### transformer_sota

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATSx | 601.9K | 8.01M | 8.03M | 8.09M | ⚠️ fallback |
| RMoK | 601.9K | 601.9K | 601.9K | 601.9K | ⚠️ fallback |
| SOFTS | 601.9K | 601.9K | 601.9K | 601.9K | ⚠️ fallback |
| StemGNN | 601.9K | 601.9K | 601.9K | 601.9K | ⚠️ fallback |
| TSMixer | 601.9K | 601.9K | 601.9K | 601.9K | ⚠️ fallback |
| iTransformer | 601.9K | 601.9K | 601.9K | 601.9K | ⚠️ fallback |
| Informer | 619.8K | 619.8K | 619.8K | 619.8K | ✅ |
| BiTCN | 7.42M | 6.14M | 5.99M | 4.54M | ✅ |
| KAN | 7.92M | 7.89M | 7.98M | 8.08M | ✅ |
| Autoformer | 7.94M | 7.89M | 7.89M | 7.89M | ✅ |
| FEDformer | 7.95M | 7.89M | 7.89M | 7.89M | ✅ |
| TimesNet | 7.96M | 8.01M | 7.98M | 7.99M | ✅ |
| PatchTST | 7.99M | 7.99M | 7.99M | 7.99M | ✅ |
| TiDE | 8.91M | 8.00M | 8.61M | 8.42M | ✅ |

## task2_forecast

### deep_classical

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATS | 601.9K | 8.01M | 8.03M | 8.09M | ⚠️ fallback |
| DeepAR | 619.8K | 619.8K | 619.8K | 619.8K | ✅ |
| NHITS | 7.96M | 8.01M | 8.03M | 8.09M | ✅ |
| TFT | 7.98M | 7.98M | 8.00M | 8.01M | ✅ |

### foundation

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| TimesFM | 601.9K | 601.9K | 601.9K | 601.9K | ⚠️ fallback |
| Moirai | 637.8K | 637.8K | 637.8K | 637.8K | ✅ |
| Chronos | 7.96M | 7.97M | 7.96M | 7.97M | ✅ |

### transformer_sota

| Model | H=1 | H=7 | H=14 | H=30 | Status |
|-------|-------:|-------:|-------:|-------:|--------|
| NBEATSx | 601.9K | 8.01M | 8.03M | 8.09M | ⚠️ fallback |
| Informer | 619.8K | 619.8K | 619.8K | 619.8K | ✅ |
| BiTCN | 7.42M | 6.14M | 5.99M | 4.54M | ✅ |
| TSMixer | 7.88M | 7.91M | 7.92M | 7.92M | ✅ |
| KAN | 7.92M | 7.89M | 7.98M | 8.08M | ✅ |
| Autoformer | 7.94M | 7.89M | 7.89M | 7.89M | ✅ |
| StemGNN | 7.94M | 7.98M | 8.01M | 8.09M | ✅ |
| FEDformer | 7.95M | 7.89M | 7.89M | 7.89M | ✅ |
| iTransformer | 7.96M | 7.99M | 7.97M | 7.98M | ✅ |
| RMoK | 7.96M | 7.95M | 7.96M | 7.98M | ✅ |
| TimesNet | 7.96M | 8.01M | 7.98M | 7.99M | ✅ |
| SOFTS | 7.98M | 7.98M | 7.99M | 7.98M | ✅ |
| PatchTST | 7.99M | 7.99M | 7.99M | 7.99M | ✅ |
| TiDE | 8.91M | 8.00M | 8.61M | 8.42M | ✅ |

## Training Time Summary

| Model | Avg (s) | Min (s) | Max (s) |
|-------|--------:|--------:|--------:|
| BiTCN | 44.2 | 42.7 | 47.3 |
| FEDformer | 4.2 | 3.7 | 4.8 |
| TimesNet | 3.7 | 3.6 | 4.2 |
| Autoformer | 2.9 | 2.6 | 3.4 |
| PatchTST | 2.9 | 1.1 | 8.2 |
| Informer | 2.8 | 2.7 | 2.9 |
| NBEATS | 2.1 | 0.7 | 6.1 |
| StemGNN | 2.0 | 0.2 | 2.8 |
| iTransformer | 1.7 | 0.9 | 2.2 |
| TFT | 1.7 | 1.5 | 1.9 |
| SOFTS | 1.4 | 0.1 | 2.0 |
| Chronos | 1.1 | 0.6 | 2.4 |
| NHITS | 1.0 | 0.8 | 1.4 |
| RMoK | 1.0 | 0.1 | 1.4 |
| TSMixer | 0.8 | 0.1 | 1.2 |
| DeepAR | 0.8 | 0.7 | 1.3 |
| KAN | 0.7 | 0.7 | 0.8 |
| TiDE | 0.7 | 0.7 | 0.8 |
| Moirai | 0.7 | 0.4 | 2.5 |
| NBEATSx | 0.6 | 0.1 | 0.8 |
| TimesFM | 0.1 | 0.1 | 0.1 |

## Completion Matrix

Shows which task/category/ablation combinations have results.

| Task | Category | Ablation | Models |
|------|----------|----------|-------:|
| task1_outcome | deep_classical | core_only | 4 |
| task1_outcome | deep_classical | full | 4 |
| task1_outcome | foundation | core_only | 3 |
| task1_outcome | foundation | full | 3 |
| task1_outcome | transformer_sota | core_only | 14 |
| task1_outcome | transformer_sota | full | 14 |
| task2_forecast | deep_classical | core_only | 4 |
| task2_forecast | deep_classical | full | 4 |
| task2_forecast | foundation | core_only | 3 |
| task2_forecast | foundation | full | 3 |
| task2_forecast | transformer_sota | core_only | 14 |
| task2_forecast | transformer_sota | full | 14 |

---
_Last updated: 2026-02-08 11:04:34_