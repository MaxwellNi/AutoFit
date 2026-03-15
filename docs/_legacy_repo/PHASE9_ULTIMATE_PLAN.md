# Phase 9 — Ultimate Fair Benchmark Plan

> Created: 2026-03-08
> Status: READY TO SUBMIT
> Supersedes: Phase 7/8 results (invalidated by critical bug fixes)

## Executive Summary

Phase 9 re-runs the **entire** Block 3 benchmark from scratch after discovering
and fixing **4 critical experimental bugs** that compromised fairness of prior
results. Additionally, 18 new SOTA models are integrated.

**Total: 100 models × 104 conditions = 10,400 expected metric records**

## Critical Bugs Fixed (Invalidate ALL Prior Results)

### Bug 1: TSLib Single-Window Prediction (ALL entities identical)
- **File**: `src/narrative/block3/models/tslib_models.py` `predict()`
- **Problem**: Took last `seq_len` rows of stacked test DataFrame (mixing entities)
  as ONE window. All entities received identical predictions.
- **Impact**: 20 TSLib models × all conditions = garbage results
- **Fix**: Per-entity batched inference. During `fit()`, save per-entity context
  windows. During `predict()`, batch forward pass per-entity and map predictions.

### Bug 2: Foundation Model Hardcoded `prediction_length=7`
- **File**: `src/narrative/block3/models/deep_models.py`
- **Problem**: Chronos, ChronosBolt, Chronos2 all called
  `self._model.predict(tensors, 7)` regardless of actual horizon.
- **Impact**: For h=1, h=14, h=30 — wrong forecast window used
- **Fix**: `pred_horizon = max(kwargs.get("horizon", 7), 1)` used throughout

### Bug 3: Moirai/Moirai2 50-Entity Cap
- **File**: `src/narrative/block3/models/deep_models.py`
- **Problem**: `ctxs[:50]` limited prediction to 50 entities while Chronos
  processed ALL entities. Unfair comparison.
- **Impact**: Entity coverage ratio << 1.0 for Moirai
- **Fix**: Removed cap, processes ALL entities

### Bug 4: LagLlama Hardcoded + 50-Entity Cap
- **File**: `src/narrative/block3/models/deep_models.py`
- **Problem**: `prediction_length=7` hardcoded in `LagLlamaEstimator`,
  `ctxs[:50]` capped entities
- **Fix**: Default prediction_length=30 (covers all horizons), processes ALL entities

## Configuration Fixes (Equalization)

| Fix | Before | After | Models Affected |
|-----|--------|-------|----------------|
| NF max_steps | Original committed values | Original committed values | No change (reverted) |
| NF val_check_steps | Original committed values | Original committed values | No change (reverted) |
| NF patience | Original committed values | Original committed values | No change (reverted) |
| BiTCN hidden_size | Original committed value | Original committed value | No change (reverted) |
| TimesNet scaler | Original committed value | Original committed value | No change (reverted) |
| iTransformer lr | Original committed value | Original committed value | No change (reverted) |
| TiDE lr | Original committed value | Original committed value | No change (reverted) |
| xLSTM lr | Original committed value | Original committed value | No change (reverted) |
| TSLib epochs | 50 | 100 | All 20 TSLib models |
| TSLib patience | 7 | 15 | All 20 TSLib models |
| TSLib normalization | Test-time stats | Training stats | All 20 TSLib models |
| ml_tabular horizons | [horizons[0]] only | ALL horizons | All 15 ML models |

## New Models Added

### TSLib SOTA (+6, total: 20)
| Model | Venue | Year |
|-------|-------|------|
| ETSformer | ICML | 2023 |
| LightTS | AAAI | 2022 |
| Pyraformer | ICLR | 2022 |
| Reformer | ICLR | 2020 |
| TiRex | ICLR | 2025 |
| Mamba | ICLR | 2024 |

### Statistical (+10, total: 15)
| Model | Source |
|-------|--------|
| CrostonClassic | StatsForecast |
| CrostonOptimized | StatsForecast |
| CrostonSBA | StatsForecast |
| DynamicOptimizedTheta | StatsForecast |
| AutoCES | StatsForecast |
| Holt | StatsForecast |
| HoltWinters | StatsForecast |
| Naive | StatsForecast |
| HistoricAverage | StatsForecast |
| WindowAverage | StatsForecast |

### Irregular (+2, total: 4)
| Model | Venue | Year |
|-------|-------|------|
| BRITS | NeurIPS | 2018 |
| CSDI | NeurIPS | 2021 |

## Complete Model Registry (100 models)

| Category | Count | Models |
|----------|------:|--------|
| statistical | 15 | AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive, CrostonClassic, CrostonOptimized, CrostonSBA, DynamicOptimizedTheta, AutoCES, Holt, HoltWinters, Naive, HistoricAverage, WindowAverage |
| ml_tabular | 15 | LogisticRegression, Ridge, Lasso, ElasticNet, SVR, DecisionTree, RandomForest, ExtraTrees, HistGradientBoosting, AdaBoost, BaggingRegressor, LightGBM, XGBoost, CatBoost, MeanPredictor |
| deep_classical | 9 | NBEATS, NHITS, TFT, DeepAR, GRU, LSTM, TCN, MLP, DilatedRNN |
| transformer_sota | 23 | PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx, BiTCN, KAN, RMoK, SOFTS, StemGNN, DLinear, NLinear, TimeMixer, TimeXer, TSMixerx, xLSTM, TimeLLM, DeepNPTS |
| foundation | 11 | Chronos, ChronosBolt, Chronos2, Moirai, MoiraiLarge, Moirai2, Timer, TimeMoE, MOMENT, LagLlama, TimesFM |
| irregular | 4 | GRU-D, SAITS, BRITS, CSDI |
| tslib_sota | 20 | TimeFilter, WPMixer, MultiPatchFormer, TiRex, MSGNet, PAttn, MambaSimple, Mamba, Koopa, FreTS, Crossformer, MICN, SegRNN, ETSformer, NonstationaryTransformer, FiLM, SCINet, LightTS, Pyraformer, Reformer |
| autofit | 3 | AutoFitV734, AutoFitV735, AutoFitV736 |
| **TOTAL** | **100** | |

## Condition Matrix (104 total)

| Task | Targets | Horizons | Ablations | Conditions |
|------|---------|----------|-----------|-----------|
| task1_outcome | total_amount_sold, number_investors, is_funded | 1, 7, 14, 30 | core_only, core_text, core_edgar, full | 48 |
| task2_forecast | total_amount_sold, number_investors | 1, 7, 14, 30 | core_only, core_text, core_edgar, full | 32 |
| task3_risk_adjust | total_amount_sold, number_investors | 1, 7, 14, 30 | core_only, core_edgar, full | 24 |
| **TOTAL** | | | | **104** |

## SLURM Submission Plan

### Script: `scripts/submit_phase9_ultimate.sh`
### Output: `runs/benchmarks/block3_phase9_fair/`

| Shard | Category | Models | Partition | Resources | Jobs |
|-------|----------|-------:|-----------|-----------|-----:|
| 9a | statistical | 15 | batch | 28c, 112GB, 2d | 11 |
| 9b | ml_tabular | 15 | batch | 28c, 112GB, 2d | 11 |
| 9c | deep_classical | 9 | gpu | 1×V100, 14c, 256GB, 2d | 11 |
| 9d | transformer_sota | 23 | gpu | 1×V100, 14c, 320GB, 2d | 22 |
| 9e | foundation | 11 | gpu | 1×V100, 14c, 512GB, 2d | 11 |
| 9f | irregular | 4 | gpu | 1×V100, 14c, 256GB, 2d | 11 |
| 9g | tslib_sota | 20 | gpu | 1×V100, 14c, 640GB, 2d | 33 |
| 9h | autofit | 3 | bigmem | 28c, 1TB, 2d | 11 |
| **Total** | | **100** | | | **121** |

### Account Split (MaxJobsPU=100)
| Account | Shards | Jobs | GPU | Batch | Bigmem |
|---------|--------|-----:|----:|------:|-------:|
| npin | 9a, 9b, 9c, 9d, 9e, 9f | 77 | 55 | 22 | 0 |
| cfisch | 9g, 9h | 44 | 33 | 0 | 11 |

### Execution Commands
```bash
# Dry run (verify job layout)
bash scripts/submit_phase9_ultimate.sh --dry-run

# Submit npin shards
bash scripts/submit_phase9_ultimate.sh --shard 9a
bash scripts/submit_phase9_ultimate.sh --shard 9b
bash scripts/submit_phase9_ultimate.sh --shard 9c
bash scripts/submit_phase9_ultimate.sh --shard 9d
bash scripts/submit_phase9_ultimate.sh --shard 9e
bash scripts/submit_phase9_ultimate.sh --shard 9f

# cfisch runs (from cfisch account):
# Copy repo, set symlinks, then:
# bash scripts/submit_phase9_ultimate.sh --shard 9g
# bash scripts/submit_phase9_ultimate.sh --shard 9h
```

### Post-Run Aggregation
```bash
python3 scripts/aggregate_block3_results.py \
    --input-dir runs/benchmarks/block3_phase9_fair
python3 scripts/make_paper_tables_v2.py \
    --input-dir runs/benchmarks/block3_phase9_fair
```

## Files Modified

| File | Changes |
|------|---------|
| `src/narrative/block3/models/deep_models.py` | Foundation horizon fix, Moirai entity cap removal, LagLlama fix (NF configs unchanged — use committed values) |
| `src/narrative/block3/models/tslib_models.py` | Per-entity prediction, training norm stats, 100 epochs, patience 15, +6 new models |
| `src/narrative/block3/models/statistical.py` | +10 new models (Croston, Holt, AutoCES, Naive, etc.) |
| `src/narrative/block3/models/irregular_models.py` | +2 new models (BRITS, CSDI) |
| `src/narrative/block3/models/registry.py` | Docstring updated to reflect new model counts |
| `scripts/run_block3_benchmark_shard.py` | Removed ml_tabular single-horizon restriction |
| `scripts/submit_phase9_ultimate.sh` | New Phase 9 SLURM submission script |
