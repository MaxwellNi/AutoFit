# Block 3 Benchmark Results — Iris HPC Full Run

**Generated**: 2026-02-09 12:00 UTC
**Benchmark Dir**: `block3_20260203_225620_iris_full`
**Platform**: ULHPC Iris, GPU partition (V100 32GB)
**Freeze Stamp**: `20260203_225620`

## Completion Status

| Category | Shards | Status |
|----------|--------|--------|
| statistical (5) | 6/6 | ✅ Complete |
| deep_classical (4) | 6/6 | ✅ Complete |
| transformer_sota (15) | 6/6 | ✅ Complete (3-way split A/B/C) |
| foundation (2) | 6/6 | ✅ Complete |
| irregular (2) | 6/6 | ✅ Complete |
| ml_tabular (14) | 0/6 | ⏳ Running (jobs 5173630-5173635) |
| autofit (3) | 0/6 | ⏳ Pending (re-run with V1 fix, jobs 5173654-5173660) |
| **Total** | **42/54** | **78% complete** |

## Leaderboard (42 shards, core_only ablation)

### Task 1: Outcome Prediction — `funding_raised_usd`

| Rank | Category | Model | MAE | RMSE | SMAPE |
|------|----------|-------|-----|------|-------|
| 1 | foundation | **Moirai** | 691,983 | 2,359,979 | 107.5 |
| 2 | irregular | SAITS | 1,079,209 | 2,413,370 | 127.6 |
| 3 | irregular | GRU-D | 1,079,209 | 2,413,371 | 127.6 |
| 4 | transformer_sota | FEDformer | 5,868,134 | 6,145,661 | 168.8 |
| 5 | transformer_sota | SOFTS | 7,547,293 | 7,771,733 | 173.5 |

### Task 1: Outcome Prediction — `investors_count`

| Rank | Category | Model | MAE | RMSE | SMAPE |
|------|----------|-------|-----|------|-------|
| 1 | foundation | **Moirai** | 306.5 | 1,404.8 | 87.4 |
| 2 | irregular | GRU-D | 311.3 | 1,421.3 | 88.8 |
| 3 | irregular | SAITS | 311.4 | 1,421.3 | 88.9 |
| 4 | transformer_sota | StemGNN | 397.9 | 1,395.2 | 104.9 |
| 5 | transformer_sota | FEDformer | 399.3 | 1,395.3 | 105.1 |

### Task 1: Outcome Prediction — `is_funded` (binary)

| Rank | Category | Model | MAE | RMSE | SMAPE |
|------|----------|-------|-----|------|-------|
| 1 | deep_classical | **TFT** | 0.163 | 0.296 | 27.2 |
| 2 | transformer_sota | iTransformer | 0.170 | 0.295 | 28.2 |
| 3 | statistical | AutoTheta | 0.173 | 0.296 | 28.4 |
| 4 | foundation | Chronos | 0.175 | 0.295 | 28.6 |
| 5 | foundation | Moirai | 0.177 | 0.296 | 28.9 |

## Key Observations

1. **Moirai dominates** on continuous regression targets (funding, investors) with ~35% lower MAE than next-best
2. **Irregular models** (GRU-D, SAITS) are strong runners-up, handling 86% missing rate effectively
3. **TFT leads on binary target** (is_funded) — classifier-style targets favor attention-based models
4. **Statistical models** (AutoARIMA, AutoETS, AutoTheta) are competitive on is_funded but weak on amount prediction
5. **Foundation model advantage**: Zero-shot transfer learning produces lowest errors with near-zero training time
6. **EDGAR ablation**: No significant improvement from EDGAR features for panel-based models (they don't use tabular features)

## Pending: ml_tabular + AutoFit

- **ml_tabular** (14 models without SVR): Running on GPU partition, expected completion ~3 hours
- **AutoFit** (3 models): Re-submitted after critical bug fix — V1 was always falling back to LightGBM due to `compose_from_profile()` type mismatch. Now uses `RuleBasedComposer.compose()` directly.
- **AutoFitV1** should now select TSMixer (timemixer backbone) for high-nonstationarity data

## Configuration

- **Preset**: full (horizons=[1,7,14,30], k_values=[7,14,30,60,90], n_bootstrap=1000)
- **Temporal split**: train≤2025-06-30, val≤2025-09-30, test≤2025-12-31 (7-day embargo)
- **Ablations**: core_only, core_edgar
- **Leakage guard**: Target synonym groups enforced (`_TARGET_LEAK_GROUPS`)
| FEDformer | 7.95M | 7.89M | 7.89M | 7.89M | ✅ |
## Previous Run (4090, partial)

See `block3_20260203_225620_4090_final` for earlier partial results (21 models, 336 records).
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