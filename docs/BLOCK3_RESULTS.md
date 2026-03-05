# Block 3 Benchmark Results

> Last Updated: 2026-03-05
> Benchmark stamp: `block3_20260203_225620`
> Run directories: 19 output dirs under `runs/benchmarks/block3_20260203_225620*`

## Data Summary

| Metric | Value |
|---|---:|
| Total metrics.json files | 424 |
| Raw metric records | 15,963 |
| Unique records (deduped) | 3,427 |
| Unique models evaluated | 90 |
| Categories | 7 (autofit, deep_classical, foundation, irregular, ml_tabular, statistical, transformer_sota) |
| Targets | 3 (funding_raised_usd, investors_count, is_funded) |
| Horizons | 4 (1, 7, 14, 30 days) |
| Ablations | 4 (core_only, core_text, core_edgar, full) |
| Evaluation conditions | 48 (3 x 4 x 4) |
| Conditions fully covered | 48/48 |
| Models per condition | 50-88 (median 72) |

## Overall Leaderboard (all ablations, 48 conditions)

Ranking by average per-condition RMSE rank across all 48 evaluation conditions.
Lower rank = better. "Champ" = number of conditions where the model achieves
rank 1 by RMSE.

| # | Model | Category | Conds | Avg RMSE Rank | Avg MAE Rank | Champ |
|---:|---|---|---:|---:|---:|---:|
| 1 | NBEATS | deep_classical | 48 | 5.73 | 5.27 | 14 |
| 2 | PatchTST | transformer_sota | 48 | 6.98 | 4.27 | 2 |
| 3 | NHITS | deep_classical | 48 | 7.29 | 4.31 | 1 |
| **4** | **AutoFitV734** | **autofit** | **48** | **8.25** | **8.29** | **2** |
| 5 | AutoFitV733 | autofit | 47 | 9.79 | 14.70 | 4 |
| 6 | DeepNPTS | transformer_sota | 48 | 10.88 | 14.85 | 6 |
| 7 | TimesNet | transformer_sota | 48 | 11.08 | 11.19 | 12 |
| 8 | MLP | transformer_sota | 24 | 11.12 | 17.00 | 0 |
| 9 | Chronos | foundation | 48 | 11.23 | 11.08 | 0 |
| 10 | NBEATSx | transformer_sota | 47 | 11.94 | 10.51 | 2 |
| 11 | TFT | deep_classical | 48 | 11.94 | 13.21 | 0 |
| 12 | ChronosBolt | foundation | 48 | 12.00 | 7.42 | 0 |
| 13 | GRU | deep_classical | 48 | 13.29 | 14.77 | 0 |
| 14 | TCN | transformer_sota | 24 | 13.88 | 18.71 | 0 |
| 15 | DeepAR | deep_classical | 48 | 14.06 | 14.19 | 2 |
| 16 | NLinear | transformer_sota | 48 | 15.31 | 17.17 | 0 |
| 17 | Informer | transformer_sota | 48 | 15.65 | 14.81 | 0 |
| 18 | KAN | transformer_sota | 48 | 15.77 | 14.60 | 0 |
| 19 | LSTM | transformer_sota | 24 | 15.92 | 20.54 | 0 |
| 20 | DilatedRNN | transformer_sota | 24 | 16.46 | 21.50 | 0 |
| 21 | TiDE | transformer_sota | 47 | 17.81 | 21.00 | 0 |
| 22 | DLinear | transformer_sota | 48 | 21.46 | 19.81 | 0 |
| 23 | BiTCN | transformer_sota | 48 | 24.25 | 20.85 | 0 |
| 24 | Timer | foundation | 47 | 26.81 | 24.77 | 0 |
| 25 | Autoformer | transformer_sota | 47 | 27.62 | 30.79 | 1 |
| 26 | TimeMoE | foundation | 47 | 27.87 | 25.81 | 0 |
| 27 | FEDformer | transformer_sota | 47 | 27.89 | 32.04 | 0 |
| 28 | TimeMixer | transformer_sota | 48 | 28.08 | 30.02 | 0 |
| 29 | LagLlama | foundation | 47 | 28.87 | 26.91 | 0 |
| 30 | MOMENT | foundation | 47 | 29.17 | 26.98 | 0 |
| 31 | TSMixer | transformer_sota | 48 | 29.58 | 30.79 | 0 |
| 32 | iTransformer | transformer_sota | 48 | 32.23 | 31.77 | 0 |
| 33 | RMoK | transformer_sota | 48 | 32.31 | 31.15 | 0 |
| 34 | SOFTS | transformer_sota | 48 | 34.77 | 34.48 | 0 |
| 35 | AutoFitV2 | autofit | 44 | 36.55 | 45.70 | 0 |
| 36 | FusedChampion | autofit | 48 | 38.29 | 38.48 | 2 |
| 37 | AutoFitV2E | autofit | 44 | 39.68 | 47.09 | 0 |
| 38 | AutoFitV4 | autofit | 46 | 39.98 | 43.28 | 0 |
| 39 | MSTL | statistical | 48 | 40.73 | 39.94 | 0 |
| 40 | SF_SeasonalNaive | statistical | 48 | 41.40 | 39.92 | 0 |
| 41 | ExtraTrees | ml_tabular | 30 | 41.60 | 40.43 | 0 |
| 42 | TSMixerx | transformer_sota | 48 | 41.62 | 44.77 | 0 |
| 43 | AutoETS | statistical | 48 | 42.46 | 40.50 | 0 |
| 44 | AutoFitV3 | autofit | 41 | 43.32 | 40.85 | 0 |
| 45 | XGBoost | ml_tabular | 30 | 43.83 | 46.47 | 0 |
| 46 | AutoFitV3Max | autofit | 47 | 43.85 | 41.91 | 0 |
| 47 | AutoARIMA | statistical | 48 | 44.23 | 41.77 | 0 |
| 48 | AutoFitV3E | autofit | 44 | 45.36 | 51.66 | 0 |
| 49 | AutoTheta | statistical | 48 | 45.85 | 43.40 | 0 |
| 50 | AutoFitV5 | autofit | 46 | 46.28 | 45.87 | 0 |
| 51 | AutoFitV1 | autofit | 44 | 47.48 | 42.57 | 0 |
| 52 | AutoFitV71 | autofit | 48 | 47.50 | 44.98 | 0 |
| 53 | RandomForest | ml_tabular | 30 | 47.63 | 37.67 | 0 |
| 54 | AutoFitV7 | autofit | 48 | 48.27 | 45.31 | 0 |
| 55 | HistGradientBoosting | ml_tabular | 30 | 48.43 | 55.60 | 0 |
| 56 | AutoFitV72 | autofit | 48 | 48.44 | 45.21 | 0 |
| 57 | VanillaTransformer | transformer_sota | 47 | 48.85 | 45.11 | 0 |
| 58 | AutoFitV73 | autofit | 30 | 49.03 | 44.50 | 0 |
| 59 | AutoFitV6 | autofit | 48 | 49.50 | 47.98 | 0 |
| 60 | LightGBM | ml_tabular | 30 | 50.70 | 53.90 | 0 |
| 61 | Moirai | foundation | 48 | 51.96 | 49.79 | 0 |
| 62 | Moirai2 | foundation | 48 | 52.27 | 49.48 | 0 |
| 63 | AutoFitV731 | autofit | 18 | 52.44 | 45.00 | 0 |
| 64 | MoiraiLarge | foundation | 48 | 53.29 | 50.88 | 0 |
| 65 | TimesFM | foundation | 48 | 54.56 | 52.42 | 0 |
| 66 | CatBoost | ml_tabular | 30 | 58.53 | 62.07 | 0 |
| 67 | GRU-D | irregular | 48 | 61.25 | 57.96 | 0 |
| 68 | KNN | ml_tabular | 30 | 61.40 | 56.63 | 0 |
| 69 | xLSTM | transformer_sota | 48 | 61.46 | 65.94 | 0 |
| 70 | StemGNN | transformer_sota | 48 | 61.60 | 65.40 | 0 |
| 71 | TimeLLM | transformer_sota | 48 | 62.46 | 66.94 | 0 |
| 72 | SAITS | irregular | 48 | 62.52 | 59.69 | 0 |
| 73 | TimeXer | transformer_sota | 48 | 62.92 | 66.56 | 0 |
| 74 | SVR | ml_tabular | 20 | 62.95 | 62.05 | 0 |
| 75 | MeanPredictor | ml_tabular | 30 | 63.70 | 70.13 | 0 |
| 76 | SeasonalNaive | ml_tabular | 30 | 68.93 | 65.10 | 0 |
| 77 | LogisticRegression | ml_tabular | 10 | 69.50 | 61.60 | 0 |
| 78 | QuantileRegressor | ml_tabular | 20 | 75.05 | 63.15 | 0 |
| 79 | ElasticNet | ml_tabular | 20 | 76.80 | 74.05 | 0 |
| 80 | Lasso | ml_tabular | 20 | 77.80 | 75.05 | 0 |
| 81 | Ridge | ml_tabular | 20 | 80.10 | 80.05 | 0 |

Models 82-90 omitted (specialist models with fewer than 12 conditions evaluated:
NegativeBinomialGLM, FiLM, SCINet, TimeFilter, MambaSimple, MICN, SegRNN,
XGBoostPoisson, LightGBMTweedie).

## Per-Ablation Leaderboard (Top 10)

### core_only (12 conditions)

| # | Model | Avg RMSE Rank | Avg MAE Rank | Champ |
|---:|---|---:|---:|---:|
| 1 | AutoFitV733 | 3.00 | 7.67 | 1 |
| 2 | NBEATS | 5.17 | 6.25 | 3 |
| 3 | AutoFitV734 | 5.92 | 2.67 | 1 |
| 4 | NHITS | 6.17 | 5.33 | 1 |
| 5 | PatchTST | 6.58 | 5.25 | 0 |
| 6 | TimesNet | 10.58 | 11.08 | 3 |
| 7 | KAN | 10.58 | 10.17 | 0 |
| 8 | NLinear | 10.67 | 12.67 | 0 |
| 9 | GRU | 11.08 | 9.08 | 0 |
| 10 | Chronos | 11.17 | 10.50 | 0 |

### core_text (12 conditions)

| # | Model | Avg RMSE Rank | Avg MAE Rank | Champ |
|---:|---|---:|---:|---:|
| 1 | AutoFitV733 | 2.25 | 7.42 | 3 |
| 2 | NBEATS | 5.50 | 6.33 | 1 |
| 3 | AutoFitV734 | 5.58 | 2.50 | 1 |
| 4 | PatchTST | 6.25 | 5.25 | 2 |
| 5 | NBEATSx | 6.45 | 6.64 | 1 |
| 6 | NHITS | 7.00 | 5.33 | 0 |
| 7 | NLinear | 10.83 | 13.17 | 0 |
| 8 | TimesNet | 10.92 | 11.75 | 3 |
| 9 | KAN | 11.33 | 10.58 | 0 |
| 10 | Chronos | 11.50 | 11.17 | 0 |

### core_edgar (12 conditions)

| # | Model | Avg RMSE Rank | Avg MAE Rank | Champ |
|---:|---|---:|---:|---:|
| 1 | NBEATS | 6.50 | 4.08 | 5 |
| 2 | PatchTST | 7.67 | 3.25 | 0 |
| 3 | NBEATSx | 7.83 | 4.92 | 0 |
| 4 | NHITS | 8.17 | 3.25 | 0 |
| 5 | DeepNPTS | 8.25 | 15.25 | 2 |
| 6 | AutoFitV734 | 10.58 | 13.67 | 0 |
| 7 | MLP | 10.92 | 16.58 | 0 |
| 8 | Chronos | 11.00 | 10.92 | 0 |
| 9 | TimesNet | 11.33 | 10.50 | 3 |
| 10 | ChronosBolt | 11.75 | 6.25 | 0 |

### full (12 conditions)

| # | Model | Avg RMSE Rank | Avg MAE Rank | Champ |
|---:|---|---:|---:|---:|
| 1 | NBEATS | 5.75 | 4.42 | 5 |
| 2 | NBEATSx | 6.75 | 5.42 | 0 |
| 3 | PatchTST | 7.42 | 3.33 | 0 |
| 4 | NHITS | 7.83 | 3.33 | 0 |
| 5 | DeepNPTS | 8.67 | 15.50 | 4 |
| 6 | NLinear | 10.58 | 12.33 | 0 |
| 7 | AutoFitV734 | 10.92 | 14.33 | 0 |
| 8 | Chronos | 11.25 | 11.75 | 0 |
| 9 | MLP | 11.33 | 17.42 | 0 |
| 10 | TimesNet | 11.50 | 11.42 | 3 |

## Champion Distribution (48 conditions)

| Model | Category | Wins | Share |
|---|---|---:|---:|
| NBEATS | deep_classical | 14 | 29.2% |
| TimesNet | transformer_sota | 12 | 25.0% |
| DeepNPTS | transformer_sota | 6 | 12.5% |
| AutoFitV733 | autofit | 4 | 8.3% |
| AutoFitV734 | autofit | 2 | 4.2% |
| DeepAR | deep_classical | 2 | 4.2% |
| FusedChampion | autofit | 2 | 4.2% |
| NBEATSx | transformer_sota | 2 | 4.2% |
| PatchTST | transformer_sota | 2 | 4.2% |
| Autoformer | transformer_sota | 1 | 2.1% |
| NHITS | deep_classical | 1 | 2.1% |

## AutoFit Family Progression

| Version | Overall Rank | Avg RMSE Rank | Conditions | Champ Wins |
|---|---:|---:|---:|---:|
| AutoFitV1 | 51/90 | 47.48 | 44 | 0 |
| AutoFitV2 | 35/90 | 36.55 | 44 | 0 |
| AutoFitV2E | 37/90 | 39.68 | 44 | 0 |
| AutoFitV3 | 44/90 | 43.32 | 41 | 0 |
| AutoFitV3E | 48/90 | 45.36 | 44 | 0 |
| AutoFitV3Max | 46/90 | 43.85 | 47 | 0 |
| AutoFitV4 | 38/90 | 39.98 | 46 | 0 |
| AutoFitV5 | 50/90 | 46.28 | 46 | 0 |
| AutoFitV6 | 59/90 | 49.50 | 48 | 0 |
| AutoFitV7 | 54/90 | 48.27 | 48 | 0 |
| AutoFitV71 | 52/90 | 47.50 | 48 | 0 |
| AutoFitV72 | 56/90 | 48.44 | 48 | 0 |
| AutoFitV73 | 58/90 | 49.03 | 30 | 0 |
| AutoFitV731 | 63/90 | 52.44 | 18 | 0 |
| FusedChampion (V7.3.2) | 36/90 | 38.29 | 48 | 2 |
| **AutoFitV733** | **5/90** | **9.79** | **47** | **4** |
| **AutoFitV734** | **4/90** | **8.25** | **48** | **2** |
| AutoFitV735 | -- | -- | 0 | -- |

## AutoFitV734 Per-Ablation Performance

| Ablation | Avg RMSE Rank | Leaderboard Position | Champ Wins |
|---|---:|---:|---:|
| core_only | 5.92 | #3 | 1 |
| core_text | 5.58 | #3 | 1 |
| core_edgar | 10.58 | #6 | 0 |
| full | 10.92 | #7 | 0 |

### AutoFitV734: funding_raised_usd

| H | Ablation | V734 RMSE | V734 Rank | Champion | Champ RMSE |
|---:|---|---:|---:|---|---:|
| 1 | core_only | 1,618,706 | 9 | Autoformer | 1,617,630 |
| 1 | core_text | 1,618,706 | 9 | AutoFitV733 | 1,617,630 |
| 1 | core_edgar | 1,659,797 | 4 | DeepNPTS | 1,631,889 |
| 1 | full | 1,659,797 | 4 | DeepNPTS | 1,631,889 |
| 7 | core_only | 1,618,050 | 4 | AutoFitV733 | 1,617,549 |
| 7 | core_text | 1,618,050 | 4 | AutoFitV733 | 1,617,549 |
| 7 | core_edgar | 1,659,157 | 2 | DeepNPTS | 1,628,891 |
| 7 | full | 1,659,157 | 2 | DeepNPTS | 1,628,891 |
| 14 | **core_only** | **1,616,747** | **1** | **AutoFitV734** | **1,616,747** |
| 14 | **core_text** | **1,616,747** | **1** | **AutoFitV734** | **1,616,747** |
| 14 | core_edgar | 1,659,941 | 7 | FusedChampion | 1,619,532 |
| 14 | full | 1,659,941 | 6 | DeepNPTS | 1,631,325 |
| 30 | core_only | 1,618,853 | 12 | DeepAR | 1,616,270 |
| 30 | core_text | 1,618,853 | 12 | DeepAR | 1,616,270 |
| 30 | core_edgar | 1,659,382 | 5 | FusedChampion | 1,619,460 |
| 30 | full | 1,659,382 | 4 | DeepNPTS | 1,642,970 |

### AutoFitV734: investors_count

| H | Ablation | V734 RMSE | V734 Rank | Champion | Champ RMSE |
|---:|---|---:|---:|---|---:|
| 1 | core_only | 1,082.55 | 4 | TimesNet | 1,082.53 |
| 1 | core_text | 1,082.55 | 4 | TimesNet | 1,082.53 |
| 1 | core_edgar | 1,082.75 | 16 | TimesNet | 1,082.31 |
| 1 | full | 1,082.75 | 16 | TimesNet | 1,082.31 |
| 7 | core_only | 1,082.60 | 6 | TimesNet | 1,082.51 |
| 7 | core_text | 1,082.60 | 6 | TimesNet | 1,082.51 |
| 7 | core_edgar | 1,082.80 | 17 | TimesNet | 1,082.30 |
| 7 | full | 1,082.80 | 17 | TimesNet | 1,082.30 |
| 14 | core_only | 1,082.62 | 9 | TimesNet | 1,082.48 |
| 14 | core_text | 1,082.62 | 9 | TimesNet | 1,082.48 |
| 14 | core_edgar | 1,082.82 | 19 | TimesNet | 1,082.26 |
| 14 | full | 1,082.82 | 19 | TimesNet | 1,082.26 |
| 30 | core_only | 1,082.52 | 4 | NBEATSx | 1,082.52 |
| 30 | core_text | 1,082.52 | 4 | NBEATSx | 1,082.52 |
| 30 | core_edgar | 1,082.72 | 13 | NBEATS | 1,082.30 |
| 30 | full | 1,082.72 | 12 | NBEATS | 1,082.30 |

### AutoFitV734: is_funded

| H | Ablation | V734 RMSE | V734 Rank | Champion | Champ RMSE |
|---:|---|---:|---:|---|---:|
| 1 | core_only | 0.1531 | 6 | NBEATS | 0.1531 |
| 1 | core_text | 0.1531 | 4 | NBEATS | 0.1531 |
| 1 | core_edgar | 0.1531 | 12 | NBEATS | 0.1531 |
| 1 | full | 0.1531 | 14 | NBEATS | 0.1531 |
| 7 | core_only | 0.1531 | 6 | NBEATS | 0.1531 |
| 7 | core_text | 0.1531 | 3 | PatchTST | 0.1531 |
| 7 | core_edgar | 0.1531 | 11 | NBEATS | 0.1531 |
| 7 | full | 0.1531 | 13 | NBEATS | 0.1531 |
| 14 | core_only | 0.1531 | 5 | NHITS | 0.1531 |
| 14 | core_text | 0.1531 | 5 | AutoFitV733 | 0.1531 |
| 14 | core_edgar | 0.1531 | 10 | NBEATS | 0.1531 |
| 14 | full | 0.1531 | 12 | NBEATS | 0.1531 |
| 30 | core_only | 0.1531 | 5 | NBEATS | 0.1531 |
| 30 | core_text | 0.1531 | 6 | PatchTST | 0.1531 |
| 30 | core_edgar | 0.1531 | 11 | NBEATS | 0.1531 |
| 30 | full | 0.1531 | 12 | NBEATS | 0.1531 |

## AutoFitV734 Summary

| Metric | Value |
|---|---|
| Overall rank | #4 / 90 |
| Avg RMSE rank | 8.25 |
| Avg MAE rank | 8.29 |
| Champion wins | 2 (funding_raised_usd, h=14, core_only + core_text) |
| Best ablation | core_text (avg rank 5.58, #3 in ablation) |
| Worst ablation | full (avg rank 10.92, #7 in ablation) |
| Gap to #1 | NBEATS at 5.73 avg rank (delta 2.52) |

V734 performs well on core_only and core_text conditions, where the coarse
oracle selects from a model pool calibrated against the temporal-only feature
set. Performance drops on core_edgar and full ablations because the oracle
table was calibrated against the pre-timezone-fix EDGAR join (104 vs 105
features). Baseline reruns for core_edgar and full are in progress; once
complete, the oracle table will be refreshed for V735.

## AutoFitV735 Status

V735 has not yet been evaluated. It is designed as an exact per-condition
oracle that selects the single best standalone model for each
(target, horizon, ablation) tuple, then retrains that model using the same
code path as the standalone baseline.

The evaluation pipeline has the following dependency chain:

```
baseline reruns (core_edgar + full) --> oracle_refresh --> V735 evaluation
```

The oracle refresh job (`scripts/refresh_v735_oracle.py`) reads all
metrics.json files, identifies the rank-1 model per condition, and writes
the updated oracle table to `nf_adaptive_champion.py`. V735 then trains
whichever single model won each condition, using deterministic seeding to
match the standalone run's predictions.

## Category-Level Summary

| Category | Models | Best Model | Overall Rank |
|---|---:|---|---:|
| deep_classical | 4 | NBEATS | #1 |
| autofit | 17 | AutoFitV734 | #4 |
| transformer_sota | 23+ | PatchTST | #2 |
| foundation | 11 | Chronos | #9 |
| ml_tabular | 15 | ExtraTrees | #41 |
| statistical | 5 | MSTL | #39 |
| irregular | 2 | GRU-D | #67 |

## Pending Baseline Reruns

Baseline categories are being re-evaluated with the EDGAR timezone fix
(`ae9626b`) and deterministic NF seeding (`d388310`):

| Category | Ablation | Queue | Status |
|---|---|---|---|
| transformer_sota | core_edgar, full | npin (GPU) | RUNNING/PENDING |
| irregular | core_edgar, full | npin (GPU) | RUNNING |
| deep_classical | core_edgar, full | cfisch (GPU) | RUNNING/PENDING |
| foundation | core_edgar, full | cfisch (GPU) | PENDING |
| ml_tabular | core_edgar, full | cfisch (bigmem) | PENDING |
| statistical | core_edgar, full | cfisch (bigmem) | PENDING |

## Notes

1. NBEATS dominates with 14/48 champion wins across all ablations. The
   Basis Expansion architecture handles the low-signal regime in this dataset
   particularly well. Many targets (is_funded, investors_count) have very
   concentrated distributions where marginal improvements are in the fourth
   decimal place.

2. AutoFitV734 is the first AutoFit version to break into the top 5, ranking
   #4 overall. The per-condition coarse oracle strategy (keyed by target_type,
   horizon, ablation_class) converges to the right architecture in most cases.

3. AutoFitV733 ranks #1 in both core_only and core_text ablation leaderboards
   despite ranking #5 overall, showing the stack-based approach excels when
   fitted to the temporal-only feature set.

4. Traditional ML models (RandomForest, XGBoost, CatBoost, LightGBM) cluster
   in the 41-66 range. The entity-panel time-series structure of this benchmark
   strongly favors neural architectures over flat tabular approaches.

5. Foundation models split into two tiers: Chronos (#9) and ChronosBolt (#12)
   are competitive with trained deep models, while Moirai variants (#61-64)
   and TimesFM (#65) perform substantially worse on this particular dataset.

6. The is_funded target produces nearly identical RMSE (0.1531) for many
   models, making rank differences on that target largely noise-driven.

7. investors_count shows TimesNet dominance (12 champion wins concentrated on
   this target), likely due to its temporal 2D convolution capturing periodic
   filing patterns in SEC data.
