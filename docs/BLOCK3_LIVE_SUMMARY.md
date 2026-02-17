# Block 3 Live Summary

- Generated at: **2026-02-17 10:29:57 **
- Freeze stamp: `20260203_225620`
- Canonical benchmark dir: `runs/benchmarks/block3_20260203_225620_phase7`
- V7.1 pilot dir: `runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205`
- V7.1 full dir: `runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737`

## 1. Live Queue and Concurrency

| Metric | Value |
|---|---:|
| Running jobs now | 8 |
| Pending jobs now | 123 |
| Running batch jobs | 8 |
| Running gpu jobs | 0 |
| Pending batch jobs | 112 |
| Pending gpu jobs | 11 |
| Theoretical max concurrency under current QOS | 12 (batch=8 + gpu=4) |
| Effective current concurrency | 8 (batch saturated, gpu waiting on group/node limits) |

## 2. 297-Job Program Status

| Program | Total | Completed | Running | Pending | Failed |
|---|---:|---:|---:|---:|---:|
| p7 | 121 | 99 | 0 | 20 | 2 |
| p7x | 110 | 43 | 1 | 66 | 0 |
| p7xF | 66 | 24 | 7 | 35 | 0 |
| **Overall** | **297** | **166** | **8** | **121** | **2** |

| ETA estimator | Value |
|---|---:|
| Remaining jobs (running+pending+failed) | 131 |
| Completed in last 24h | 13 |
| Completed in last 48h | 48 |
| ETA by 24h throughput | 10.15 days |
| ETA by 48h throughput | 5.50 days |

## 3. Canonical SOTA Panorama (Materialized So Far)

- Materialized records: **4369**
- Models with materialized metrics: **66**
- Categories represented: **7**

### 3.1 Best-in-target snapshots (current materialized state)

| Target | Best model | Category | MAE | Task | Ablation | Horizon |
|---|---|---|---:|---|---|---:|
| funding_raised_usd | NHITS | deep_classical | 374,432.36 | task3_risk_adjust | full | 7 |
| investors_count | NBEATS | deep_classical | 44.7267 | task3_risk_adjust | core_only | 7 |
| is_funded | PatchTST | transformer_sota | 0.032281 | task1_outcome | full | 14 |

### 3.2 Top-20 models by normalized MAE ratio

- Definition: geometric mean of `MAE_model / MAE_best_on_same_key` across each model's materialized keys. Lower is better; `1.00` is ideal.

| Rank | Model | Category | Keys | Normalized ratio | Median MAE funding_raised_usd | Median MAE investors_count | Median MAE is_funded |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | PatchTST | transformer_sota | 95 | 1.0014 | 375,472.40 | 44.8383 | 0.032380 |
| 2 | NHITS | deep_classical | 104 | 1.0015 | 376,672.15 | 44.8030 | 0.033039 |
| 3 | NBEATSx | transformer_sota | 95 | 1.0034 | 378,312.04 | 44.7916 | 0.032644 |
| 4 | NBEATS | deep_classical | 104 | 1.0038 | 378,312.04 | 44.7916 | 0.033310 |
| 5 | ChronosBolt | foundation | 104 | 1.0072 | 375,044.63 | 44.9924 | 0.034105 |
| 6 | KAN | transformer_sota | 92 | 1.0122 | 380,946.12 | 44.8995 | 0.035020 |
| 7 | NLinear | transformer_sota | 92 | 1.0146 | 382,043.80 | 45.3745 | 0.034723 |
| 8 | TimesNet | transformer_sota | 95 | 1.0157 | 378,715.07 | 44.9044 | 0.036246 |
| 9 | DLinear | transformer_sota | 92 | 1.0195 | 384,507.13 | 46.3120 | 0.033090 |
| 10 | Chronos | foundation | 104 | 1.0195 | 374,776.14 | 45.5544 | 0.035666 |
| 11 | Informer | transformer_sota | 95 | 1.0250 | 377,689.50 | 45.2494 | 0.037983 |
| 12 | TiDE | transformer_sota | 95 | 1.0259 | 383,559.12 | 46.6208 | 0.033785 |
| 13 | DeepAR | deep_classical | 104 | 1.0275 | 377,147.18 | 45.2603 | 0.037387 |
| 14 | BiTCN | transformer_sota | 92 | 1.0307 | 381,795.57 | 46.0521 | 0.037839 |
| 15 | TFT | deep_classical | 104 | 1.0336 | 377,034.40 | 45.3469 | 0.038387 |
| 16 | Autoformer | transformer_sota | 95 | 1.1418 | 395,015.41 | 49.1919 | 0.049170 |
| 17 | FEDformer | transformer_sota | 95 | 1.1934 | 381,405.47 | 46.9919 | 0.126842 |
| 18 | TimeMoE | foundation | 102 | 1.2076 | 383,569.93 | 57.7491 | 0.053507 |
| 19 | MOMENT | foundation | 102 | 1.2076 | 383,569.93 | 57.7491 | 0.053507 |
| 20 | Timer | foundation | 102 | 1.2076 | 383,569.93 | 57.7491 | 0.053513 |

### 3.3 Full model table (all materialized models)

| Model | Category | Keys | Normalized ratio | Median funding MAE | Best funding MAE | Median investors MAE | Best investors MAE | Median is_funded MAE | Best is_funded MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PatchTST | transformer_sota | 95 | 1.0014 | 375,472.40 | 374,564.44 | 44.8383 | 44.7718 | 0.032380 | 0.032281 |
| NHITS | deep_classical | 104 | 1.0015 | 376,672.15 | 374,432.36 | 44.8030 | 44.7380 | 0.033039 | 0.032322 |
| NBEATSx | transformer_sota | 95 | 1.0034 | 378,312.04 | 374,514.68 | 44.7916 | 44.7267 | 0.032644 | 0.032389 |
| NBEATS | deep_classical | 104 | 1.0038 | 378,312.04 | 374,514.68 | 44.7916 | 44.7267 | 0.033310 | 0.032463 |
| ChronosBolt | foundation | 104 | 1.0072 | 375,044.63 | 375,044.63 | 44.9924 | 44.9274 | 0.034105 | 0.033394 |
| KAN | transformer_sota | 92 | 1.0122 | 380,946.12 | 374,801.68 | 44.8995 | 44.7450 | 0.035020 | 0.034085 |
| NLinear | transformer_sota | 92 | 1.0146 | 382,043.80 | 375,899.02 | 45.3745 | 44.8941 | 0.034723 | 0.033150 |
| TimesNet | transformer_sota | 95 | 1.0157 | 378,715.07 | 374,906.40 | 44.9044 | 44.8331 | 0.036246 | 0.034819 |
| DLinear | transformer_sota | 92 | 1.0195 | 384,507.13 | 378,362.35 | 46.3120 | 45.9913 | 0.033090 | 0.032354 |
| Chronos | foundation | 104 | 1.0195 | 374,776.14 | 374,610.31 | 45.5544 | 45.4895 | 0.035666 | 0.034949 |
| Informer | transformer_sota | 95 | 1.0250 | 377,689.50 | 375,246.60 | 45.2494 | 45.1816 | 0.037983 | 0.036892 |
| TiDE | transformer_sota | 95 | 1.0259 | 383,559.12 | 377,414.34 | 46.6208 | 45.6893 | 0.033785 | 0.033019 |
| DeepAR | deep_classical | 104 | 1.0275 | 377,147.18 | 374,858.26 | 45.2603 | 45.1744 | 0.037387 | 0.036029 |
| BiTCN | transformer_sota | 92 | 1.0307 | 381,795.57 | 375,650.79 | 46.0521 | 45.9872 | 0.037839 | 0.035418 |
| TFT | deep_classical | 104 | 1.0336 | 377,034.40 | 374,667.59 | 45.3469 | 45.2784 | 0.038387 | 0.035838 |
| Autoformer | transformer_sota | 95 | 1.1418 | 395,015.41 | 378,350.61 | 49.1919 | 48.9593 | 0.049170 | 0.045033 |
| FEDformer | transformer_sota | 95 | 1.1934 | 381,405.47 | 377,676.93 | 46.9919 | 46.8475 | 0.126842 | 0.069965 |
| TimeMoE | foundation | 102 | 1.2076 | 383,569.93 | 383,569.93 | 57.7491 | 57.6841 | 0.053507 | 0.052796 |
| MOMENT | foundation | 102 | 1.2076 | 383,569.93 | 383,569.93 | 57.7491 | 57.6841 | 0.053507 | 0.052796 |
| Timer | foundation | 102 | 1.2076 | 383,569.93 | 383,569.93 | 57.7491 | 57.6841 | 0.053513 | 0.052796 |
| LagLlama | foundation | 102 | 1.2076 | 383,569.93 | 383,569.93 | 57.7491 | 57.6841 | 0.053513 | 0.052796 |
| AutoFitV3 | autofit | 43 | 1.4378 | 400,776.58 | 395,551.54 | 113.9118 | 113.9118 | NA | NA |
| AutoFitV1 | autofit | 43 | 1.4835 | 437,068.19 | 436,701.55 | 113.9118 | 113.9118 | NA | NA |
| RandomForest | ml_tabular | 12 | 1.6585 | 400,776.58 | 400,776.58 | 95.9587 | 95.9587 | 0.089894 | 0.089894 |
| TSMixer | transformer_sota | 95 | 1.6813 | 381,515.92 | 379,362.46 | 120.8671 | 119.3628 | 0.074605 | 0.074459 |
| iTransformer | transformer_sota | 95 | 1.7022 | 381,956.83 | 375,074.35 | 121.3770 | 119.2519 | 0.079874 | 0.079109 |
| TimeMixer | transformer_sota | 92 | 1.7168 | 381,618.55 | 375,473.78 | 120.7220 | 119.2177 | 0.077186 | 0.074741 |
| RMoK | transformer_sota | 92 | 1.7195 | 382,451.57 | 376,306.79 | 120.7363 | 119.2320 | 0.077591 | 0.075015 |
| SOFTS | transformer_sota | 92 | 1.7294 | 382,920.82 | 376,637.72 | 120.8958 | 119.3915 | 0.077717 | 0.074647 |
| AutoFitV2E | autofit | 43 | 1.8229 | 434,083.92 | 428,554.10 | 251.6665 | 222.1337 | NA | NA |
| AutoFitV2 | autofit | 43 | 1.8229 | 434,083.92 | 428,554.47 | 251.6665 | 222.1337 | NA | NA |
| AutoFitV3Max | autofit | 66 | 1.8620 | 400,776.58 | 395,551.54 | 113.9118 | 113.9118 | 0.108319 | 0.108319 |
| ExtraTrees | ml_tabular | 12 | 1.8633 | 507,041.30 | 507,041.30 | 113.9118 | 113.9118 | 0.065405 | 0.065405 |
| TSMixerx | transformer_sota | 92 | 1.8707 | 467,043.39 | 388,174.02 | 121.8603 | 120.3560 | 0.089735 | 0.086818 |
| AutoFitV6 | autofit | 66 | 2.0091 | 400,776.58 | 395,551.54 | 113.9118 | 113.9118 | 0.094999 | 0.094999 |
| XGBoost | ml_tabular | 12 | 2.2783 | 469,788.23 | 469,788.23 | 170.6267 | 170.6267 | 0.096309 | 0.096309 |
| AutoFitV3E | autofit | 43 | 2.3670 | 434,217.38 | 429,839.00 | 1,062.90 | 203.3371 | NA | NA |
| AutoFitV4 | autofit | 66 | 2.3954 | 431,376.91 | 426,969.13 | 224.6417 | 192.7781 | 0.088906 | 0.088906 |
| VanillaTransformer | transformer_sota | 95 | 2.5396 | 612,877.37 | 606,802.76 | 207.9468 | 207.3179 | 0.035250 | 0.034459 |
| TimesFM | foundation | 102 | 2.5711 | 447,448.91 | 446,608.10 | 230.3774 | 230.3774 | 0.108635 | 0.106073 |
| Moirai | foundation | 104 | 2.5842 | 447,415.61 | 446,570.97 | 228.7647 | 228.7647 | 0.108027 | 0.105348 |
| Moirai2 | foundation | 104 | 2.5842 | 447,415.61 | 446,570.97 | 228.7647 | 228.7647 | 0.108027 | 0.105348 |
| MoiraiLarge | foundation | 104 | 2.5843 | 447,415.61 | 446,570.64 | 228.7647 | 228.7647 | 0.108027 | 0.105348 |
| AutoFitV5 | autofit | 66 | 2.6167 | 400,776.58 | 395,551.54 | 286.7767 | 252.9940 | 0.098954 | 0.098954 |
| LightGBM | ml_tabular | 12 | 2.7208 | 480,601.66 | 480,585.70 | 254.8569 | 254.8569 | 0.096803 | 0.096803 |
| KNN | ml_tabular | 12 | 2.9702 | 588,341.68 | 588,341.68 | 245.3256 | 245.3256 | 0.108702 | 0.108702 |
| HistGradientBoosting | ml_tabular | 12 | 3.0154 | 500,904.52 | 500,904.52 | 313.1252 | 313.1252 | 0.096681 | 0.096681 |
| AutoFitV7 | autofit | 66 | 3.0868 | 397,244.48 | 396,903.77 | 255.4266 | 252.9390 | 0.098933 | 0.090683 |
| MSTL | statistical | 28 | 3.1044 | 4,141,202 | 4,132,860 | 51.0362 | 48.2222 | 0.051138 | 0.049817 |
| SVR | ml_tabular | 10 | 3.4275 | 679,120.78 | 679,120.78 | 294.6342 | 294.6342 | NA | NA |
| SF_SeasonalNaive | statistical | 28 | 3.6152 | 4,082,584 | 4,064,931 | 74.9633 | 74.9084 | 0.046296 | 0.046009 |
| AutoETS | statistical | 28 | 3.6306 | 4,091,934 | 4,080,068 | 75.5006 | 75.1120 | 0.046153 | 0.046152 |
| AutoARIMA | statistical | 28 | 3.6792 | 4,179,370 | 4,133,427 | 76.0673 | 75.1758 | 0.046675 | 0.046316 |
| GRU-D | irregular | 104 | 3.6915 | 813,692.17 | 810,288.88 | 289.5783 | 279.5032 | 0.116600 | 0.114546 |
| SAITS | irregular | 104 | 3.7853 | 813,709.80 | 810,306.50 | 299.1408 | 289.9621 | 0.121346 | 0.117540 |
| AutoTheta | statistical | 28 | 3.7936 | 4,389,101 | 4,161,478 | 76.5425 | 75.1907 | 0.049973 | 0.046435 |
| CatBoost | ml_tabular | 12 | 3.9639 | 602,553.25 | 602,553.25 | 485.1359 | 485.1359 | 0.105205 | 0.105205 |
| QuantileRegressor | ml_tabular | 10 | 4.3330 | 1,091,025 | 1,091,025 | 293.1105 | 293.1105 | NA | NA |
| LogisticRegression | ml_tabular | 2 | 4.6539 | NA | NA | NA | NA | 0.155394 | 0.152627 |
| MeanPredictor | ml_tabular | 12 | 7.3394 | 2,133,181 | 2,133,181 | 484.4789 | 484.4789 | 0.180358 | 0.180358 |
| StemGNN | transformer_sota | 92 | 7.4671 | 2,133,181 | 2,133,181 | 484.4789 | 484.4789 | 0.180358 | 0.180358 |
| TimeXer | transformer_sota | 92 | 7.4671 | 2,133,181 | 2,133,181 | 484.4789 | 484.4789 | 0.180358 | 0.180358 |
| SeasonalNaive | ml_tabular | 12 | 18.2952 | 31,034,759 | 31,034,759 | 381.1212 | 381.1212 | 0.097650 | 0.097650 |
| ElasticNet | ml_tabular | 10 | 19.1181 | 14,792,101 | 14,792,101 | 420.8635 | 420.8635 | NA | NA |
| Lasso | ml_tabular | 10 | 20.5753 | 15,214,820 | 15,214,820 | 473.9220 | 473.9220 | NA | NA |
| Ridge | ml_tabular | 10 | 1482.7597 | 9,792,314,878 | 2,088,461,179 | 4,753.83 | 4,753.83 | NA | NA |

## 4. AutoFit V1-V7.1 Snapshot

### 4.1 Canonical AutoFit family (V1-V7)

| Model | Keys | Normalized ratio | Median funding MAE | Best funding MAE | Median investors MAE | Best investors MAE | Median is_funded MAE | Best is_funded MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AutoFitV1 | 43 | 1.4835 | 437,068.19 | 436,701.55 | 113.9118 | 113.9118 | NA | NA |
| AutoFitV2 | 43 | 1.8229 | 434,083.92 | 428,554.47 | 251.6665 | 222.1337 | NA | NA |
| AutoFitV2E | 43 | 1.8229 | 434,083.92 | 428,554.10 | 251.6665 | 222.1337 | NA | NA |
| AutoFitV3 | 43 | 1.4378 | 400,776.58 | 395,551.54 | 113.9118 | 113.9118 | NA | NA |
| AutoFitV3E | 43 | 2.3670 | 434,217.38 | 429,839.00 | 1,062.90 | 203.3371 | NA | NA |
| AutoFitV3Max | 66 | 1.8620 | 400,776.58 | 395,551.54 | 113.9118 | 113.9118 | 0.108319 | 0.108319 |
| AutoFitV4 | 66 | 2.3954 | 431,376.91 | 426,969.13 | 224.6417 | 192.7781 | 0.088906 | 0.088906 |
| AutoFitV5 | 66 | 2.6167 | 400,776.58 | 395,551.54 | 286.7767 | 252.9940 | 0.098954 | 0.098954 |
| AutoFitV6 | 66 | 2.0091 | 400,776.58 | 395,551.54 | 113.9118 | 113.9118 | 0.094999 | 0.094999 |
| AutoFitV7 | 66 | 3.0868 | 397,244.48 | 396,903.77 | 255.4266 | 252.9390 | 0.098933 | 0.090683 |

### 4.2 V7.1 (g02) quality and fair-comparison against V7

| V7.1 run | Materialized rows | Fairness pass | Min coverage | Max fallback fraction | Overlap with V7 | Win rate vs V7 | Median relative gain vs V7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pilot g02 | 103 | 103/103 | 1.0000 | 0.0000 | 66 | 30.30% | -0.95% |
| full g02 | 56 | 56/56 | 1.0000 | 0.0000 | 46 | 32.61% | -0.95% |

| Target | Pilot g02 win rate vs V7 | Pilot median gain | Full g02 win rate vs V7 | Full median gain |
|---|---:|---:|---:|---:|
| funding_raised_usd | 28.57% | -0.95% | 33.33% | -0.95% |
| investors_count | 14.29% | -27.91% | 0.00% | -18.62% |
| is_funded | 80.00% | 7.32% | 100.00% | 12.65% |

### 4.3 V7.1 anomaly rows (investors_count)

| Run | MAE | Task | Ablation | Horizon |
|---|---:|---|---|---:|
| pilot_g02 | 24,425,199 | task1_outcome | core_edgar | 14 |
| pilot_g02 | 24,425,199 | task1_outcome | core_edgar | 7 |
| pilot_g02 | 24,425,199 | task1_outcome | core_edgar | 1 |
| pilot_g02 | 24,425,199 | task1_outcome | core_edgar | 30 |

## 5. Failed Jobs and Resource Remediation

- Failed canonical jobs (latest attempt):
  - `p7_af1_t1_ce`
  - `p7_af2_t1_ce`
- Resubmitted with larger resources:
  - `p7r_af1_t1_ce` -> JobID `5182721`
  - `p7r_af2_t1_ce` -> JobID `5182722`
- Recommended remediation for these failures:
  - Increase wall time from `2-00:00:00` to `4-00:00:00` for heavy AutoFit shards.
  - Increase memory from `112G` to `160G` for the specific failed `core_edgar` AutoFit shards.
  - Keep full-data setting unchanged (no downsampling, no reduced horizon set).
  - Keep same task/ablation/model roster to preserve fairness.

## 6. V7.2 Upgrade Directions from Current Evidence

1. Add hard target-domain constraints in V7.2 for count targets (`investors_count`): enforce non-negativity and robust clipping after inverse-transform to prevent catastrophic MAE spikes.
2. Add outlier-safe blend guard: reject blend/meta predictions when OOF distribution shifts beyond calibrated bounds, then fallback to best single candidate on OOF.
3. Split objective selection more aggressively by target family: count -> Poisson/Tweedie-first stack; binary -> calibrated logistic stack; heavy-tail -> Huber/quantile-first stack.
4. Introduce a two-stage gate for V7.2 admission: (a) fairness/coverage audit pass, (b) per-target robust win-rate threshold against V7 before full submission.
5. For hyperparameter exploration, retain full data but prune only redundant variant grid once one variant is dominated, to free batch slots for final fairness-complete runs.
