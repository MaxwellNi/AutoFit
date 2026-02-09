# Block 3 Benchmark Results — KDD'26

*Generated: 2026-02-09 12:26 UTC*

**32 models** evaluated across **3 targets** × **4 horizons** × **3 tasks** = 1564 metric records.

---

## Table 1: Main Leaderboard (MAE, per-target)

Best ablation selected per model. All horizons averaged.


### funding_raised_usd (USD)

| Rank | Model | Category | Ablation | MAE | RMSE | SMAPE |
|------|-------|----------|----------|-----|------|-------|
| 1 | **XGBoost** | ml_tabular | core_only | 469,788 | 2,072,097 | 76.59 |
| 2 | **LightGBM** | ml_tabular | core_only | 480,602 | 2,215,767 | 76.71 |
| 3 | **HistGradientBoosting** | ml_tabular | core_only | 500,905 | 2,316,877 | 79.57 |
| 4 | CatBoost | ml_tabular | core_only | 602,553 | 2,627,687 | 90.79 |
| 5 | Moirai | foundation | core_only | 691,983 | 2,359,979 | 107.51 |
| 6 | SAITS | irregular | core_only | 1,079,209 | 2,413,370 | 127.57 |
| 7 | GRU-D | irregular | core_only | 1,079,209 | 2,413,371 | 127.56 |
| 8 | FEDformer | transformer_sota | core_only | 5,868,134 | 6,145,661 | 168.77 |
| 9 | SOFTS | transformer_sota | core_only | 7,547,293 | 7,771,733 | 173.46 |
| 10 | iTransformer | transformer_sota | core_only | 7,653,245 | 7,876,826 | 173.63 |
| 11 | BiTCN | transformer_sota | core_only | 7,722,817 | 7,943,994 | 173.85 |
| 12 | Chronos | foundation | core_only | 7,734,267 | 7,955,136 | 173.86 |
| 13 | RMoK | transformer_sota | core_edgar | 7,767,216 | 7,987,929 | 173.91 |
| 14 | DeepAR | deep_classical | core_edgar | 7,784,553 | 8,004,623 | 173.98 |
| 15 | NBEATS | deep_classical | core_only | 7,800,736 | 8,020,746 | 174.00 |
| 16 | NBEATSx | transformer_sota | core_only | 7,800,736 | 8,020,746 | 174.00 |
| 17 | NHITS | deep_classical | core_only | 7,868,058 | 8,086,958 | 174.14 |
| 18 | KAN | transformer_sota | core_only | 7,909,351 | 8,127,483 | 174.23 |
| 19 | TFT | deep_classical | core_edgar | 7,917,089 | 8,135,120 | 174.25 |
| 20 | StemGNN | transformer_sota | core_edgar | 7,981,564 | 8,198,415 | 174.39 |
| 21 | PatchTST | transformer_sota | core_only | 8,117,369 | 8,333,168 | 174.61 |
| 22 | Informer | transformer_sota | core_only | 8,218,217 | 8,432,078 | 174.83 |
| 23 | TiDE | transformer_sota | core_only | 8,466,736 | 8,677,994 | 175.28 |
| 24 | TSMixer | transformer_sota | core_only | 10,101,260 | 10,304,817 | 177.61 |
| 25 | TimesNet | transformer_sota | core_only | 10,134,463 | 10,338,783 | 177.58 |
| 26 | SF_SeasonalNaive | statistical | core_edgar | 12,148,596 | 12,266,851 | 182.39 |
| 27 | AutoETS | statistical | core_only | 12,248,328 | 12,366,054 | 182.49 |
| 28 | AutoARIMA | statistical | core_edgar | 12,260,770 | 12,378,369 | 182.51 |
| 29 | MSTL | statistical | core_only | 12,523,453 | 12,639,846 | 182.77 |
| 30 | AutoTheta | statistical | core_edgar | 12,918,872 | 13,033,833 | 183.14 |
| 31 | Autoformer | transformer_sota | core_edgar | 14,197,299 | 14,406,871 | 181.67 |

### investors_count (count)

| Rank | Model | Category | Ablation | MAE | RMSE | SMAPE |
|------|-------|----------|----------|-----|------|-------|
| 1 | **XGBoost** | ml_tabular | core_only | 197.2 | 1,188.7 | 62.55 |
| 2 | **Moirai** | foundation | core_only | 306.5 | 1,404.8 | 87.39 |
| 3 | **GRU-D** | irregular | core_only | 311.3 | 1,421.3 | 88.84 |
| 4 | SAITS | irregular | core_edgar | 311.4 | 1,421.3 | 88.85 |
| 5 | HistGradientBoosting | ml_tabular | core_only | 313.1 | 1,250.1 | 90.73 |
| 6 | StemGNN | transformer_sota | core_only | 397.9 | 1,395.2 | 104.93 |
| 7 | FEDformer | transformer_sota | core_edgar | 399.3 | 1,395.3 | 105.14 |
| 8 | iTransformer | transformer_sota | core_only | 400.1 | 1,395.3 | 105.23 |
| 9 | BiTCN | transformer_sota | core_only | 400.3 | 1,395.3 | 105.25 |
| 10 | TiDE | transformer_sota | core_only | 400.5 | 1,395.3 | 105.28 |
| 11 | RMoK | transformer_sota | core_only | 400.8 | 1,395.3 | 105.33 |
| 12 | Chronos | foundation | core_edgar | 401.0 | 1,395.4 | 105.32 |
| 13 | PatchTST | transformer_sota | core_only | 401.1 | 1,395.4 | 105.37 |
| 14 | SOFTS | transformer_sota | core_only | 401.1 | 1,395.4 | 105.37 |
| 15 | TSMixer | transformer_sota | core_only | 401.6 | 1,395.4 | 105.44 |
| 16 | DeepAR | deep_classical | core_edgar | 401.6 | 1,395.4 | 105.44 |
| 17 | NHITS | deep_classical | core_only | 403.0 | 1,395.4 | 105.62 |
| 18 | NBEATS | deep_classical | core_edgar | 403.0 | 1,395.4 | 105.62 |
| 19 | NBEATSx | transformer_sota | core_edgar | 403.0 | 1,395.4 | 105.62 |
| 20 | TFT | deep_classical | core_only | 403.2 | 1,395.5 | 105.65 |
| 21 | TimesNet | transformer_sota | core_only | 403.3 | 1,395.5 | 105.66 |
| 22 | VanillaTransformer | transformer_sota | core_only | 403.3 | 1,395.5 | 105.66 |
| 23 | Informer | transformer_sota | core_only | 403.5 | 1,395.5 | 105.69 |
| 24 | KAN | transformer_sota | core_only | 404.5 | 1,395.5 | 105.83 |
| 25 | Autoformer | transformer_sota | core_edgar | 407.6 | 1,395.7 | 106.26 |
| 26 | SF_SeasonalNaive | statistical | core_only | 422.3 | 1,397.0 | 108.88 |
| 27 | MSTL | statistical | core_edgar | 425.2 | 1,397.2 | 109.23 |
| 28 | AutoETS | statistical | core_only | 429.2 | 1,397.6 | 109.72 |
| 29 | AutoTheta | statistical | core_only | 429.9 | 1,397.6 | 109.81 |
| 30 | AutoARIMA | statistical | core_edgar | 431.5 | 1,397.8 | 110.00 |
| 31 | LightGBM | ml_tabular | core_only | 482.7 | 1,397.9 | 115.80 |
| 32 | CatBoost | ml_tabular | core_only | 485.1 | 1,404.3 | 116.03 |

### is_funded (binary)

| Rank | Model | Category | Ablation | MAE | RMSE | SMAPE |
|------|-------|----------|----------|-----|------|-------|
| 1 | **TFT** | deep_classical | core_edgar | 0.1626 | 0.2956 | 27.24 |
| 2 | **iTransformer** | transformer_sota | core_edgar | 0.1690 | 0.2953 | 28.03 |
| 3 | **VanillaTransformer** | transformer_sota | core_edgar | 0.1711 | 0.2953 | 28.30 |
| 4 | SOFTS | transformer_sota | core_edgar | 0.1713 | 0.2953 | 28.32 |
| 5 | KAN | transformer_sota | core_only | 0.1713 | 0.2953 | 28.32 |
| 6 | NHITS | deep_classical | core_edgar | 0.1714 | 0.2953 | 28.33 |
| 7 | PatchTST | transformer_sota | core_edgar | 0.1714 | 0.2953 | 28.34 |
| 8 | TSMixer | transformer_sota | core_only | 0.1715 | 0.2953 | 28.35 |
| 9 | StemGNN | transformer_sota | core_edgar | 0.1717 | 0.2953 | 28.37 |
| 10 | Autoformer | transformer_sota | core_only | 0.1718 | 0.2954 | 28.38 |
| 11 | TiDE | transformer_sota | core_only | 0.1721 | 0.2953 | 28.42 |
| 12 | RMoK | transformer_sota | core_only | 0.1724 | 0.2953 | 28.46 |
| 13 | NBEATS | deep_classical | core_only | 0.1724 | 0.2952 | 28.46 |
| 14 | NBEATSx | transformer_sota | core_edgar | 0.1724 | 0.2952 | 28.46 |
| 15 | AutoTheta | statistical | core_only | 0.1728 | 0.2964 | 28.36 |
| 16 | MSTL | statistical | core_only | 0.1742 | 0.2964 | 28.52 |
| 17 | TimesNet | transformer_sota | core_edgar | 0.1743 | 0.2952 | 28.69 |
| 18 | Chronos | foundation | core_edgar | 0.1745 | 0.2952 | 28.64 |
| 19 | Informer | transformer_sota | core_edgar | 0.1760 | 0.2953 | 28.90 |
| 20 | SF_SeasonalNaive | statistical | core_edgar | 0.1774 | 0.2964 | 28.93 |
| 21 | AutoARIMA | statistical | core_only | 0.1777 | 0.2964 | 28.96 |
| 22 | AutoETS | statistical | core_only | 0.1777 | 0.2964 | 28.96 |
| 23 | DeepAR | deep_classical | core_only | 0.1803 | 0.2953 | 29.45 |
| 24 | BiTCN | transformer_sota | core_only | 0.1865 | 0.2958 | 30.23 |
| 25 | FEDformer | transformer_sota | core_edgar | 0.2373 | 0.3107 | 37.04 |
| 26 | SAITS | irregular | core_only | 0.3074 | 0.3530 | 47.38 |
| 27 | Moirai | foundation | core_edgar | 0.3624 | 0.3769 | 55.08 |
| 28 | GRU-D | irregular | core_edgar | 0.4214 | 0.4358 | 66.39 |

---

## Table 2: Horizon Breakdown — funding_raised_usd (Top 10)

| Model | h=1 | h=7 | h=14 | h=30 | Avg |
|-------|-----|-----|------|------|-----|
| XGBoost | 469,788 | 469,788 | 469,788 | 469,788 | 469,788 |
| LightGBM | 480,602 | 480,602 | 480,602 | 480,602 | 480,602 |
| HistGradientBoosting | 500,905 | 500,905 | 500,905 | 500,905 | 500,905 |
| CatBoost | 602,553 | 602,553 | 602,553 | 602,553 | 602,553 |
| Moirai | 691,983 | 691,983 | 691,983 | 691,983 | 691,983 |
| SAITS | 1,079,209 | 1,079,209 | 1,079,209 | 1,079,208 | 1,079,209 |
| GRU-D | 1,079,210 | 1,079,208 | 1,079,210 | 1,079,209 | 1,079,209 |
| FEDformer | 5,852,860 | 4,616,452 | 5,973,542 | 7,029,681 | 5,868,134 |
| iTransformer | 6,625,097 | 6,889,195 | 8,000,469 | 7,125,348 | 7,160,027 |
| SOFTS | 7,255,272 | 7,647,989 | 7,867,929 | 7,417,981 | 7,547,293 |


## Table 3: Horizon Breakdown — investors_count (Top 10)

| Model | h=1 | h=7 | h=14 | h=30 | Avg |
|-------|-----|-----|------|------|-----|
| XGBoost | 197.2 | — | — | — | 197.2 |
| Moirai | 306.5 | 306.5 | 306.5 | 306.5 | 306.5 |
| GRU-D | 311.0 | 312.3 | 311.0 | 310.9 | 311.3 |
| SAITS | 311.5 | 311.2 | 311.5 | 311.4 | 311.4 |
| HistGradientBoosting | 313.1 | — | — | — | 313.1 |
| StemGNN | 399.2 | 400.0 | 396.3 | 396.1 | 397.9 |
| iTransformer | 399.5 | 399.4 | 399.0 | 398.4 | 399.1 |
| FEDformer | 399.5 | 400.3 | 398.0 | 399.2 | 399.3 |
| BiTCN | 401.1 | 401.6 | 396.1 | 402.3 | 400.3 |
| TiDE | 400.7 | 400.9 | 400.2 | 400.1 | 400.5 |

---

## Table 4: Category Champions (Best Model per Category)

| Category | Champion | MAE (fund.) | MAE (inv.) | MAE (is_f) | Overall Rank |
|----------|----------|-------------|------------|------------|-------------|
| foundation | **Moirai** | 691,983 | 306.5 | 0.3624 | #2 |
| irregular | **SAITS** | 1,079,209 | 311.4 | 0.3074 | #7 |
| ml_tabular | **XGBoost** | 469,788 | 197.2 | — | #3 |
| deep_classical | **DeepAR** | 7,784,553 | 401.6 | 0.1803 | #14 |
| transformer_sota | **VanillaTransformer** | — | 403.3 | 0.1711 | #1 |
| statistical | **SF_SeasonalNaive** | 12,148,596 | 422.3 | 0.1774 | #27 |

---

## Table 5: Normalized Overall Ranking

Per-target MAE normalized by best model (1.00 = best). Overall = geometric mean of per-target ratios.

| Rank | Model | Category | fund_ratio | inv_ratio | isf_ratio | GeoMean |
|------|-------|----------|------------|-----------|-----------|---------|
| 1 | **Moirai** | foundation | 1.00 | 1.00 | 2.23 | 1.306 |
| 2 | **SAITS** | irregular | 1.56 | 1.02 | 1.89 | 1.442 |
| 3 | **GRU-D** | irregular | 1.56 | 1.02 | 2.59 | 1.601 |
| 4 | SOFTS | transformer_sota | 10.91 | 1.31 | 1.05 | 2.468 |
| 5 | TFT | deep_classical | 11.44 | 1.32 | 1.00 | 2.469 |
| 6 | RMoK | transformer_sota | 11.22 | 1.31 | 1.06 | 2.497 |
| 7 | Chronos | foundation | 11.18 | 1.31 | 1.07 | 2.504 |
| 8 | NBEATS | deep_classical | 11.27 | 1.31 | 1.06 | 2.505 |
| 9 | NBEATSx | transformer_sota | 11.27 | 1.31 | 1.06 | 2.505 |
| 10 | NHITS | deep_classical | 11.37 | 1.31 | 1.05 | 2.507 |
| 11 | StemGNN | transformer_sota | 11.53 | 1.30 | 1.06 | 2.510 |
| 12 | KAN | transformer_sota | 11.43 | 1.32 | 1.05 | 2.514 |
| 13 | FEDformer | transformer_sota | 8.48 | 1.30 | 1.46 | 2.526 |
| 14 | PatchTST | transformer_sota | 11.73 | 1.31 | 1.05 | 2.530 |
| 15 | DeepAR | deep_classical | 11.25 | 1.31 | 1.11 | 2.538 |
| 16 | iTransformer | transformer_sota | 12.06 | 1.31 | 1.04 | 2.541 |
| 17 | BiTCN | transformer_sota | 11.16 | 1.31 | 1.15 | 2.557 |
| 18 | TiDE | transformer_sota | 12.24 | 1.31 | 1.06 | 2.567 |
| 19 | Informer | transformer_sota | 11.88 | 1.32 | 1.08 | 2.568 |
| 20 | TSMixer | transformer_sota | 14.60 | 1.31 | 1.06 | 2.723 |
| 21 | TimesNet | transformer_sota | 14.65 | 1.32 | 1.07 | 2.744 |
| 22 | SF_SeasonalNaive | statistical | 17.56 | 1.38 | 1.09 | 2.978 |
| 23 | MSTL | statistical | 18.10 | 1.39 | 1.07 | 2.996 |
| 24 | AutoETS | statistical | 17.70 | 1.40 | 1.09 | 3.003 |
| 25 | AutoARIMA | statistical | 17.72 | 1.41 | 1.09 | 3.010 |
| 26 | AutoTheta | statistical | 18.67 | 1.40 | 1.06 | 3.031 |
| 27 | Autoformer | transformer_sota | 20.52 | 1.33 | 1.06 | 3.066 |

---

## Table 6: Computational Cost (Avg per target-horizon combo)

| Model | Category | Train Time (s) | Inference Time (s) | Total (s) |
|-------|----------|----------------|-------------------|-----------|
| SF_SeasonalNaive | statistical | 0.4 | 0.1 | 0.5 |
| AutoETS | statistical | 1.1 | 0.1 | 1.2 |
| Moirai | foundation | 1.2 | 0.1 | 1.3 |
| AutoTheta | statistical | 1.5 | 0.1 | 1.5 |
| Chronos | foundation | 1.6 | 1.0 | 2.6 |
| SAITS | irregular | 2.6 | 0.1 | 2.7 |
| AutoARIMA | statistical | 6.7 | 0.1 | 6.8 |
| KAN | transformer_sota | 8.0 | 0.2 | 8.2 |
| TiDE | transformer_sota | 8.7 | 0.2 | 8.9 |
| NBEATSx | transformer_sota | 8.9 | 0.2 | 9.1 |
| NBEATS | deep_classical | 9.8 | 0.2 | 10.0 |
| NHITS | deep_classical | 10.1 | 0.2 | 10.2 |
| GRU-D | irregular | 11.6 | 0.1 | 11.7 |
| MSTL | statistical | 12.6 | 0.1 | 12.7 |
| TSMixer | transformer_sota | 13.0 | 0.2 | 13.1 |
| HistGradientBoosting | ml_tabular | 12.6 | 1.6 | 14.2 |
| RMoK | transformer_sota | 14.5 | 0.2 | 14.7 |
| BiTCN | transformer_sota | 17.8 | 0.2 | 18.0 |
| DeepAR | deep_classical | 21.9 | 0.2 | 22.1 |
| LightGBM | ml_tabular | 14.3 | 10.9 | 25.2 |
| SOFTS | transformer_sota | 25.3 | 0.2 | 25.5 |
| iTransformer | transformer_sota | 25.4 | 0.2 | 25.5 |
| PatchTST | transformer_sota | 33.5 | 0.2 | 33.7 |
| CatBoost | ml_tabular | 41.7 | 0.2 | 41.8 |
| TFT | deep_classical | 56.7 | 0.2 | 56.9 |
| XGBoost | ml_tabular | 56.6 | 4.6 | 61.1 |
| StemGNN | transformer_sota | 107.1 | 0.2 | 107.3 |
| VanillaTransformer | transformer_sota | 115.7 | 0.2 | 115.8 |
| TimesNet | transformer_sota | 157.0 | 0.3 | 157.3 |
| Informer | transformer_sota | 226.8 | 0.2 | 227.0 |
| Autoformer | transformer_sota | 254.5 | 0.2 | 254.7 |
| FEDformer | transformer_sota | 372.4 | 0.2 | 372.6 |

---

## Summary

- **Total models evaluated**: 32
- **Total metric records**: 1564
- **Categories**: deep_classical, foundation, irregular, ml_tabular, statistical, transformer_sota
- **Targets**: funding_raised_usd, investors_count, is_funded
- **Horizons**: 1, 7, 14, 30
- **Tasks**: task1_outcome, task2_forecast, task3_risk_adjust
- **Still running**: ml_tabular (full), autofit (stacking)
- **Freeze stamp**: 20260203_225620