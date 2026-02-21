# Block3 Local 4090 AutoFitV72 Completion Summary (2026-02-21)

- Generated at (UTC): `2026-02-21T23:00:06.001656+00:00`
- Source run root: `runs/benchmarks/block3_20260203_225620_phase7_v72_4090_20260219_173137`
- Model: `AutoFitV72`

## Overall

- Completed pairs: `20/20` (100.0%)
- Fairness all pass: `True`
- Minimum prediction coverage ratio: `1.0`

## Task Status

| Task | Ablation | Status | Started (UTC) | Finished (UTC) | Completed Pairs | Avg Train Seconds |
|---|---|---|---|---|---:|---:|
| task1_outcome | core_only | completed | 2026-02-21T01:08:22.626514+00:00 | 2026-02-21T19:45:34.085967+00:00 | 12/12 | 8115.57 |
| task2_forecast | core_edgar | completed | 2026-02-21T01:08:22.624360+00:00 | 2026-02-21T12:31:30.222303+00:00 | 8/8 | 9618.46 |

## Per Pair Metrics

| Task | Ablation | Target | Horizon | MAE | RMSE | SMAPE | Coverage | Fairness | Train Sec |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| task1_outcome | core_only | funding_raised_usd | 1 | 401006.306256 | 2230759.737041 | 60.056105 | 1.000 | True | 15187.27 |
| task1_outcome | core_only | funding_raised_usd | 7 | 401006.306256 | 2230759.737041 | 60.056105 | 1.000 | True | 15380.38 |
| task1_outcome | core_only | funding_raised_usd | 14 | 401006.306256 | 2230759.737041 | 60.056105 | 1.000 | True | 7748.52 |
| task1_outcome | core_only | funding_raised_usd | 30 | 401006.306256 | 2230759.737041 | 60.056105 | 1.000 | True | 2252.90 |
| task1_outcome | core_only | investors_count | 1 | 257.790005 | 1366.829589 | 75.989047 | 1.000 | True | 8792.99 |
| task1_outcome | core_only | investors_count | 7 | 257.790005 | 1366.829589 | 75.989047 | 1.000 | True | 9081.06 |
| task1_outcome | core_only | investors_count | 14 | 257.790005 | 1366.829589 | 75.989047 | 1.000 | True | 8270.96 |
| task1_outcome | core_only | investors_count | 30 | 257.790005 | 1366.829589 | 75.989047 | 1.000 | True | 7382.85 |
| task1_outcome | core_only | is_funded | 1 | 0.092537 | 0.218922 | 23.783958 | 1.000 | True | 5829.22 |
| task1_outcome | core_only | is_funded | 7 | 0.092537 | 0.218922 | 23.783958 | 1.000 | True | 5807.03 |
| task1_outcome | core_only | is_funded | 14 | 0.092537 | 0.218922 | 23.783958 | 1.000 | True | 5825.40 |
| task1_outcome | core_only | is_funded | 30 | 0.092537 | 0.218922 | 23.783958 | 1.000 | True | 5828.22 |
| task2_forecast | core_edgar | funding_raised_usd | 1 | 396381.291929 | 2104851.019115 | 58.795004 | 1.000 | True | 18012.55 |
| task2_forecast | core_edgar | funding_raised_usd | 7 | 396381.291929 | 2104851.019115 | 58.795004 | 1.000 | True | 18512.76 |
| task2_forecast | core_edgar | funding_raised_usd | 14 | 396381.291929 | 2104851.019115 | 58.795004 | 1.000 | True | 9345.91 |
| task2_forecast | core_edgar | funding_raised_usd | 30 | 396381.291929 | 2104851.019115 | 58.795004 | 1.000 | True | 2603.18 |
| task2_forecast | core_edgar | investors_count | 1 | 274.408533 | 1385.445286 | 74.564334 | 1.000 | True | 7324.39 |
| task2_forecast | core_edgar | investors_count | 7 | 274.406955 | 1385.431916 | 74.564227 | 1.000 | True | 6768.38 |
| task2_forecast | core_edgar | investors_count | 14 | 274.407323 | 1385.430872 | 74.564110 | 1.000 | True | 7250.54 |
| task2_forecast | core_edgar | investors_count | 30 | 274.407565 | 1385.430906 | 74.564163 | 1.000 | True | 7129.97 |
