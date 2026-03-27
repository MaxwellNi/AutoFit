# Block 3 Full SOTA Benchmark

> Last verified: 2026-03-27 17:10 CET
> Source: `runs/benchmarks/block3_phase9_fair/all_results.csv`
> Scope: **current clean shared leaderboard** over all **non-retired**, **post-filter**, **160/160 complete** models.

## Interpretation Rule

This file is the current benchmark-facing view of the **fully landed clean frontier**.
It is narrower than the raw benchmark scan on purpose.

- The raw benchmark still has **62 active models @160/160** in the direct phase9 snapshot.
- The stricter leaderboard below keeps only models that are:
  - non-retired,
  - present in post-filter `all_results.csv`,
  - fully comparable on the shared **160-cell** evaluation surface.
- Under that stricter comparability rule, the current clean full frontier contains **55 models**.

So the right reading is:
- `62` = raw active complete models from the physical benchmark scan.
- `55` = current **post-filter clean comparable leaderboard** used for like-for-like ranking.

## Current Clean Full Leaderboard

| Rank | Model | Mean Rank | Wins | Shared Conditions |
| --- | --- | ---: | ---: | ---: |
| 1 | PatchTST | 4.275 | 4 | 160 |
| 2 | NHITS | 4.375 | 21 | 160 |
| 3 | NBEATS | 5.412 | 6 | 160 |
| 4 | NBEATSx | 5.412 | 3 | 160 |
| 5 | ChronosBolt | 7.425 | 0 | 160 |
| 6 | KAN | 10.406 | 16 | 160 |
| 7 | Chronos | 10.550 | 23 | 160 |
| 8 | TimesNet | 10.925 | 0 | 160 |
| 9 | TCN | 11.262 | 0 | 160 |
| 10 | MLP | 11.594 | 0 | 160 |
| 11 | TFT | 12.231 | 0 | 160 |
| 12 | GRU | 12.925 | 11 | 160 |
| 13 | LSTM | 13.006 | 0 | 160 |
| 14 | DeepAR | 13.875 | 0 | 160 |
| 15 | NLinear | 14.150 | 0 | 160 |
| 16 | Informer | 14.569 | 0 | 160 |
| 17 | DilatedRNN | 15.719 | 0 | 160 |
| 18 | DeepNPTS | 15.981 | 16 | 160 |
| 19 | BiTCN | 17.531 | 0 | 160 |
| 20 | DLinear | 18.519 | 1 | 160 |
| 21 | TiDE | 18.744 | 0 | 160 |
| 22 | FEDformer | 25.925 | 0 | 160 |
| 23 | Timer | 26.188 | 0 | 160 |
| 24 | Autoformer | 26.300 | 0 | 160 |
| 25 | TimeMixer | 29.019 | 0 | 160 |
| 26 | RMoK | 30.375 | 0 | 160 |
| 27 | iTransformer | 33.750 | 0 | 160 |
| 28 | TSMixer | 34.081 | 0 | 160 |
| 29 | SOFTS | 34.169 | 0 | 160 |
| 30 | WindowAverage | 34.469 | 0 | 160 |
| 31 | SF_SeasonalNaive | 34.819 | 0 | 160 |
| 32 | RandomForest | 35.331 | 0 | 160 |
| 33 | MSTL | 36.562 | 0 | 160 |
| 34 | AutoETS | 36.619 | 0 | 160 |
| 35 | Naive | 36.744 | 0 | 160 |
| 36 | CrostonOptimized | 36.925 | 0 | 160 |
| 37 | CrostonClassic | 38.231 | 0 | 160 |
| 38 | DynamicOptimizedTheta | 39.138 | 0 | 160 |
| 39 | LightGBMTweedie | 39.847 | 0 | 160 |
| 40 | TSMixerx | 39.888 | 0 | 160 |
| 41 | VanillaTransformer | 40.013 | 0 | 160 |
| 42 | ExtraTrees | 40.100 | 0 | 160 |
| 43 | HistoricAverage | 40.125 | 0 | 160 |
| 44 | AutoARIMA | 40.431 | 0 | 160 |
| 45 | Holt | 40.538 | 0 | 160 |
| 46 | CrostonSBA | 41.388 | 0 | 160 |
| 47 | LightGBM | 41.678 | 0 | 160 |
| 48 | TimesFM | 41.981 | 0 | 160 |
| 49 | AutoTheta | 42.519 | 0 | 160 |
| 50 | HoltWinters | 42.619 | 0 | 160 |
| 51 | HistGradientBoosting | 44.394 | 0 | 160 |
| 52 | CatBoost | 46.312 | 0 | 160 |
| 53 | GRU-D | 46.500 | 0 | 160 |
| 54 | BRITS | 46.669 | 0 | 160 |
| 55 | SAITS | 47.469 | 0 | 160 |

## Practical Reading

1. The current clean full frontier is still led by the same core champion families the project has already identified repeatedly: `PatchTST`, `NHITS`, `NBEATS`, `NBEATSx`, `ChronosBolt`, `KAN`, `Chronos`, and `TimesNet`.
2. `NBEATS` remains the most important champion-mechanism source even when its full-shared-slice mean rank is not rank-1 here; its broader raw benchmark champion count still dominates in the project-wide analysis.
3. `AutoFitV739` does **not** appear in this table yet because it is still only `132/160` landed and therefore is not part of the current fully comparable clean frontier.
4. This file should be read together with:
   - `docs/BLOCK3_RESULTS.md`
   - `docs/BLOCK3_MODEL_STATUS.md`
   - `docs/benchmarks/phase9_current_snapshot.md`
   - `docs/references/BLOCK3_CHAMPION_COMPONENT_ANALYSIS.md`
