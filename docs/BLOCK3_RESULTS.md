# Block 3 Results (Current)

> Last verified: 2026-03-16 14:24 CET (live scan — not estimated)
> Canonical benchmark root: `runs/benchmarks/block3_phase9_fair/`
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`

This file reports only the current clean Phase 9 / V739 benchmark reality.
The previous large static result tables were archived to:
- `docs/_legacy_repo/BLOCK3_RESULTS_table_20260314.md`

## Current Snapshot

The current benchmark surface is verified from:
- `docs/benchmarks/phase9_current_snapshot.md`
- `runs/benchmarks/block3_phase9_fair/all_results.csv`
- `runs/benchmarks/block3_phase9_fair/REPLICATION_MANIFEST.json`

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw metrics files | 132 | direct scan 2026-03-16 14:24 |
| raw records | 13549 | direct scan 2026-03-16 14:24 |
| raw models | 91 | direct scan 2026-03-16 14:24 |
| raw complete models (≥104) | 80 | direct scan 2026-03-16 14:24 |
| raw partial models | 11 | direct scan 2026-03-16 14:24 |
| total unique conditions | 164 | 28 co + 28 ce + 24 cs2 + 28 ces2 + 28 ct + 28 fu |
| universal conditions (ALL 80 complete models) | 72 | 28 co + 28 ce + 10 ct + 6 fu |
| ranking conditions (co+ce) | 56 | 28 core_only + 28 core_edgar |

### Condition Matrix Explained (56 vs 104 vs 160)

- **3 tasks**: task1_outcome (3 targets), task2_forecast (2 targets), task3_risk_adjust (2 targets)
- **4 horizons**: h1, h7, h14, h30
- **Per ablation**: task1=3×4=12, task2=2×4=8, task3=2×4=8 → **28 conditions per ablation**
- **6 ablations**: core_only, core_edgar, core_only_seed2, core_edgar_seed2, core_text, full
- **Full maximum**: 28 × 6 = **168** (but core_only_seed2 × task3 = 0 → capped at 164)
- **≥104**: The minimum records for a model to be considered "complete" (≥ co+ce+cs2+ces2 = 100+)
- **56 ranking conditions**: core_only (28) + core_edgar (28) — the only ablations shared by ALL 80 complete models, used for fair cross-model ranking
- **160**: Maximum records for fully-landed models (28×6 minus 4 missing task3 cs2 = 168−8 ≈ 160 observed)

## Current AutoFit Reality

| Fact | Value | Evidence |
| --- | --- | --- |
| active AutoFit baseline | `AutoFitV739` | Root `AGENTS.md` |
| canonical landed conditions | `112/112` (ALL COMPLETE) | direct scan: 12 metrics.json under `*/autofit/*` |
| ablation breakdown | co=28, ce=28, ct=28, fu=28 | direct scan |
| quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| mean rank (56 universal conditions) | **#13/80** (top 16%) | computed across 56 universal conditions shared by all 80 models |
| mean rank score | 14.38 | lower is better |
| conditions won (champion) | 3/56 (1 per task) | best MAE in that condition |
| top-5 by mean rank | NHITS (#1, 4.21), PatchTST (#2, 4.36), NBEATS (#3, 4.77), NBEATSx (#4, 5.84), ChronosBolt (#5, 7.11) | 56 universal conditions |

## How to Interpret the Current Benchmark Surface

1. The canonical benchmark root is `runs/benchmarks/block3_phase9_fair/`.
2. The current physical Phase 9 result surface has 6 ablations per model:
   - `core_only` (28 conditions — all models have)
   - `core_edgar` (28 conditions — all models have)
   - `core_only_seed2` (24 conditions — most complete models have 20)
   - `core_edgar_seed2` (28 conditions — most complete models have 20+)
   - `core_text` (28 conditions — Phase 12 landed, 72/80 have 28)
   - `full` (28 conditions — Phase 12 landed, 66/80 have 28)
3. Text embedding artifacts fully functional. Phase 12 text reruns: 42/48 COMPLETED, 6 RUNNING (cfisch tslib only).
4. `core_text` now covers **91/91** models, `full` covers **91/91** models (NegBinGLM has partial records, structural failure).
5. Current champion model: **NBEATS** — 24/56 conditions won (43%), dominant across all 3 tasks.
6. Phase 15: 23 new TSLib SOTA models submitted (3 npin RUNNING, 3 npin PENDING, 6 cfisch PENDING). Not yet in benchmark.

## Full Ranking Table — 80 Complete Models (56 co+ce conditions)

> Rank by mean MAE rank across 56 core_only + core_edgar conditions.
> "Won" = number of conditions where this model has the lowest MAE.

| Rank | Model | Category | MeanRank | Won | Records | Status |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | **NHITS** | deep_classical | 4.21 | 9 | 160/160 | COMPLETE |
| 2 | **PatchTST** | transformer_sota | 4.36 | 2 | 160/160 | COMPLETE |
| 3 | **NBEATS** | deep_classical | 4.77 | 24 | 160/160 | COMPLETE |
| 4 | NBEATSx | transformer_sota | 5.84 | 0 | 160/160 | COMPLETE |
| 5 | ChronosBolt | foundation | 7.11 | 0 | 160/160 | COMPLETE |
| 6 | Chronos | foundation | 10.62 | 6 | 160/160 | COMPLETE |
| 7 | TimesNet | transformer_sota | 10.93 | 0 | 160/160 | COMPLETE |
| 8 | KAN | transformer_sota | 11.43 | 5 | 160/160 | COMPLETE |
| 9 | TFT | deep_classical | 12.02 | 0 | 160/160 | COMPLETE |
| 10 | TCN | deep_classical | 13.18 | 0 | 160/160 | COMPLETE |
| 11 | MLP | deep_classical | 13.52 | 0 | 160/160 | COMPLETE |
| 12 | DeepAR | deep_classical | 13.75 | 0 | 160/160 | COMPLETE |
| **13** | **AutoFitV739** | **autofit** | **14.38** | **3** | **112/160** | **DONE** |
| 14 | Informer | transformer_sota | 14.73 | 0 | 160/160 | COMPLETE |
| 15 | GRU | deep_classical | 14.82 | 3 | 160/160 | COMPLETE |
| 16 | LSTM | deep_classical | 14.88 | 0 | 160/160 | COMPLETE |
| 17 | NLinear | transformer_sota | 15.16 | 0 | 160/160 | COMPLETE |
| 18 | DilatedRNN | deep_classical | 17.50 | 0 | 160/160 | COMPLETE |
| 19 | BiTCN | transformer_sota | 18.70 | 0 | 160/160 | COMPLETE |
| 20 | TiDE | transformer_sota | 19.48 | 0 | 160/160 | COMPLETE |
| 21 | DLinear | transformer_sota | 19.82 | 0 | 160/160 | COMPLETE |
| 22 | DeepNPTS | transformer_sota | 20.32 | 4 | 160/160 | COMPLETE |
| 23 | FEDformer | transformer_sota | 27.79 | 0 | 160/160 | COMPLETE |
| 24 | Timer | foundation | 29.43 | 0 | 160/160 | COMPLETE |
| 25 | TimeMoE | foundation | 30.04 | 0 | 160/160 | COMPLETE |
| 26 | Autoformer | transformer_sota | 31.02 | 0 | 160/160 | COMPLETE |
| 27 | Sundial | foundation | 31.05 | 0 | 160/160 | COMPLETE |
| 28 | MOMENT | foundation | 31.14 | 0 | 160/160 | COMPLETE |
| 29 | TimesFM2 | foundation | 32.05 | 0 | 160/160 | COMPLETE |
| 30 | TTM | foundation | 32.05 | 0 | 160/160 | COMPLETE |
| 31 | LagLlama | foundation | 33.05 | 0 | 160/160 | COMPLETE |
| 32 | Moirai | foundation | 34.05 | 0 | 160/160 | COMPLETE |
| 33 | MoiraiLarge | foundation | 35.05 | 0 | 160/160 | COMPLETE |
| 34 | TimeMixer | transformer_sota | 35.88 | 0 | 160/160 | COMPLETE |
| 35 | Moirai2 | foundation | 36.05 | 0 | 160/160 | COMPLETE |
| 36 | Chronos2 | foundation | 37.05 | 0 | 160/160 | COMPLETE |
| 37 | RMoK | transformer_sota | 37.18 | 0 | 160/160 | COMPLETE |
| 38 | TSMixer | transformer_sota | 40.00 | 0 | 160/160 | COMPLETE |
| 39 | iTransformer | transformer_sota | 40.61 | 0 | 160/160 | COMPLETE |
| 40 | SOFTS | transformer_sota | 42.00 | 0 | 160/160 | COMPLETE |
| 41 | RandomForest | ml_tabular | 46.34 | 0 | 157/160 | DONE |
| 42 | MSTL | statistical | 47.45 | 0 | 160/160 | COMPLETE |
| 43 | WindowAverage | statistical | 49.64 | 0 | 160/160 | COMPLETE |
| 44 | SF_SeasonalNaive | statistical | 49.98 | 0 | 160/160 | COMPLETE |
| 45 | LightGBMTweedie | ml_tabular | 50.14 | 0 | 157/160 | DONE |
| 46 | VanillaTransformer | transformer_sota | 50.46 | 0 | 160/160 | COMPLETE |
| 47 | TSMixerx | transformer_sota | 51.04 | 0 | 160/160 | COMPLETE |
| 48 | AutoETS | statistical | 51.80 | 0 | 160/160 | COMPLETE |
| 49 | Naive | statistical | 51.93 | 0 | 160/160 | COMPLETE |
| 50 | ExtraTrees | ml_tabular | 52.00 | 0 | 157/160 | DONE |
| 51 | HistoricAverage | statistical | 52.00 | 0 | 160/160 | COMPLETE |
| 52 | CrostonOptimized | statistical | 52.11 | 0 | 160/160 | COMPLETE |
| 53 | XGBoost | ml_tabular | 52.73 | 0 | 157/160 | DONE |
| 54 | XGBoostPoisson | ml_tabular | 52.75 | 0 | 157/160 | DONE |
| 55 | LightGBM | ml_tabular | 54.34 | 0 | 157/160 | DONE |
| 56 | DynamicOptimizedTheta | statistical | 54.34 | 0 | 160/160 | COMPLETE |
| 57 | TimesFM | foundation | 54.75 | 0 | 160/160 | COMPLETE |
| 58 | CrostonClassic | statistical | 54.82 | 0 | 160/160 | COMPLETE |
| 59 | AutoARIMA | statistical | 55.64 | 0 | 160/160 | COMPLETE |
| 60 | Holt | statistical | 55.93 | 0 | 160/160 | COMPLETE |
| 61 | HistGradientBoosting | ml_tabular | 57.52 | 0 | 157/160 | DONE |
| 62 | AutoTheta | statistical | 58.07 | 0 | 160/160 | COMPLETE |
| 63 | HoltWinters | statistical | 58.18 | 0 | 160/160 | COMPLETE |
| 64 | CrostonSBA | statistical | 58.25 | 0 | 160/160 | COMPLETE |
| 65 | GRU-D | irregular | 59.75 | 0 | 160/160 | COMPLETE |
| 66 | BRITS | irregular | 59.88 | 0 | 160/160 | COMPLETE |
| 67 | SAITS | irregular | 61.21 | 0 | 160/160 | COMPLETE |
| 68 | CSDI | irregular | 64.07 | 0 | 160/160 | COMPLETE |
| 69 | CatBoost | ml_tabular | 64.36 | 0 | 157/160 | DONE |
| 70 | KANAD | tslib_sota | 66.91 | 0 | 121/160 | DONE |
| 71 | AutoCES | statistical | 67.41 | 0 | 160/160 | COMPLETE |
| 72 | FITS | tslib_sota | 67.91 | 0 | 121/160 | DONE |
| 73 | xLSTM | transformer_sota | 68.62 | 0 | 160/160 | COMPLETE |
| 74 | MeanPredictor | ml_tabular | 68.84 | 0 | 157/160 | DONE |
| 75 | CATS | tslib_sota | 68.91 | 0 | 121/160 | DONE |
| 76 | TimeLLM | transformer_sota | 69.62 | 0 | 160/160 | COMPLETE |
| 77 | StemGNN | transformer_sota | 69.84 | 0 | 160/160 | COMPLETE |
| 78 | WPMixer | tslib_sota | 69.91 | 0 | 121/160 | DONE |
| 79 | TimeXer | transformer_sota | 70.84 | 0 | 160/160 | COMPLETE |
| 80 | SeasonalNaive | ml_tabular | 72.68 | 0 | 157/160 | DONE |

## Partial Models (gap-fill in progress)

| Model | Category | Records | CO | CE | CS2 | CES2 | CT | FU | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Crossformer | tslib_sota | 91/160 | 22 | 21 | 16 | 15 | 11 | 6 | GAP-FILL RUNNING |
| ETSformer | tslib_sota | 104/160 | 25 | 25 | 22 | 15 | 11 | 6 | GAP-FILL RUNNING |
| LightTS | tslib_sota | 104/160 | 25 | 25 | 22 | 15 | 11 | 6 | GAP-FILL RUNNING |
| MSGNet | tslib_sota | 91/160 | 22 | 21 | 16 | 15 | 11 | 6 | GAP-FILL RUNNING |
| MambaSimple | tslib_sota | 91/160 | 22 | 21 | 16 | 15 | 11 | 6 | GAP-FILL RUNNING |
| MultiPatchFormer | tslib_sota | 91/160 | 22 | 21 | 16 | 15 | 11 | 6 | GAP-FILL RUNNING |
| NegativeBinomialGLM | ml_tabular | 21/160 | 4 | 4 | 4 | 4 | 4 | 1 | STRUCTURAL FAILURE |
| PAttn | tslib_sota | 91/160 | 22 | 21 | 16 | 15 | 11 | 6 | GAP-FILL RUNNING |
| Pyraformer | tslib_sota | 104/160 | 25 | 25 | 22 | 15 | 11 | 6 | GAP-FILL RUNNING |
| Reformer | tslib_sota | 104/160 | 25 | 25 | 22 | 15 | 11 | 6 | GAP-FILL RUNNING |
| TimeFilter | tslib_sota | 91/160 | 22 | 21 | 16 | 15 | 11 | 6 | GAP-FILL RUNNING |

## Phase 15: 23 New TSLib Models (not yet in benchmark)

Submitted 2026-03-16. All pending/running. No benchmark results yet.

Models: CARD, CFPT, DeformableTST, DUET, FiLM, FilterTS, FreTS, Fredformer,
MICN, ModernTCN, NonstationaryTransformer, PDF, PIR, PathFormer, SCINet, SEMPO,
SRSNet, SegRNN, SparseTSF, TimeBridge, TimePerceiver, TimeRecipe, xPatch

Job status (2026-03-16 14:24 CET):
- npin RUNNING (3): t1_co, t1_ce, t2_co — each ~3h elapsed, ~4-8 models done per job
- npin PENDING (3): t2_ce, t3_co, t3_ce
- cfisch PENDING (6): t1_ct, t1_fu, t2_ct, t2_fu, t3_ct, t3_fu

Known errors in running jobs (old code, pre-fix c4d214e):
- DeformableTST: "No module named timm" (t1_co, t2_co) / "no attribute n_vars" (t1_ce)
- DUET: "no attribute noisy_gating"
- FilterTS: "Invalid filter type"
- PathFormer, SEMPO: not yet reached (will fail on old code)
- **All fixed in code (c4d214e + n_vars fix). PENDING jobs will use fixed code.**
- **Targeted rerun needed for 5 errored models × 3 old-code conditions after current jobs finish.**

## Champion Summary by Task

| Task | #1 Champion | Other Winners |
| --- | --- | --- |
| task1_outcome (24 conditions) | NBEATS (8/24) | NHITS (5), DeepNPTS (4), Chronos (2), PatchTST (2), KAN (1), GRU (1), AutoFitV739 (1) |
| task2_forecast (16 conditions) | NBEATS (8/16) | Chronos (2), NHITS (2), KAN (2), GRU (1), AutoFitV739 (1) |
| task3_risk_adjust (16 conditions) | NBEATS (8/16) | Chronos (2), NHITS (2), KAN (2), GRU (1), AutoFitV739 (1) |

## Where to Inspect Actual Results

1. Current filtered leaderboard:
   - `runs/benchmarks/block3_phase9_fair/all_results.csv`
2. Current benchmark interpretation manifest:
   - `runs/benchmarks/block3_phase9_fair/REPLICATION_MANIFEST.json`
3. Current fact snapshot:
   - `docs/benchmarks/phase9_current_snapshot.md`
4. Current model status summary:
   - `docs/BLOCK3_MODEL_STATUS.md`

## What Is No Longer Current

The following are preserved for history or reference only and must not be used as current benchmark truth:
- `docs/_legacy_repo/`
- `docs/benchmarks/LEGACY__block3_truth_pack__v72_v73/`
- Phase 7 / Phase 8 results
- V72 / early V73 benchmark narratives
- V734-V738 empirical outputs or design narratives
