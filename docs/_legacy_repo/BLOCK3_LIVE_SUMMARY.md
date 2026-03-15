# Block 3 Live Summary

- Generated at: **2026-03-08 UTC**
- Freeze stamp: `20260203_225620`
- Canonical benchmark dir: `runs/benchmarks/block3_phase9_fair`
- Phase: **9 (Fair Benchmark)** — Phase 7/8 results are DEPRECATED

## 1. Phase 9 Overview

- Valid records: **5,083**
- Models with valid results: **50**
- Categories represented: **6** (autofit, deep_classical, foundation, irregular, statistical, transformer_sota)
- Pending: ml_tabular, tslib_sota, new statistical/irregular models

### 1.1 Best-in-target snapshots

| Target | Best model | Category | MAE | Task | Ablation | Horizon |
|---|---|---|---:|---|---|---:|
| funding_raised_usd | NHITS | deep_classical | 374,432.36 | task3_risk_adjust | full | 7 |
| investors_count | AutoFitV736 | autofit | 44.66 | task1_outcome | core_text | 7 |
| is_funded | PatchTST | transformer_sota | 0.032281 | task1_outcome | full | 14 |

### 1.2 Top-20 models by normalized MAE ratio

Geometric mean of `MAE_model / MAE_best_on_same_key` across each model's conditions. Lower is better.

| Rank | Model | Category | Conditions | Norm. Ratio |
|---:|---|---|---:|---:|
| 1 | NHITS | deep_classical | 104 | 1.0018 |
| 2 | PatchTST | transformer_sota | 99 | 1.0020 |
| 3 | NBEATSx | transformer_sota | 99 | 1.0037 |
| 4 | NBEATS | deep_classical | 104 | 1.0042 |
| 5 | ChronosBolt | foundation | 104 | 1.0075 |
| 6 | AutoFitV735 | autofit | 104 | 1.0104 |
| 7 | AutoFitV734 | autofit | 104 | 1.0104 |
| 8 | MLP | deep_classical | 104 | 1.0141 |
| 9 | TimesNet | transformer_sota | 99 | 1.0158 |
| 10 | KAN | transformer_sota | 104 | 1.0163 |
| 11 | NLinear | transformer_sota | 104 | 1.0187 |
| 12 | Chronos | foundation | 104 | 1.0199 |
| 13 | TCN | deep_classical | 104 | 1.0217 |
| 14 | DLinear | transformer_sota | 104 | 1.0219 |
| 15 | GRU | deep_classical | 104 | 1.0233 |
| 16 | DeepNPTS | transformer_sota | 104 | 1.0242 |
| 17 | Informer | transformer_sota | 99 | 1.0249 |
| 18 | AutoFitV736 | autofit | 39 | 1.0254 |
| 19 | LSTM | deep_classical | 104 | 1.0255 |
| 20 | DilatedRNN | deep_classical | 104 | 1.0263 |

## 2. Phase 9 Re-Run Status

66 SLURM scripts prepared in `.slurm_scripts/phase9/` targeting ~50 models that need
re-running due to bug fixes (TSLib per-entity, foundation prediction_length, Moirai entity
cap, ml_tabular single-horizon) or first-time execution (new statistical, irregular models).

## 3. AutoFit Status

Only V734, V735, V736 are active. All prior versions (V1-V7, V71-V733, FusedChampion) are dropped.

| Model | Conditions | Norm. Ratio | Status |
|---|---:|---:|---|
| AutoFitV735 | 104 | 1.0104 | COMPLETE |
| AutoFitV734 | 104 | 1.0104 | COMPLETE |
| AutoFitV736 | 39 | 1.0254 | Partial — re-run needed |
