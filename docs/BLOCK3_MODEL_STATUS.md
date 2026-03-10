# Block 3 Model Benchmark Status

> Last updated: 2026-03-10 (Phase 9 + Phase 10 V737)
> Full results: `docs/BLOCK3_RESULTS.md`

## Snapshot

| Metric | Value |
|---|---:|
| Correct evaluation conditions | **104** (48+32+24) |
| metrics.json files | 66 |
| Valid metric records | 5,083 |
| Target model count (Phase 9) | **100** |
| Models with valid results | 50 |
| Active AutoFit versions | V734, V735, V736, **V737** |

## Condition Count Explained

**Correct total: 104 conditions per model**

| Task | Targets | Horizons | Ablations | Conditions |
|---|---:|---:|---:|---:|
| task1_outcome | 3 (total_amount_sold, number_investors, is_funded) | 4 (1,7,14,30) | 4 (core_only, core_text, core_edgar, full) | 48 |
| task2_forecast | 2 (total_amount_sold, number_investors) | 4 | 4 | 32 |
| task3_risk_adjust | 2 (total_amount_sold, number_investors) | 4 | 3 (core_only, core_edgar, full — **no core_text**) | 24 |
| **Total** | | | | **104** |

## Phase Status

### Phase 7/8 (DEPRECATED)
Phase 7 (91 models, 6,670 records) and Phase 8 (99 models, 7,478 records) results are
**invalidated** by 4 critical bug fixes. All prior records are marked deprecated.
See `runs/benchmarks/block3_20260203_225620_phase7/DEPRECATED.md` for details.

### Phase 9 Fair Benchmark (IN PROGRESS)
After fixing TSLib per-entity prediction, foundation prediction_length, Moirai entity cap,
and ml_tabular single-horizon bugs, 5,083 valid records from 50 models were preserved in
`runs/benchmarks/block3_phase9_fair/`. 66 SLURM scripts in `.slurm_scripts/phase9/` target
the remaining ~50 models.

## Models with Valid Phase 9 Results (50)

AutoARIMA, AutoETS, AutoFitV734, AutoFitV735, AutoFitV736, AutoTheta,
Autoformer, BiTCN, Chronos, ChronosBolt, DLinear, DeepAR, DeepNPTS,
DilatedRNN, FEDformer, GRU, GRU-D, Informer, KAN, LSTM, MLP, MOMENT,
MSTL, NBEATS, NBEATSx, NHITS, NLinear, PatchTST, RMoK, SAITS,
SF_SeasonalNaive, SOFTS, StemGNN, Sundial, TCN, TFT, TSMixer, TSMixerx,
TiDE, TimeLLM, TimeMixer, TimeMoE, TimeXer, Timer, TimesFM, TimesFM2,
TimesNet, VanillaTransformer, iTransformer, xLSTM

## Models Requiring Phase 9 Re-Run (~50)

| Category | Models Needing Re-Run | Reason |
|---|---|---|
| foundation (3) | LagLlama, Moirai, Moirai2 | prediction_length bug, entity cap |
| tslib_sota (14) | TimeFilter, WPMixer, MSGNet, etc. | Per-entity prediction bug |
| irregular (2) | BRITS, CSDI | Not yet run |
| ml_tabular (15-18) | All ml_tabular models | Single-horizon bug, NaN passthrough |
| autofit (1) | AutoFitV736 | Incomplete conditions |
| statistical (10) | New models (Croston, Holt, etc.) | Not yet run |

## Bugs Fixed (Phase 9)

1. **TSLib per-entity prediction**: All entities received identical predictions from a single
   mixed-entity input window. Fixed with per-entity batched inference.

2. **Foundation prediction_length=7**: Chronos/ChronosBolt hardcoded `predict(tensors, 7)`.
   Fixed to use actual horizon.

3. **Moirai 50-entity cap**: `ctxs[:50]` limited predictions to 50 entities while Chronos
   processed all. Removed the cap.

4. **ml_tabular single-horizon**: Only `horizons[0]` was evaluated. Fixed to run all horizons.

5. **ml_tabular fillna(0)**: Tree models handle NaN natively. Fixed to pass NaN through for
   tree-based models (XGBoost, LightGBM, CatBoost).

6. **Constant-prediction fairness guard**: Was warning-only with `fairness_pass=True` — constant
   predictions got cached. Fixed to `fairness_pass=False` so they are retried.

## Category Summary

| Category | Phase 9 Target | With Valid Results | Status |
|---|---:|---:|---|
| ml_tabular | 15 | 0 | Needs re-run (single-horizon bug) |
| statistical | 15 | 5 | 10 new models need first run |
| deep_classical | 9 | 9 | VALID (no code changes affected these) |
| transformer_sota | 23 | 23 | VALID |
| foundation | 11 | 10 | LagLlama needs re-run (prediction_length bug) |
| irregular | 4 | 2 | BRITS/CSDI need first run |
| tslib_sota | 20 | 0 | All need re-run (per-entity bug) |
| autofit | 3 | 3 | V734/V735 complete; V736 partial |

## Notes

1. Horizons: {1, 7, 14, 30} days.
2. task3_risk_adjust has only 3 ablations (core_only, core_edgar, full) — NO core_text.
3. V735 oracle table built from Phase 7 data (6,670 records) — will be rebuilt after Phase 9 completes.
4. NF training configs use their original committed values (no max_steps equalization applied).
5. Registry has 128 entries total (V737 added), 101 targeted for Phase 9/10.

## V736 Post-Mortem Analysis (2026-03-10)

V736 NormRank #9/85, raw MAE #27/93. Root causes:

| Root Cause | Impact | Fix (V737) |
|---|---|---|
| EDGAR overfitting | +1.76% avg MAE (core_edgar/full) | Variance filter + PCA → 5 components |
| core_text ≡ core_only | Text ablation dead | Text embeddings pipeline (Job 5225807) |
| funding_raised_usd heavy tail | #27/84 on this target | asinh target transform |

### EDGAR Impact Across AutoFit Versions
| Version | EDGAR Delta (avg MAE) |
|---|---:|
| V734 | +2.47% |
| V735 | +3.12% |
| V736 | +1.76% |
| V737 (expected) | -0.5% to +0.5% |

## Phase 10: V737 EDGAR-Aware Stacking Ensemble

AutoFitV737 = V736 + 2 root-cause fixes:
1. **EDGAR PCA**: 41 raw EDGAR cols → variance filter (≥20% non-zero) → PCA(5)
2. **asinh transform**: `funding_raised_usd` / `funding_goal_usd` heavy tail → `np.arcsinh(y)`

11 SLURM scripts in `.slurm_scripts/phase10/p10_v737_*.sh`
