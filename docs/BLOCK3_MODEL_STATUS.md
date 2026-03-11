# Block 3 Model Benchmark Status

> Last updated: 2026-03-12 (Phase 10 — V738/V737 ALL COMPLETE, oracle 5-layer root cause, tsA TIMEOUT)
> Full results: `docs/BLOCK3_RESULTS.md`

## Snapshot

| Metric | Value |
|---|---:|
| Evaluation conditions per model | **104** (48+32+24) |
| Phase 9 metrics.json files | 88 |
| Phase 9 valid metric records | 8,928 |
| Phase 9 unique models | 93 |
| Phase 9 complete (104/104) | 76 |
| Phase 9 partial (<104) | 10 |
| Phase 10 AutoFit records | V737=104, V738=104 |
| Active SLURM jobs | 2 RUNNING (tsC t1_ce + t3_ce) |
| **AutoFit integrity** | ❌ **ALL V733-V738 have oracle test-set leakage** |

## CRITICAL: 5-Layer Oracle Test-Set Leakage

See `docs/BLOCK3_RESULTS.md` for the complete 5-layer root cause analysis.

**Summary of layers**:
1. **Data source**: Benchmark pipeline outputs ONLY test-set metrics — no validation split exists
2. **Copy-paste perpetuation**: Each version expanded oracle table without auditing data source
3. **Missing validation infrastructure**: No train/val/test 3-way split in benchmark harness
4. **Dead code false confidence**: V738's `val_frac=0.2` set but never used (V753 has working version)
5. **No automated guard**: No test prevents test-set info → fit()

**Oracle tables (ALL from test-set)**:
- `ORACLE_TABLE` (L50): V733, hand-coded, 5,848 records
- `ORACLE_TABLE_V734` (L425): Coarse oracle, avg_rank from test-set
- `ORACLE_TABLE_V735` (L689): Single best model from test-set RMSE
- `ORACLE_TABLE_TOP3` (L751): Top-3 per condition from test-set RMSE → used by V736/V737
- `ORACLE_TABLE_V738` (L1313): Top-5 per condition from test-set MAE → used by V738

**All V734-V738 rankings are scientifically invalid for paper.**

## Condition Count Explained

| Task | Targets | Horizons | Ablations | Conditions |
|---|---:|---:|---:|---:|
| task1_outcome | 3 (funding_raised_usd, investors_count, is_funded) | 4 (1,7,14,30) | 4 (core_only, core_text, core_edgar, full) | 48 |
| task2_forecast | 2 (funding_raised_usd, investors_count) | 4 | 4 | 32 |
| task3_risk_adjust | 2 (funding_raised_usd, investors_count) | 4 | 3 (core_only, core_edgar, full — **no core_text**) | 24 |
| **Total** | | | | **104** |

## Active SLURM Jobs (2026-03-12 ~04:30 UTC)

### RUNNING (2 jobs)

| Job ID | Name | Time | Memory | Node | Progress |
|-------:|------|------|--------|------|----------|
| 5221718 | p10r_tsC_t1_ce | 26h | 320G | iris-180 | Reformer/ETSformer/LightTS/Pyraformer recovery |
| 5221721 | p10r_tsC_t3_ce | 19h | 320G | iris-176 | Reformer/ETSformer/LightTS/Pyraformer recovery |

### RECENTLY COMPLETED

| Batch | Jobs | Elapsed | Result |
|-------|------|---------|--------|
| tsA (5219864-66) | 3 | 2d 0h (TIMEOUT) | MSGNet/PAttn/MambaSimple stuck at 52/104 each |
| tsC (5221719-20,5221722) | 3 | COMPLETED | tsC recovery shards |
| V737 npin (5226208-13) | 6 | 49-71 min | ✅ 104/104 records |
| V738 cfisch (5226229-33) | 5 | 30-75 min | ✅ 104/104 records (oracle-leaked) |
| V737 cfisch (5226239-43) | 5 | 62-115 min | ✅ merged into V737 results |

## Model Completion (76 complete + 17 partial)

### Complete (76 models × 104/104)

| Category | Count | Models |
|---|---:|---|
| deep_classical | 9 | NBEATS, NHITS, TFT, DeepAR, GRU, LSTM, TCN, MLP, DilatedRNN |
| foundation | 12 | Chronos, ChronosBolt, Timer, TimeMoE, MOMENT, TimesFM, Sundial, TimesFM2, LagLlama, Moirai, MoiraiLarge, Moirai2 |
| irregular | 4 | GRU-D, SAITS, BRITS, CSDI |
| ml_tabular | 9 | LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, HistGradientBoosting, MeanPredictor, LightGBMTweedie, XGBoostPoisson |
| statistical | 15 | AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive, AutoCES, CrostonClassic, CrostonOptimized, CrostonSBA, DynamicOptimizedTheta, HistoricAverage, Holt, HoltWinters, Naive, WindowAverage |
| transformer_sota | 23 | PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx, xLSTM, TimeLLM, DeepNPTS, BiTCN, KAN, RMoK, SOFTS, StemGNN, DLinear, NLinear, TimeMixer, TimeXer, TSMixerx |
| tslib_sota | 1 | WPMixer |
| autofit | 3 | AutoFitV734, AutoFitV735, AutoFitV736 |

### Partial (10 models in Phase 9)

| Model | Records | Category | Status |
|---|---:|---|---|
| Chronos2 | 58 | foundation | 🔄 |
| TTM | 58 | foundation | 🔄 |
| MSGNet | 52 | tslib_sota | ❌ tsA TIMEOUT |
| PAttn | 52 | tslib_sota | ❌ tsA TIMEOUT |
| MambaSimple | 52 | tslib_sota | ❌ tsA TIMEOUT |
| Crossformer | 52 | tslib_sota | 🔄 |
| TimeFilter | 52 | tslib_sota | ❌ constant pred |
| MultiPatchFormer | 52 | tslib_sota | ❌ constant pred |
| ETSformer | 50 | tslib_sota | 🔄 tsC running |
| LightTS | 50 | tslib_sota | 🔄 tsC running |
| Pyraformer | 50 | tslib_sota | 🔄 tsC running |
| Reformer | 50 | tslib_sota | 🔄 tsC running |
| NegativeBinomialGLM | 16 | ml_tabular | 🔄 |

### Phase 10 AutoFit (INVALID — oracle leakage)

| Model | Records | V737 vs V738 | Status |
|---|---:|---|---|
| AutoFitV737 | 104/104 | V737 wins 27, V738 wins 77 | ❌ oracle leak (ORACLE_TABLE_TOP3) |
| AutoFitV738 | 104/104 | see above | ❌ oracle leak (ORACLE_TABLE_V738) |

V737 ablation check: core_text==core_only for 16/20, differs for 4 (NF non-determinism).
V737 ablation check: full==core_edgar for 10/28, differs for 18 (NF non-determinism).

## Audit Exclusion List (21+ models)

| Finding | Models | Reason |
|---------|--------|--------|
| A (6) | Sundial, TimesFM2, LagLlama, Moirai, MoiraiLarge, Moirai2 | Context-mean fallback |
| B (5) | AutoCES, xLSTM, TimeLLM, StemGNN, TimeXer | Training crash fallback |
| C (2) | TimeMoE, MOMENT | Near-duplicate of Timer |
| G (3) | MICN, MultiPatchFormer, TimeFilter | 100% constant predictions |
| J (5) | AutoFitV734, AutoFitV735, AutoFitV736, AutoFitV737, AutoFitV738 | Oracle test-set leakage |

## Memory Requirements (empirical)

| Ablation | Actual Peak RSS | SLURM Allocation | Status |
|----------|----------------:|------------------:|--------|
| core_only | ~80-100 GB | 128G | ✅ sufficient |
| core_edgar | ~140-160 GB | 192G | ✅ sufficient |
| core_text | ~200-268 GB | 384G | ✅ (was OOM at 128G, 192G, 256G) |
| full | ~200-268 GB | 384G | ✅ (was OOM at 128G, 192G, 256G) |

## Phase History

| Phase | Models | Records | Status | Notes |
|-------|-------:|--------:|--------|-------|
| Phase 7 | 91 | 6,670 | ❌ DEPRECATED | 4 critical bugs |
| Phase 8 | 99 | 7,478 | ❌ DEPRECATED | 4 critical bugs |
| Phase 9 | 93 | 8,928 | ✅ CURRENT | Fair benchmark after bug fixes |
| Phase 10 | +2 | V737=104, V738=104 | ❌ INVALID | Oracle test-set leakage (5-layer root cause) |

## Bugs Fixed (Phase 9)

1. **TSLib per-entity prediction**: Fixed with per-entity batched inference
2. **Foundation prediction_length=7**: Fixed to use actual horizon
3. **Moirai 50-entity cap**: Removed cap
4. **ml_tabular single-horizon**: Fixed to run all horizons
5. **ml_tabular fillna(0)**: Pass NaN through for tree-based models
6. **Constant-prediction fairness guard**: `fairness_pass=False`
7. **asinh transform bug** (V737/V738): NF wrapper reads original-scale targets, removed outer asinh
