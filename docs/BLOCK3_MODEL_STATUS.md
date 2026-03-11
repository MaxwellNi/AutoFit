# Block 3 Model Benchmark Status

> Last updated: 2026-03-11 (Phase 10 — V738/V737 oracle leak audit, 93 models registered)
> Full results: `docs/BLOCK3_RESULTS.md`

## Snapshot

| Metric | Value |
|---|---:|
| Evaluation conditions per model | **104** (48+32+24) |
| metrics.json files | 88 |
| Valid metric records | 8,928 |
| Unique models with results | 93 |
| Models with 104/104 complete | 76 |
| Models with partial results | 17 |
| Active SLURM jobs | 14 RUNNING + 10 PENDING |
| Active AutoFit versions | V734, V735, V736, V737, V738 |
| **AutoFit integrity** | ❌ **ALL versions have oracle test-set leakage** |

## CRITICAL: Oracle Test-Set Leakage in All AutoFit Versions

See `docs/BLOCK3_RESULTS.md` §V738 Fairness Audit for full details.
- `ORACLE_TABLE_V738` (L1313): 44 conditions × top-5 models, MAE from Phase 9 test set
- `ORACLE_TABLE_TOP3` (L751): 44 conditions × top-3 models, RMSE from Phase 9 test set
- `val_frac=0.2` is dead code — never used in fit()/predict()
- **All V734-V738 rankings are scientifically invalid**

## Condition Count Explained

| Task | Targets | Horizons | Ablations | Conditions |
|---|---:|---:|---:|---:|
| task1_outcome | 3 (funding_raised_usd, investors_count, is_funded) | 4 (1,7,14,30) | 4 (core_only, core_text, core_edgar, full) | 48 |
| task2_forecast | 2 (funding_raised_usd, investors_count) | 4 | 4 | 32 |
| task3_risk_adjust | 2 (funding_raised_usd, investors_count) | 4 | 3 (core_only, core_edgar, full — **no core_text**) | 24 |
| **Total** | | | | **104** |

## Active SLURM Jobs (2026-03-11 ~03:00 UTC)

### RUNNING (14 jobs)

| Job ID | Name | Time | Memory | Node | Progress |
|-------:|------|------|--------|------|----------|
| 5219864 | p9r2_tsA_t1_fu | 1d 17h | 640G | iris-184 | 4 saves, PAttn epoch 10/100 |
| 5219865 | p9r2_tsA_t2_fu | 1d 17h | 640G | iris-171 | 4 saves, MambaSimple loading |
| 5219866 | p9r2_tsA_t3_fu | 1d 17h | 640G | iris-173 | 4 saves, MambaSimple training |
| 5221718 | p10r_tsC_t1_ce | 18h | 320G | iris-180 | 8 saves, EDGAR joining |
| 5221719 | p10r_tsC_t1_co | 18h | 320G | iris-175 | 10 saves, Reformer epoch 14/100 |
| 5221720 | p10r_tsC_t1_ct | 18h | 320G | iris-177 | 11 saves, Reformer epoch 20/100 |
| 5221721 | p10r_tsC_t3_ce | 11h | 320G | iris-176 | 5 saves, Reformer epoch 0/100 |
| 5221722 | p10r_tsC_t3_co | 11h | 320G | iris-185 | 7 saves, LightTS loading |
| 5226208 | p10_v737_t1_co | 37m | 128G | iris-179 | 6 saves |
| 5226209 | p10_v737_t2_co | 33m | 128G | iris-186 | 6 saves |
| 5226210 | p10_v737_t3_co | 33m | 128G | iris-178 | 5 saves |
| 5226211 | p10_v737_t1_ce | 32m | 192G | iris-174 | 6 saves |
| 5226212 | p10_v737_t2_ce | 32m | 192G | iris-170 | 6 saves |
| 5226213 | p10_v737_t3_ce | 32m | 192G | iris-183 | 6 saves |

### PENDING (10 jobs)

| Job ID | Name | Memory | Reason |
|-------:|------|--------|--------|
| 5226229 | cf_v738_t1_ct | 384G | Resources |
| 5226230 | cf_v738_t2_ct | 384G | Priority |
| 5226231 | cf_v738_t1_fu | 384G | Priority |
| 5226232 | cf_v738_t2_fu | 384G | Priority |
| 5226233 | cf_v738_t3_fu | 384G | Priority |
| 5226239 | cf_v737_t1_ct | 384G | Priority |
| 5226240 | cf_v737_t2_ct | 384G | Priority |
| 5226241 | cf_v737_t1_fu | 384G | Priority |
| 5226242 | cf_v737_t2_fu | 384G | Priority |
| 5226243 | cf_v737_t3_fu | 384G | Priority |

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

### Partial (17 models)

| Model | Records | Category | Status |
|---|---:|---|---|
| Chronos2 | 58 | foundation | 🔄 |
| TTM | 58 | foundation | 🔄 |
| TimeFilter | 52 | tslib_sota | ❌ constant pred |
| MultiPatchFormer | 52 | tslib_sota | ❌ constant pred |
| MSGNet | 52 | tslib_sota | 🔄 tsA |
| PAttn | 52 | tslib_sota | 🔄 tsA |
| MambaSimple | 52 | tslib_sota | 🔄 tsA |
| Crossformer | 52 | tslib_sota | 🔄 |
| ETSformer | 41 | tslib_sota | 🔄 tsC |
| LightTS | 41 | tslib_sota | 🔄 tsC |
| Pyraformer | 41 | tslib_sota | 🔄 tsC |
| Reformer | 41 | tslib_sota | 🔄 tsC |
| NegativeBinomialGLM | 16 | ml_tabular | 🔄 |
| KANAD | 104 | transformer_sota | ✅ |
| FITS | 104 | transformer_sota | ✅ |
| CATS | 104 | tslib_sota | ✅ |
| SeasonalNaive | 104 | ml_tabular | ✅ |

## Audit Exclusion List (19 models)

| Finding | Models | Reason |
|---------|--------|--------|
| A (6) | Sundial, TimesFM2, LagLlama, Moirai, MoiraiLarge, Moirai2 | Context-mean fallback |
| B (5) | AutoCES, xLSTM, TimeLLM, StemGNN, TimeXer | Training crash fallback |
| C (2) | TimeMoE, MOMENT | Near-duplicate of Timer |
| G (3) | MICN, MultiPatchFormer, TimeFilter | 100% constant predictions |
| J (3) | AutoFitV734, AutoFitV735, AutoFitV736 | Oracle test-set leakage |

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
| Phase 10 | +2 | ongoing | 🔄 IN PROGRESS | V737, V738 (both oracle-leaked) |

## Bugs Fixed (Phase 9)

1. **TSLib per-entity prediction**: Fixed with per-entity batched inference
2. **Foundation prediction_length=7**: Fixed to use actual horizon
3. **Moirai 50-entity cap**: Removed cap
4. **ml_tabular single-horizon**: Fixed to run all horizons
5. **ml_tabular fillna(0)**: Pass NaN through for tree-based models
6. **Constant-prediction fairness guard**: `fairness_pass=False`
7. **asinh transform bug** (V737/V738): NF wrapper reads original-scale targets, removed outer asinh
