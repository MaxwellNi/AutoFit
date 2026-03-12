# Block 3 Model Benchmark Status

> Last updated: 2026-03-13 (V734-V738 ALL invalidated — oracle test-set leakage)
> Full results: `docs/BLOCK3_RESULTS.md`

## Snapshot

| Metric | Value |
|---|---:|
| Evaluation conditions per model | **104** (48+32+24) |
| Total metrics.json files | 110 (p9=88, p10=22) |
| Total metric records (all) | 9,180 (p9=8,972 + p10=208) |
| **Valid metric records** | **8,660** (excl. V734/V735/V736/V737/V738) |
| Invalid records (oracle leak) | 520 (p9: V734+V735+V736=312, p10: V737+V738=208) |
| Unique models with results | 95 (90 valid + 5 invalid) |
| **Valid complete (104/104)** | **77** |
| Partial (<104) | **13** |
| **INVALID AutoFit (ALL oracle-leaked)** | V734, V735, V736, V737, V738 |
| Phase 11 new models | 14 TSLib SOTA + V739 validation-based AutoFit |
| Total registered models | **42 tslib_sota + 9 deep + 23 transformer + 14 foundation + 4 irregular + 15 stat + 11 ml_tabular + 6 autofit = 127** |
| Active SLURM jobs | **0 RUNNING, 125 PENDING** (npin=47, cfisch=78) |
| Text embeddings | ❌ EMPTY — 4 generation jobs PENDING |
| V739 results | ❌ 0/104 — all 18 jobs PENDING (11 gpu + 7 l40s), 0 results landed |
| **AutoFit integrity** | ❌ V734-V738 ALL invalid; ✅ V739 clean (validation-based, no oracle) |

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

## Active SLURM Jobs (2026-03-12 ~21:00 UTC)

### RUNNING: 0 jobs

All jobs stuck in PENDING (Priority) — cluster heavily loaded.

### PENDING: 136 jobs (npin=47, cfisch=89)

| Category | Account | Partition | Jobs | Status |
|----------|---------|-----------|-----:|--------|
| V739 AutoFit (new) | npin | gpu | 11 | ⏳ PENDING (Priority) |
| V739 AutoFit (l40s) | cfisch | l40s | 7 | ⏳ PENDING (Priority) |
| Gap-fill (Chronos2, TTM) | npin | gpu | 10 | ⏳ PENDING (Priority) |
| Gap-fill (ETSformer, LightTS, Pyraformer, Reformer) | npin | gpu | 24 | ⏳ PENDING (Priority) |
| NegBinomGLM gap-fill | cfisch | bigmem | 11 | ⏳ PENDING (Priority) |
| Phase 11 TSLib SOTA (14 models) | cfisch | gpu | 11 | ⏳ PENDING (Priority) |
| Text embedding generation | npin+cfisch | gpu+l40s | 4 | ⏳ PENDING (Priority) |
| **Phase 12 re-runs** (core_text/full) | — | — | 40 scripts ready | 🚫 Blocked on text embeddings |

### RECENTLY COMPLETED

| Batch | Jobs | Elapsed | Result |
|-------|------|---------|--------|
| V737 npin (5226208-13) | 6 | 49-71 min | ✅ 104/104 records |
| V738 cfisch (5226229-33) | 5 | 30-75 min | ✅ 104/104 records (oracle-leaked) |
| V737 cfisch (5226239-43) | 5 | 62-115 min | ✅ merged into V737 results |
| tsC recovery (5221718-22) | 5 | up to 2d | ✅ ETSformer/LightTS/Pyraformer/Reformer → 52/104 |

## Model Completion (77 valid complete + 13 partial + 5 INVALID)

### Valid Complete (77 models × 104/104 in Phase 9)

| Category | Count | Models |
|---|---:|---|
| deep_classical | 9 | NBEATS, NHITS, TFT, DeepAR, GRU, LSTM, TCN, MLP, DilatedRNN |
| foundation | 12 | Chronos, ChronosBolt, Timer, TimeMoE, MOMENT, TimesFM, Sundial, TimesFM2, LagLlama, Moirai, MoiraiLarge, Moirai2 |
| irregular | 4 | GRU-D, SAITS, BRITS, CSDI |
| ml_tabular | 10 | LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, HistGradientBoosting, MeanPredictor, SeasonalNaive, LightGBMTweedie, XGBoostPoisson |
| statistical | 15 | AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive, AutoCES, CrostonClassic, CrostonOptimized, CrostonSBA, DynamicOptimizedTheta, HistoricAverage, Holt, HoltWinters, Naive, WindowAverage |
| transformer_sota | 23 | PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx, xLSTM, TimeLLM, DeepNPTS, BiTCN, KAN, RMoK, SOFTS, StemGNN, DLinear, NLinear, TimeMixer, TimeXer, TSMixerx |
| tslib_sota | 4 | WPMixer, FITS, KANAD, CATS |
| ~~autofit~~ | 0 | ~~AutoFitV734, AutoFitV735, AutoFitV736~~ → ALL INVALID (see below) |

### Partial (13 models in Phase 9)

| Model | Records | Category | Status |
|---|---:|---|---|
| Chronos2 | 58 | foundation | ⏳ 5 gap-fill jobs PENDING |
| TTM | 58 | foundation | ⏳ 5 gap-fill jobs PENDING |
| Crossformer | 52 | tslib_sota | ⏳ gap-fill needed |
| MSGNet | 52 | tslib_sota | ❌ tsA TIMEOUT |
| PAttn | 52 | tslib_sota | ❌ tsA TIMEOUT |
| MambaSimple | 52 | tslib_sota | ❌ tsA TIMEOUT |
| TimeFilter | 52 | tslib_sota | ❌ constant pred |
| MultiPatchFormer | 52 | tslib_sota | ❌ constant pred |
| ETSformer | 52 | tslib_sota | ⏳ 6 gap-fill jobs PENDING |
| LightTS | 52 | tslib_sota | ⏳ 6 gap-fill jobs PENDING |
| Pyraformer | 52 | tslib_sota | ⏳ 6 gap-fill jobs PENDING |
| Reformer | 52 | tslib_sota | ⏳ 6 gap-fill jobs PENDING |
| NegativeBinomialGLM | 16 | ml_tabular | ⏳ 11 gap-fill jobs PENDING (bigmem) |

### INVALID AutoFit (ALL oracle-leaked — V734 through V738)

| Model | Phase | Records | Oracle Table | Status |
|---|---|---:|---|---|
| AutoFitV734 | 9 | 104/104 | `ORACLE_TABLE_V734` (L425) — test-set avg_rank | ❌ oracle leak |
| AutoFitV735 | 9 | 104/104 | `ORACLE_TABLE_V735` (L689) — test-set RMSE, single best | ❌ oracle leak |
| AutoFitV736 | 9 | 104/104 | `ORACLE_TABLE_TOP3` (L751) — test-set RMSE, top-3 | ❌ oracle leak |
| AutoFitV737 | 10 | 104/104 | inherits V736's `ORACLE_TABLE_TOP3` | ❌ oracle leak |
| AutoFitV738 | 10 | 104/104 | `ORACLE_TABLE_V738` (L1313) — test-set MAE, top-5 | ❌ oracle leak |

All oracle tables are built from Phase 9 test-set metrics ("Rebuilt from Phase 9 clean data: 4,564 records").
This is test-set information leakage: models select/weight sub-models using test-set RMSE/MAE rankings.
All 520 records (312 Phase 9 + 208 Phase 10) are **scientifically invalid** for any paper ranking.

### Phase 11: New SOTA Models (14 TSLib + V739 AutoFit)

| Model | Category | Venue | Status |
|---|---|---|---|
| CFPT | tslib_sota | ICLR 2025 | ✅ Registered, awaiting SLURM |
| DeformableTST | tslib_sota | ICLR 2025 | ✅ Registered, awaiting SLURM |
| ModernTCN | tslib_sota | ICLR 2024 | ✅ Registered, awaiting SLURM |
| PathFormer | tslib_sota | ICLR 2024 | ✅ Registered, awaiting SLURM |
| SEMPO | tslib_sota | ICML 2024 | ✅ Registered, awaiting SLURM |
| TimePerceiver | tslib_sota | arXiv 2024 | ✅ Registered, awaiting SLURM |
| TimeBridge | tslib_sota | NeurIPS 2024 | ✅ Registered, awaiting SLURM |
| TQNet | tslib_sota | ICML 2024 | ✅ Registered, awaiting SLURM |
| PIR | tslib_sota | NeurIPS 2024 | ✅ Registered, awaiting SLURM |
| CARD | tslib_sota | ICLR 2024 | ✅ Registered, awaiting SLURM |
| PDF | tslib_sota | ICML 2024 | ✅ Registered, awaiting SLURM |
| TimeRecipe | tslib_sota | NeurIPS 2024 | ✅ Registered, awaiting SLURM |
| DUET | tslib_sota | NeurIPS 2024 | ✅ Registered (TSLib adapter), awaiting SLURM |
| SRSNet | tslib_sota | arXiv 2024 | ✅ Registered (TSLib adapter), awaiting SLURM |
| AutoFitV739 | autofit | Phase 11 | ✅ Registered, 11 SLURM jobs PENDING |

**Not integrated (3 models with blocking reasons):**

| Model | Reason | Resolution |
|---|---|---|
| Kairos | Foundation model (T5-based), requires IBM `tsfm` package — not pip-installable | Future: install `tsfm` from source |
| TimeMixerPP | Updated TimeMixer — vendor already has working TimeMixer, updating would break Phase 9 reproducibility | Post-Phase 9: update vendored TimeMixer.py |
| TabPFN_TS | Tabular foundation model, different paradigm from time series | Already partially handled as TabPFN in ml_tabular |

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
| Phase 9 | 90 valid + 3 invalid | 8,660 valid + 312 invalid | ✅ CURRENT (excl. V734-V736) | V734/V735/V736 ALL oracle-leaked |
| Phase 10 | +2 | V737=104, V738=104 | ❌ INVALID | Oracle test-set leakage (5-layer root cause) |
| Phase 11 | +14+1 | 0 | ⏳ PENDING | 14 TSLib SOTA + V739 AutoFit (no oracle) |
| Phase 12 | — | 0 | 🚫 BLOCKED | core_text/full re-runs after text embeddings ready |

## Text Embedding Status 🔴

| Item | Status |
|------|--------|
| Model | GTE-Qwen2-1.5B-instruct (Alibaba-NLP, Apache-2.0) |
| Output dim | 1536 → PCA 64 |
| Directory | `runs/text_embeddings/` |
| State | **EMPTY** — 6 prior generation attempts ALL failed |
| Root cause | Arrow string OOM (2.55 TiB allocation on .loc[] with ArrowStringArray) |
| Fix | `.astype("object")` + per-field truncation (commit f67d69e) |
| Pending jobs | 4 generation jobs PENDING (2 gpu + 2 l40s) |
| Impact | All existing core_text ≡ core_only, full ≡ core_edgar (text features = raw numeric only) |
| Re-run plan | 40 Phase 12 scripts in `.slurm_scripts/phase12/rerun/` |

## Code Audit (2026-03-12) ✅

Comprehensive code audit completed. Key findings:
- **No data leakage**: Temporal split with 7-day embargo correctly enforced; explicit leakage guards for co-determined columns
- **No critical logic bugs**: Horizon handling (h_nf=max(h,7)) is correct for all horizons [7,14,30,60]
- **V739 validation-based selection is sound**: Uses harness val_raw (temporal split), no test-set information leak
- **No anomalous metrics**: 0 NaN/Inf/zero MAE across 9,180 records
- **Historical OOM**: 106 failed jobs (mostly tslib_sota, ml_tabular, irregular) — already addressed with higher memory requests

## Bugs Fixed (Phase 9)

1. **TSLib per-entity prediction**: Fixed with per-entity batched inference
2. **Foundation prediction_length=7**: Fixed to use actual horizon
3. **Moirai 50-entity cap**: Removed cap
4. **ml_tabular single-horizon**: Fixed to run all horizons
5. **ml_tabular fillna(0)**: Pass NaN through for tree-based models
6. **Constant-prediction fairness guard**: `fairness_pass=False`
7. **asinh transform bug** (V737/V738): NF wrapper reads original-scale targets, removed outer asinh
