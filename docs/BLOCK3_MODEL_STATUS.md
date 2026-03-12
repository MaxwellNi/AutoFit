# Block 3 Model Benchmark Status

> Last updated: 2026-03-12 (V734-V738 removed from display, 78 valid complete, V739 = new AutoFit baseline)
> Full results: `docs/BLOCK3_RESULTS.md`

## Snapshot

| Metric | Value |
|---|---:|
| Evaluation conditions per model | **104** (48+32+24) |
| Total metrics.json files | 88 |
| Total metric records | 8,660 |
| Unique models with results | 90 |
| **Valid complete (104/104)** | **78** |
| Partial (<104) | **12** |
| Phase 11 new models | 14 TSLib SOTA + V739 validation-based AutoFit |
| Total registered models | **42 tslib_sota + 9 deep + 23 transformer + 14 foundation + 4 irregular + 15 stat + 11 ml_tabular + 6 autofit = 127** |
| Active SLURM jobs | **0 RUNNING, 78 PENDING** (npin=47, cfisch=31) |
| Text embeddings | ❌ EMPTY — 4 generation jobs PENDING |
| V739 results | ❌ 0/104 — all 18 jobs PENDING (11 gpu + 7 l40s), 0 results landed |
| **AutoFit status** | V739 (validation-based, clean) = new baseline; prior versions retired |

## Oracle Test-Set Leakage (Historical — V734-V738 retired)

V734–V738 all used oracle tables built from Phase 9 **test-set** metrics to select/weight sub-models.
This constitutes test-set information leakage. All 5 versions have been permanently retired.
V739 replaces them using proper validation-based model selection (harness `val_raw` with 7-day embargo).
Full root cause analysis: see `docs/BLOCK3_RESULTS.md`.

## Condition Count Explained

| Task | Targets | Horizons | Ablations | Conditions |
|---|---:|---:|---:|---:|
| task1_outcome | 3 (funding_raised_usd, investors_count, is_funded) | 4 (1,7,14,30) | 4 (core_only, core_text, core_edgar, full) | 48 |
| task2_forecast | 2 (funding_raised_usd, investors_count) | 4 | 4 | 32 |
| task3_risk_adjust | 2 (funding_raised_usd, investors_count) | 4 | 3 (core_only, core_edgar, full — **no core_text**) | 24 |
| **Total** | | | | **104** |

## Active SLURM Jobs (2026-03-12)

### RUNNING: 0 jobs

All 78 PENDING jobs stuck behind other users (GPU partition: 78 running from other users, 504 total PENDING).
Top GPU consumer: elavdusinovi (51 running GPU jobs). Not a configuration issue — pure cluster congestion.

### PENDING: 78 jobs (npin=47, cfisch=31)

| Category | Account | Partition | Jobs | Status |
|----------|---------|-----------|-----:|--------|
| V739 AutoFit (new) | npin | gpu | 11 | ⏳ PENDING (Priority) |
| V739 AutoFit (l40s) | cfisch | l40s | 7 | ⏳ PENDING (Priority) |
| Gap-fill (Chronos2, TTM) | npin | gpu | 10 | ⏳ PENDING (Priority) |
| Gap-fill (ETSformer, LightTS, Pyraformer, Reformer) | npin | gpu | 24 | ⏳ PENDING (Priority) |
| NegBinomGLM gap-fill | cfisch | bigmem | 0 | ✅ COMPLETED (in NegativeBinomialGLM 16/104) |
| Gap-fill (6 tslib models) | cfisch | gpu | 11 | ⏳ PENDING (Priority) |
| Phase 11 TSLib SOTA (14 models) | cfisch | gpu | 11 | ⏳ PENDING (Priority) |
| Text embedding generation | npin+cfisch | gpu+l40s | 4 | ⏳ PENDING (Priority) |
| **Phase 12 re-runs** (core_text/full) | — | — | 40 scripts ready | 🚫 Blocked on text embeddings |

### RECENTLY COMPLETED

| Batch | Jobs | Elapsed | Result |
|-------|------|---------|--------|
| tsC recovery (5221718-22) | 5 | up to 2d | ✅ ETSformer/LightTS/Pyraformer/Reformer → 52/104 |

## Model Completion (78 valid complete + 12 partial)

### Valid Complete (78 models × 104/104 in Phase 9)

| Category | Count | Models |
|---|---:|---|
| deep_classical | 9 | NBEATS, NHITS, TFT, DeepAR, GRU, LSTM, TCN, MLP, DilatedRNN |
| foundation | 12 | Chronos, ChronosBolt, Timer, TimeMoE, MOMENT, TimesFM, Sundial, TimesFM2, LagLlama, Moirai, MoiraiLarge, Moirai2 |
| irregular | 4 | GRU-D, SAITS, BRITS, CSDI |
| ml_tabular | 11 | LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, HistGradientBoosting, MeanPredictor, SeasonalNaive, LightGBMTweedie, XGBoostPoisson, NegativeBinomialGLM (16/16 binary-only) |
| statistical | 15 | AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive, AutoCES, CrostonClassic, CrostonOptimized, CrostonSBA, DynamicOptimizedTheta, HistoricAverage, Holt, HoltWinters, Naive, WindowAverage |
| transformer_sota | 23 | PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx, xLSTM, TimeLLM, DeepNPTS, BiTCN, KAN, RMoK, SOFTS, StemGNN, DLinear, NLinear, TimeMixer, TimeXer, TSMixerx |
| tslib_sota | 4 | WPMixer, FITS, KANAD, CATS |
| autofit | 0 | V739 (0/104 PENDING — validation-based, no oracle) |

### Partial (12 models in Phase 9)

| Model | Records | Category | Status |
|---|---:|---|---|
| Chronos2 | 58 | foundation | ⏳ 5 gap-fill jobs PENDING |
| TTM | 58 | foundation | ⏳ 5 gap-fill jobs PENDING |
| Crossformer | 52 | tslib_sota | ⏳ gap-fill PENDING (cfisch) |
| MSGNet | 52 | tslib_sota | ⏳ gap-fill PENDING (cfisch) — was tsA TIMEOUT |
| PAttn | 52 | tslib_sota | ⏳ gap-fill PENDING (cfisch) — was tsA TIMEOUT |
| MambaSimple | 52 | tslib_sota | ⏳ gap-fill PENDING (cfisch) |
| TimeFilter | 52 | tslib_sota | ⏳ gap-fill PENDING (cfisch) — ⚠️ constant pred on first 52 |
| MultiPatchFormer | 52 | tslib_sota | ⏳ gap-fill PENDING (cfisch) — ⚠️ constant pred on first 52 |
| ETSformer | 52 | tslib_sota | ⏳ 6 gap-fill jobs PENDING (npin) |
| LightTS | 52 | tslib_sota | ⏳ 6 gap-fill jobs PENDING (npin) |
| Pyraformer | 52 | tslib_sota | ⏳ 6 gap-fill jobs PENDING (npin) |
| Reformer | 52 | tslib_sota | ⏳ 6 gap-fill jobs PENDING (npin) |

### AutoFit: V739 as New Baseline

V739 uses validation-based model selection (no oracle). All prior versions (V734–V738) retired due to oracle test-set leakage.
V739: 0/104 — 18 SLURM jobs PENDING (11 npin gpu + 7 cfisch l40s). Future iterations start from V739.

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

## Audit Exclusion List (16+ models)

| Finding | Models | Reason |
|---------|--------|--------|
| A (6) | Sundial, TimesFM2, LagLlama, Moirai, MoiraiLarge, Moirai2 | Context-mean fallback |
| B (5) | AutoCES, xLSTM, TimeLLM, StemGNN, TimeXer | Training crash fallback |
| C (2) | TimeMoE, MOMENT | Near-duplicate of Timer |
| G (3) | MICN, MultiPatchFormer, TimeFilter | 100% constant predictions |

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
| Phase 9 | 90 | 8,660 | ✅ CURRENT | Main benchmark |
| Phase 10 | +1 (V739) | 0 | ⏳ PENDING | V739 validation-based AutoFit |
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
