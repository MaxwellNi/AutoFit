# Execution Plan (Block 3 Modeling)

This plan tracks Block 3 implementation after the WIDE2 freeze seal (stamp `20260203_225620`).

## ✅ WIDE2 Freeze Complete
All steps completed, all gates PASS:
- `pointer_valid`: PASS
- `column_manifest`: PASS
- `raw_cardinality_coverage`: PASS
- `freeze_candidates`: PASS
- `offer_day_coverage_exact`: PASS

Verification: `scripts/block3_verify_freeze.py`

---

## Block 3 Implementation Status

### Phase A: Infrastructure (COMPLETE)
- [x] Freeze verification script (`scripts/block3_verify_freeze.py`)
- [x] Unified dataset interface (`src/narrative/data_preprocessing/block3_dataset.py`)
- [x] Data profiling script (`scripts/block3_profile_data.py`)
- [x] Block 3 configuration (`configs/block3.yaml`)

### Phase B: Benchmark Framework (COMPLETE)
- [x] Benchmark harness (`scripts/run_block3_benchmark.py`)
- [x] Statistical baselines (SeasonalNaive)
- [x] ML Tabular (LightGBM, XGBoost, CatBoost)
- [x] Deep Classical (N-BEATS, TFT)
- [x] Transformer SOTA (PatchTST, iTransformer, TimeMixer)
- [x] GluonTS (DeepAR, WaveNet)
- [x] Foundation (TimesFM, Chronos, Moirai)

### Phase C: AutoFit (COMPLETE)
- [x] Rule-based composer (`src/narrative/auto_fit/rule_based_composer.py`)
- [x] Meta-feature extraction via profiler

### Phase D: Interpretability (COMPLETE)
- [x] Concept bottleneck (`src/narrative/explainability/concept_bottleneck.py`)
- [x] Concept bank with 10 concepts
- [x] Marginal contribution logging

### Phase E: Execution (IN PROGRESS)
- [x] GPU benchmark execution (deep_classical, transformer_sota)
- [x] ml_tabular partial run (XGBoost, LightGBM)
- [x] Model registry rewrite: 44 models across 6 categories
- [x] Panel data fix: all categories receive entity-panel kwargs
- [x] Deep/Transformer: 19 NeuralForecast models (NBEATS, NHITS, TFT, DeepAR, PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx, BiTCN, KAN, RMoK, SOFTS, StemGNN)
- [x] Statistical: 5 StatsForecast models with entity-sampled panel
- [x] Foundation: Chronos + Moirai + TimesFM wrappers
- [x] Irregular: GRU-D + SAITS via PyPOTS
- [x] Benchmark harness updated for all 6 categories
- [ ] Full benchmark run on 4090 (dual GPU)
- [ ] Full benchmark run on 3090 (dual GPU)
- [ ] Leaderboard generation + paper tables

**Detailed Status**: See [docs/BLOCK3_MODEL_STATUS.md](docs/BLOCK3_MODEL_STATUS.md)

### Phase F: Analysis (PENDING — awaiting full benchmark completion)
- [ ] TCAV-style concept importance analysis
- [ ] Ablation study (core_only vs full)
- [ ] Horizon sensitivity analysis
- [ ] Error analysis by entity type

---

## Block 3 Profile Results
From `runs/orchestrator/20260129_073037/block3_20260203_225620/profile/profile.json`:

| Meta-Feature            | Value  |
|-------------------------|--------|
| `nonstationarity_score` | 0.5000 |
| `periodicity_score`     | 0.5156 |
| `multiscale_score`      | 0.7324 |
| `long_memory_score`     | 0.0404 |
| `irregular_score`       | 1.2317 |
| `heavy_tail_score`      | 1.0000 |
| `exog_strength`         | 0.0000 |
| `edgar_strength`        | 0.0000 |
| `text_strength`         | 0.0000 |
| `missing_rate`          | 0.7068 |

### AutoFit Recommendation (based on profile)
- **Backbone**: TimeMixer (multiscale_score > 0.7)
- **Fusion**: none (exog_strength < 0.1)
- **Loss**: Huber (heavy_tail_score > 0.3)

---

## Current Benchmark Results (2026-02-07)

### Paper-Ready Results
| Model | Category | MAE | Status |
|-------|----------|-----|--------|
| XGBoost | ml_tabular | **274,952** | ✅ Paper-ready |
| LightGBM | ml_tabular | 360,834 | ✅ Paper-ready |

### Fixed (2026-02-08) — All Categories Now Panel-Aware
| Category | Models | Status |
|----------|--------|--------|
| deep_classical (4) | NBEATS, NHITS, TFT, DeepAR | ✅ Fixed (entity-sampled panel via NeuralForecast) |
| transformer_sota (15) | PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx, BiTCN, KAN, RMoK, SOFTS, StemGNN | ✅ Fixed (NeuralForecast) |
| statistical (5) | AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive | ✅ Fixed (entity-sampled panel via StatsForecast) |
| foundation (3) | Chronos, Moirai, TimesFM | ✅ Fixed (entity context extraction) |
| irregular (2) | GRU-D, SAITS | ✅ Fixed (irregular panel via PyPOTS) |

See [docs/BLOCK3_MODEL_STATUS.md](docs/BLOCK3_MODEL_STATUS.md) for full details.

---

## Next Actions

### Priority 1: Full Benchmark on 4090 (Dual GPU)
Run all 44 models × 3 tasks × ablations on 4090 (2× RTX 4090).
Distribute GPU-heavy categories across GPU0/GPU1 via `CUDA_VISIBLE_DEVICES`.

### Priority 2: Full Benchmark on 3090 (Dual GPU)
Sync code + data to 3090 (2× RTX 3090), replicate same run.

### Priority 3: Results Aggregation & Paper Tables
Collect all results into leaderboard MD table.
Generate LaTeX tables for KDD'26 paper.

---

## Artifacts Reference

### Freeze Artifacts (READ-ONLY)
All paths resolved via `docs/audits/FULL_SCALE_POINTER.yaml`:
- `${pointer.offers_core_daily.dir}/offers_core_daily.parquet`
- `${pointer.offers_text.dir}/offers_text.parquet`
- `${pointer.edgar_store_full_daily.dir}/`

**Hard Rule**: No hard-coded stamp paths in production code.

### Block 3 Outputs
- Verification: `runs/orchestrator/.../block3_${stamp}/verify/`
- Profile: `runs/orchestrator/.../block3_${stamp}/profile/`
- Benchmark: `runs/benchmarks/block3_${stamp}/`

---

## Commit History
- `3ce5509` WIDE2 freeze seal complete, all gates PASS
- `5ca7fd8` Block3 init: benchmark harness + autofit + concept bottleneck + docs
