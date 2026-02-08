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

## Current Benchmark Results (2026-02-08)

### 4090 Run Progress
Output: `runs/benchmarks/block3_20260203_225620_4090_final/`

**12/36 shards complete** (GPU0: deep+transformer+foundation tasks 1-2 done, task3 in progress; GPU1: ml_tabular task1 in progress)

| Category | Real Results | Fallback | Notes |
|----------|-------------|----------|-------|
| deep_classical (4) | 15 | 1 | NBEATS H=1 borderline |
| transformer_sota (15) | 52 | 28 | iTransformer/TSMixer/RMoK/SOFTS/StemGNN n_series bug → re-run |
| foundation (3) | 8 | 4 | TimesFM not installed, Moirai MAE=637K best |
| ml_tabular (15) | — | — | In progress (SVR running) |
| statistical (5) | — | — | Pending |
| irregular (2) | — | — | Pending |

### Best Models So Far (task1_outcome, H=1)
| Model | Category | MAE | RMSE |
|-------|----------|-----|------|
| **Informer** | transformer_sota | 619,770 | 2,234,081 |
| **DeepAR** | deep_classical | 619,773 | 2,234,081 |
| **Moirai** | foundation | 637,781 | 2,162,798 |
| NBEATS | deep_classical | 601,865 | 2,181,472 |

### n_series Bug Fix (commit `014ac92`)
5 models affected: iTransformer, TSMixer, RMoK, SOFTS, StemGNN.
Fix: `n_series=panel["unique_id"].nunique()` (was hardcoded to 1).
Re-run script: `scripts/rerun_nseries_models.sh`.

See [docs/BLOCK3_RESULTS.md](docs/BLOCK3_RESULTS.md) for full live results.

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
- `014ac92` Fix n_series dynamic computation, SLURM mem 128G, 4090 launch scripts
- `fcbe970` Model registry rewrite: 44 models, panel data fix, all 6 categories
- `3ce5509` WIDE2 freeze seal complete, all gates PASS
- `5ca7fd8` Block3 init: benchmark harness + autofit + concept bottleneck + docs
