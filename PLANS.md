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
- [ ] ⚠️ Fix panel data compatibility for deep/transformer models
- [ ] ⚠️ Fix statistical models (data scale overflow)
- [ ] Complete ml_tabular for all tasks/ablations
- [ ] Foundation models (Chronos, TimesFM)
- [ ] Leaderboard generation

**Detailed Status**: See [docs/BLOCK3_MODEL_STATUS.md](docs/BLOCK3_MODEL_STATUS.md)

### Phase F: Analysis (BLOCKED)
- [ ] TCAV-style concept importance analysis
- [ ] Ablation study (core_only vs full)
- [ ] Horizon sensitivity analysis
- [ ] Error analysis by entity type

**Note**: Blocked pending valid deep/transformer results.

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

### Blocked Results (Fallback)
| Category | Models | Issue |
|----------|--------|-------|
| deep_classical | NBEATS, NHITS, TFT, DeepAR | Panel data incompatibility |
| transformer_sota | PatchTST, iTransformer, TimesNet, TSMixer | Same as above |
| statistical | AutoARIMA, ETS, Theta, MSTL | Data scale overflow |

See [docs/BLOCK3_MODEL_STATUS.md](docs/BLOCK3_MODEL_STATUS.md) for full details.

---

## Next Actions

### Priority 1: Complete ml_tabular (Immediate)
```bash
# Run all ml_tabular models for all tasks
for TASK in task1_outcome task2_forecast task3_risk_adjust; do
  for ABL in core_only full; do
    python scripts/run_block3_benchmark_shard.py \
      --task $TASK --category ml_tabular --ablation $ABL \
      --preset standard \
      --output-dir runs/benchmarks/block3_20260203_225620_4090_standard/$TASK/ml_tabular/$ABL
  done
done
```

### Priority 2: Fix Deep/Transformer Models
- Implement entity-sampling in `src/narrative/block3/models/deep_models.py`
- Or switch to panel-aware backend (pytorch-forecasting, tsai)

### Priority 3: Fix Statistical Models
- Implement entity-sampling in `src/narrative/block3/models/statistical.py`

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
