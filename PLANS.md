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
- [ ] Smoke test (10 entities, 2 horizons, 2 baselines, <1 min)
- [ ] Full benchmark run
- [ ] AutoFit model selection
- [ ] Leaderboard generation

### Phase F: Analysis (NOT STARTED)
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

## Next Actions

### Immediate (Smoke Test)
```bash
cd ~/projects/repo_root
python scripts/run_block3_benchmark.py --smoke-test
```

### Full Benchmark
```bash
python scripts/run_block3_benchmark.py \
    --config configs/block3.yaml \
    --output-dir runs/benchmarks/block3_20260203_225620
```

### AutoFit Selection
```bash
python scripts/run_auto_fit.py \
    --profile runs/orchestrator/20260129_073037/block3_20260203_225620/profile/profile.json \
    --config configs/block3.yaml
```

---

## Artifacts Reference

### Freeze Artifacts (READ-ONLY)
- `runs/offers_core_full_daily_wide_20260203_225620/offers_core_daily.parquet`
- `runs/offers_text_v1_20260129_073037_full/offers_text.parquet`
- `runs/edgar_feature_store_full_daily_wide_20260201_211317/`

### Block 3 Outputs
- Verification: `runs/orchestrator/.../block3_20260203_225620/verify/`
- Profile: `runs/orchestrator/.../block3_20260203_225620/profile/`
- Benchmark: `runs/benchmarks/block3_20260203_225620/`

---

## Commit History
- `3ce5509` WIDE2 freeze seal complete, all gates PASS
- (pending) Block 3 infrastructure + benchmark harness + docs
