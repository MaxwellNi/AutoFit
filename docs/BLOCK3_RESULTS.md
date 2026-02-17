# Block 3 Benchmark Results — Phase 7 IN PROGRESS

**Last Updated**: 2026-02-17 10:29 CET
**Phase 7 Dir**: `runs/benchmarks/block3_20260203_225620_phase7/`
**Platform**: ULHPC Iris HPC — GPU (4×V100 32GB, 756GB) + Batch (28c, 112GB)
**Freeze Stamp**: `20260203_225620`
**Model Registry (code)**: 72 models across 7 categories (submitted Phase 7 roster: 67)
**Ablations**: core_only, core_text, core_edgar, full
**Progress (canonical 121 shards)**: 99 completed, 0 running, 20 pending, 2 failed
**Detailed live tables**: `docs/BLOCK3_LIVE_SUMMARY.md`

---

## Live Status Snapshot (2026-02-17 10:29 CET)

Data sources:
- `squeue -u $USER`
- `sacct -u $USER -S 2026-02-12`
- `sacctmgr show qos iris-batch-long,iris-gpu-long`
- `scontrol show job <jobid>` + log tails in `/work/projects/eint/logs/phase7/`

Queue status:
- RUNNING: 8 (batch=8, gpu=0)
- PENDING: 122 (batch=111, gpu=11)
- Pending reasons:
  - batch queue: mostly `QOSMaxJobsPerUserLimit`
  - gpu queue: mostly `QOSGrpNodeLimit`
- QOS hard caps confirmed:
  - `iris-batch-long MaxJobsPerUser=8`
  - `iris-gpu-long MaxJobsPerUser=4`

Canonical shard progress (121 names from submitted Phase 7 scripts):
- COMPLETED: 99
- RUNNING: 0
- PENDING: 20
- FAILED: 2 (`p7_af1_t1_ce`, `p7_af2_t1_ce`)

Program-wide progress (canonical + V7.1 pilot + V7.1 full):
- COMPLETED: 166 / 297
- RUNNING: 8 / 297
- PENDING: 121 / 297
- FAILED: 2 / 297

Per-shard-group status:

| Group | Completed | Running | Pending | Total |
|-------|-----------|---------|---------|-------|
| ml_tabular | 5 | 0 | 6 | 11 |
| statistical | 3 | 0 | 8 | 11 |
| deep_classical | 11 | 0 | 0 | 11 |
| transformer_sota_A | 8 | 1 | 2 | 11 |
| transformer_sota_B | 11 | 0 | 0 | 11 |
| foundation_chronos | 8 | 0 | 3 | 11 |
| foundation_moirai | 8 | 0 | 3 | 11 |
| foundation_hf | 8 | 0 | 3 | 11 |
| irregular | 11 | 0 | 0 | 11 |
| autofit_shard1 | 3 | 4 | 4 | 11 |
| autofit_shard2 | 1 | 4 | 6 | 11 |

Running-job log health:
- No new fatal exceptions in sampled running jobs.
- Non-blocking warnings observed:
  - deterministic CUDA warnings from NeuralForecast scaler
  - expected constant-candidate skips in AutoFit quick screen
  - leakage-guard column-drop messages are active

Materialized output snapshot under current Phase 7 dir:
- `metrics.json` files: 59
- metric records currently on disk: 3884

## AutoFit V7.1 Implementation Status (Code Complete, Pilot Pending)

Implemented components:
- fairness hard guards and coverage schema fields in `scripts/run_block3_benchmark_shard.py`
- comparability filter in `scripts/aggregate_block3_results.py`
- V7.1 wrapper with lane routing, dynamic thresholds, objective switching, and dynamic blending in `src/narrative/block3/models/autofit_wrapper.py`
- candidate expansion wrappers in `src/narrative/block3/models/traditional_ml.py`
- registry exposure updates in `src/narrative/block3/models/registry.py`
- V7.1 config flags in `configs/block3_tasks.yaml`
- Pilot submission mode and V7.1 inclusion in `scripts/submit_phase7_full_benchmark.sh`

Validation status:
- `test/test_model_registry.py`: PASS
- `test/test_autofit_v71_no_leakage.py`: PASS
- `test/test_autofit_v71_coverage_guard.py`: PASS
- `test/test_autofit_v71_objective_switch.py`: PASS
- `test/test_autofit_v71_reproducibility.py`: PASS
- Total: 35 passed

Pilot submission status (V7.1 extreme):
- Command: `bash scripts/submit_phase7_v71_extreme.sh --pilot`
- Submitted jobs: 110 (`p7x_*`, JobID `5179607-5179716`)
- Run tag: `20260214_032205`
- Output root: `runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205`
- Log root: `/work/projects/eint/logs/phase7_v71extreme_20260214_032205`
- Stage-A composition:
  - 11 `ml_tabular` tuned reference shards
  - 11 `deep_classical` reference shards
  - 11 `transformer_sota` reference shards
  - 11 `foundation` reference shards
  - 11 `autofit` baseline shards (`AutoFitV6,AutoFitV7`)
  - 55 `AutoFitV71` grid shards (5 variants × 11 task-ablation combos)
- Live queue state:
  - `p7x_*`: COMPLETED=9, RUNNING=3, PENDING=98, FAILED=0

Full submission status (V7.1 extreme):
- Command: `bash scripts/submit_phase7_v71_extreme.sh --full --v71-variant=g02`
- Submitted jobs: 66 (`p7xF_*`, JobID `5179902-5179967`)
- Run tag: `20260214_130737`
- Output root: `runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737`
- Queue state: COMPLETED=0, RUNNING=0, PENDING=66

Current cross-model snapshot (materialized outputs only):
- Phase 7 (`runs/benchmarks/block3_20260203_225620_phase7`):
  - rows=3884, models=66
  - best category-level models by target:
    - `funding_raised_usd`: deep_classical (`NHITS`, MAE=374,432)
    - `investors_count`: deep_classical / transformer_sota tie (`NBEATS`/`NBEATSx`, MAE=44.73)
    - `is_funded`: deep_classical (`NHITS`, MAE=0.03232)
  - AutoFit family best observed MAE:
    - `funding_raised_usd`: `AutoFitV3` (395,552)
    - `investors_count`: `AutoFitV1`/`AutoFitV3` (113.91)
    - `is_funded`: `AutoFitV4` (0.08891)
- V7.1 pilot run (`..._phase7_v71extreme_20260214_032205`):
  - rows=396, models=4 (deep refs completed so far)
  - `AutoFitV71` materialized rows so far: 0 (jobs still pending/running)

## Phase 7 Overview (2026-02-12 -> 2026-02-14)

Phase 7 is the definitive full-scale benchmark with 67 models × 3 tasks × 4 ablations on ULHPC Iris.
Previous phases (1-6) served as development runs; Phase 7 incorporates ALL fixes.

### Key Code Changes (Phase 7)
1. **Entity coverage**: Increased from 200 to 2000 entities (non-n_series models)
2. **RobustFallback**: Catches all exceptions, auto-fallback to Ridge regression
3. **Hybrid predict**: Ridge regression on features for unseen test entities
4. **EDGAR covariates**: Now properly passed to deep/transformer models as exogenous features
5. **AutoFit target_transform**: Auto log1p/expm1 for skewed regression targets
6. **EDGAR timezone fix**: Strip UTC timezone from datetime64 for merge_asof compatibility
7. **Statistical OOM fix**: Memory increased from 64G to 112G for text/EDGAR joins

### Completed Shards (23/121)

| Shard | Records |
|-------|---------|
| task1_outcome/autofit/core_only | 15 |
| task1_outcome/autofit/core_text | 15 |
| task1_outcome/deep_classical/core_only | 48 |
| task1_outcome/deep_classical/core_text | 48 |
| task1_outcome/irregular/core_only | 24 |
| task1_outcome/ml_tabular/core_only | 38 |
| task1_outcome/ml_tabular/core_text | 38 |
| task1_outcome/statistical/core_only | 40 |
| task1_outcome/transformer_sota/core_only | 60 |
| task1_outcome/transformer_sota/core_text | 50 |
| task2_forecast/autofit/core_only | 15 |
| task2_forecast/autofit/core_text | 15 |
| task2_forecast/deep_classical/core_only | 32 |
| task2_forecast/deep_classical/core_text | 32 |
| task2_forecast/ml_tabular/core_only | 28 |
| task2_forecast/ml_tabular/core_text | 28 |
| task2_forecast/statistical/core_only | 40 |
| task2_forecast/transformer_sota/core_only | 50 |
| task2_forecast/transformer_sota/core_text | 50 |
| task3_risk_adjust/autofit/core_only | 15 |
| task3_risk_adjust/deep_classical/core_only | 32 |
| task3_risk_adjust/ml_tabular/core_only | 28 |
| task3_risk_adjust/statistical/core_only | 40 |
| **Total** | **781** |

### SLURM Queue Status (2026-02-13 02:15 UTC)
- **COMPLETED**: 23 shards (781 records)
- **RUNNING**: 12 jobs (autofit×7, statistical_co×1, transformer_sota_A×4)
- **PENDING**: 96 jobs
- **FAILED (OOM, fixed)**: 4 statistical tasks → resubmitted with 112G

---

## Prior Results Reference (Phase 1 — 2,646 records)

---

## Phase 3 Fixes (2026-02-12)

Phase 3 addresses 6 critical issues identified from Phase 1 results analysis:

### Fix 1: Deep/Transformer Entity Coverage (SEVERE)
- **Problem**: 19 NeuralForecast models produced near-constant predictions because only 200 training entities were sampled, giving <5% test coverage. Unseen entities fell back to global_mean.
- **Fix**: Increased `max_entities` to 2000 (non-n_series) / kept 200 (n_series models) and reduced `min_obs` to 10. Added Ridge regression fallback for unseen test entities instead of global_mean.
- **Impact**: Predictions now vary per-entity even for unseen entities via feature-based Ridge regression.

### Fix 2: EDGAR As-Of Join (MEDIUM)
- **Problem**: Exact `cik+crawled_date_day` JOIN had near-0% match rate — SEC filings don't align with exact crawl dates.
- **Fix**: Switched to `pd.merge_asof(direction="backward", tolerance="90D")` — matches most recent EDGAR filing within 90 days (quarterly filing cadence). No future leakage.
- **Impact**: EDGAR features now actually populated for rows with CIK, enabling meaningful core_only vs core_edgar ablation.

### Fix 3: AutoFit V3Max Timeout (MEDIUM)
- **Problem**: Exhaustive search with K=8 (256 combos) exceeded 48h SLURM walltime, producing incomplete records.
- **Fix**: Reduced `_MAX_EXHAUSTIVE_K` from 8 to 6 (64 combos). Added 30-minute time budget with early termination.
- **Impact**: V3Max now completes within walltime for all target×horizon combinations.

### Fix 4: GBDT Count-Target Loss (MEDIUM)
- **Problem**: LightGBM/CatBoost with default MSE loss performed ≈ MeanPredictor on `investors_count` (MAE≈483 vs 484).
- **Fix**: Auto-detects count-like targets (non-negative integers) and switches to `tweedie` (LightGBM) or `count:poisson` (XGBoost) objective.
- **Impact**: Proper loss function for discrete count distributions improves gradient signal.

### Fix 5: Horizon Deduplication (LOW)
- **Problem**: Cross-sectional models (ml_tabular) produced identical results across all horizons, wasting 75% compute.
- **Fix**: `ml_tabular` category now runs single horizon only (features are horizon-independent).
- **Impact**: ~75% reduction in ml_tabular SLURM runtime.

### Fix 6: Entity Coverage (LOW)
- **Problem**: Statistical models sampled only 50 entities, irregular models 200.
- **Fix**: Statistical → 500, Irregular → 1000.
- **Impact**: Better test coverage, more diverse entity representation.

### Phase 3 SLURM Jobs
- **Job IDs**: 5176995 – 5177036 (42 shards)
- **Configuration**: 7 categories × 3 tasks × 2 ablations = 42 shards
- **Verification**: 29/29 audit checks PASS (`scripts/verify_phase3_fixes.py`)

---

## Summary

Phase 1 benchmark is **100% complete**. All 12 SLURM jobs (6 autofit + 6 ml_tabular) finished
with exit code 0. Results cover 7 model categories, 49 distinct models, 4 horizons, 2 ablations,
and 3 prediction tasks.

## Completion Status

| Category | Models | Records | Status |
|----------|--------|---------|--------|
| ml_tabular (15) | 15 | ~784 | COMPLETE |
| statistical (5) | 5 | ~384 | COMPLETE |
| deep_classical (4) | 4 | ~384 | COMPLETE |
| transformer_sota (15) | 15 | ~384 | COMPLETE |
| foundation (2) | 2 | ~192 | COMPLETE |
| irregular (2) | 2 | ~192 | COMPLETE |
| autofit (6) | 6 | ~326 | COMPLETE |
| **Total** | **49** | **2,646** | **100%** |

## Configuration

- **Horizons**: [1, 7, 14, 30] days
- **Ablations**: `core_only`, `core_edgar`
- **Tasks**: `task1_outcome`, `task2_forecast`, `task3_risk_adjust`
- **Targets**: `funding_raised_usd`, `investors_count`, `is_funded`
- **Temporal split**: strict temporal ordering, NO shuffle
- **Leakage guard**: Target synonym groups enforced (`_TARGET_LEAK_GROUPS`)
- **Bootstrap**: n_bootstrap=1000 for confidence intervals

---

## Leaderboard: `funding_raised_usd` (1,144 records, 47 models)

| Rank | Model | Category | Best MAE | Best RMSE | vs #1 |
|------|-------|----------|----------|-----------|-------|
| 1 | **RandomForest** | ml_tabular | 400,777 | 2,013,231 | -- |
| 2 | AutoFitV1 | autofit | 400,777 | 2,013,231 | +0.0% |
| 3 | AutoFitV3 | autofit | 400,777 | 2,013,231 | +0.0% |
| 4 | AutoFitV3Max | autofit | 400,777 | 2,013,231 | +0.0% |
| 5 | AutoFitV2E | autofit | 429,035 | 1,814,931 | +7.1% |
| 6 | AutoFitV2 | autofit | 429,035 | 1,814,931 | +7.1% |
| 7 | AutoFitV3E | autofit | 442,300 | 1,856,673 | +10.4% |
| 8 | XGBoost | ml_tabular | 469,788 | 2,072,097 | +17.2% |
| 9 | LightGBM | ml_tabular | 480,588 | 2,215,708 | +19.9% |
| 10 | HistGBT | ml_tabular | 500,905 | 2,316,877 | +25.0% |
| 11 | ExtraTrees | ml_tabular | 507,041 | 2,335,810 | +26.5% |
| 12 | KNN | ml_tabular | 588,342 | 4,039,821 | +46.8% |
| 13 | CatBoost | ml_tabular | 602,553 | 2,627,687 | +50.3% |
| 14 | SVR | ml_tabular | 679,121 | 2,373,647 | +69.5% |
| 15 | Moirai | foundation | 691,983 | 2,359,979 | +72.7% |
| 16 | SAITS | irregular | 1,079,208 | 2,413,370 | +169.3% |
| 17 | GRU-D | irregular | 1,079,208 | 2,413,370 | +169.3% |
| 18 | QuantileReg | ml_tabular | 1,091,025 | 41,089,575 | +172.2% |
| 19 | MeanPredictor | ml_tabular | 2,133,181 | 2,863,605 | +432.3% |
| 20 | FEDformer | transformer | 4,616,452 | 4,955,261 | +1052% |

*27 more models omitted (all >+1500% vs #1)*

## Leaderboard: `investors_count` (1,148 records, 48 models)

| Rank | Model | Category | Best MAE | Best RMSE | vs #1 |
|------|-------|----------|----------|-----------|-------|
| 1 | **RandomForest** | ml_tabular | 95.96 | 1,163.77 | -- |
| 2 | ExtraTrees | ml_tabular | 113.91 | 1,276.11 | +18.7% |
| 3 | AutoFitV3 | autofit | 113.91 | 1,276.11 | +18.7% |
| 4 | AutoFitV3Max | autofit | 113.91 | 1,276.11 | +18.7% |
| 5 | XGBoost | ml_tabular | 197.21 | 1,188.69 | +105.5% |
| 6 | KNN | ml_tabular | 245.33 | 1,333.82 | +155.7% |
| 7 | AutoFitV3E | autofit | 263.83 | 1,214.47 | +174.9% |
| 8 | QuantileReg | ml_tabular | 293.11 | 1,416.68 | +205.5% |
| 9 | SVR | ml_tabular | 294.63 | 1,409.77 | +207.0% |
| 10 | Moirai | foundation | 306.48 | 1,404.81 | +219.4% |
| 11 | GRU-D | irregular | 310.90 | 1,420.96 | +224.0% |
| 12 | AutoFitV1 | autofit | 310.90 | 1,421.50 | +224.0% |
| 13 | SAITS | irregular | 311.21 | 1,421.27 | +224.3% |
| 14 | HistGBT | ml_tabular | 313.13 | 1,250.07 | +226.3% |
| 15 | AutoFitV2 | autofit | 334.00 | 1,400.44 | +248.1% |
| 16 | AutoFitV2E | autofit | 334.00 | 1,400.44 | +248.1% |
| 17 | SeasonalNaive | ml_tabular | 381.12 | 1,396.19 | +297.2% |

*31 more models omitted (all >+310% vs #1)*

## Leaderboard: `is_funded` (354 records, 44 models)

| Rank | Model | Category | Best MAE | Best RMSE | vs #1 |
|------|-------|----------|----------|-----------|-------|
| 1 | **ExtraTrees** | ml_tabular | 0.065 | 0.18 | -- |
| 2 | AutoFitV2E | autofit | 0.084 | 0.19 | +28.6% |
| 3 | AutoFitV3E | autofit | 0.084 | 0.18 | +28.9% |
| 4 | AutoFitV2 | autofit | 0.086 | 0.18 | +32.2% |
| 5 | AutoFitV3Max | autofit | 0.088 | 0.19 | +34.9% |
| 6 | AutoFitV3 | autofit | 0.088 | 0.19 | +34.9% |
| 7 | RandomForest | ml_tabular | 0.090 | 0.25 | +37.4% |
| 8 | AutoFitV1 | autofit | 0.094 | 0.20 | +44.4% |
| 9 | XGBoost | ml_tabular | 0.096 | 0.20 | +47.3% |
| 10 | HistGBT | ml_tabular | 0.097 | 0.19 | +47.8% |
| 11 | LightGBM | ml_tabular | 0.097 | 0.20 | +48.0% |
| 12 | SeasonalNaive | ml_tabular | 0.098 | 0.31 | +49.3% |
| 13 | CatBoost | ml_tabular | 0.105 | 0.21 | +60.9% |
| 14 | KNN | ml_tabular | 0.109 | 0.27 | +66.2% |
| 15 | LogisticReg | ml_tabular | 0.154 | 0.39 | +135.5% |

*29 more models omitted (all >+146% vs #1)*

---

## Key Findings

### 1. Tree Ensembles Dominate All Targets
- **RandomForest** is #1 on two of three targets (funding_raised_usd, investors_count)
- **ExtraTrees** is #1 on is_funded and #2 on investors_count
- Tree ensembles (RF, ET, XGBoost, LightGBM) occupy the top 5-11 ranks across all targets
- Deep learning, transformers, and statistical models are consistently outperformed

### 2. AutoFit Model Selection Issue (Phase 1)
- **AutoFitV1/V3/V3Max** tie with RandomForest on funding_raised_usd (correctly selected RF)
- **AutoFitV3/V3Max** selected ExtraTrees on investors_count (+18.7% vs best), not RandomForest
- **Root cause**: Single 80/20 validation split led to ExtraTrees winning on validation but RF generalizing better to test set
- **Fix applied for Phase 2**: 5-fold expanding-window temporal CV with stability penalty

### 3. Foundation Models Underperform on Tabular Tasks
- **Moirai** ranks #15 on funding_raised_usd (+72.7% vs RF), #10 on investors_count
- **Chronos** ranks #35 on funding_raised_usd (>1800% gap)
- Pre-trained foundation models are not competitive on structured tabular data

### 4. Deep/Transformer Models Show Near-Constant Predictions
- Most transformer models (PatchTST, iTransformer, TimesNet, etc.) show near-identical MAE on investors_count (~401)
- This suggests they converge to predicting the mean — likely due to short training or dataset-model mismatch
- On funding_raised_usd, the gap is even more extreme (>1700% vs #1)

### 5. EDGAR Ablation Has Minor Impact
- `core_only` vs `core_edgar` shows marginal differences for most models
- Tree models can utilize EDGAR features; deep/transformer models ignore them

---

## Phase 1 Issues and Phase 2 Fixes

### Issues Identified and Resolved

| # | Issue | Severity | File | Status |
|---|-------|----------|------|--------|
| 1 | Single 80/20 split causes unstable model selection | CRITICAL | autofit_wrapper.py | FIXED |
| 2 | `del locals()[_v]` is a no-op (memory leak) | HIGH | benchmark_shard.py | FIXED |
| 3 | `max_rows` truncation before temporal split | HIGH | benchmark_shard.py | FIXED |
| 4 | SklearnModelWrapper missing feature names | MEDIUM | base.py | FIXED |
| 5 | SubsampledSklearnWrapper wrong log count | LOW | traditional_ml.py | FIXED |
| 6 | V3 had separate 2-fold CV, inconsistent with V1/V2 | MEDIUM | autofit_wrapper.py | FIXED |
| 7 | `_MODEL_TIMEOUT_SECONDS` never enforced | LOW | benchmark_shard.py | Noted |
| 8 | `GradientBoostingWrapper` dead code in base.py | LOW | base.py | Noted |

### Phase 2 Methodology Changes

1. **5-fold expanding-window temporal CV** (was: single 80/20 split)
   - Cut points at [50%, 60%, 70%, 80%, 90%] of data
   - Models must succeed on 2+ folds or are rejected

2. **Stability penalty** (new)
   - `adj_MAE = mean_MAE * (1 + 0.25 * CV(MAE))`
   - Penalizes high-variance models that win on one fold but fail on others

3. **Unified evaluation** (was: V1/V2/V2E used separate logic, V3 had its own 2-fold)
   - All 6 AutoFit variants now share `_temporal_kfold_evaluate_all()`

4. **Memory management fix**
   - Explicit `del model; del X_train, y_train` instead of no-op `del locals()[x]`

5. **max_rows safety**
   - Truncation now applies only to training set (`.tail()` for recency), never to test set

---

## SLURM Job Summary

| Job ID | Name | Runtime | ExitCode |
|--------|------|---------|----------|
| 5173849 | afv3_e_on | 1d 16h 48m | 0 |
| 5173850 | afv3_e_ed | 1d 16h 58m | 0 |
| 5173851 | afstk_on | 23h 05m | 0 |
| 5173852 | afstk_ed | 23h 05m | 0 |
| 5173853 | afv12_on | 23h 26m | 0 |
| 5173854 | afv12_ed | 23h 24m | 0 |
| 5175268 | mlt_on_1 | 6h 22m | 0 |
| 5175269 | mlt_ed_1 | 5h 28m | 0 |
| 5175270 | mlt_on_2 | 5h 07m | 0 |
| 5175271 | mlt_ed_2 | 4h 12m | 0 |
| 5175272 | mlt_on_3 | 5h 28m | 0 |
| 5175273 | mlt_ed_3 | 4h 39m | 0 |

---

_Last updated: 2026-02-11_
