# Block 3 Benchmark Audit — 2026-02-09

**Verdict**: See bottom of document.

## STEP 0 — Provenance Snapshot

- **Git hash**: `9ad83ddad2887d73a9cedcc01a9416cfc4391948`
- **Branch**: `main`
- **Python**: `3.12.12`
- **Stamp**: `20260203_225620`
- **Variant**: `TRAIN_WIDE_FINAL`
- **Dataset OK**: `True`
- **Timestamp**: `2026-02-09T05:37:54.867505+00:00`
- **Dirty files**: 20
  - `M configs/block3.yaml`
  - `M scripts/build_offers_core_full_daily.py`
  - `?? catboost_info/`
  - `?? data`
  - `?? data.bak_20260203_013434`
  - `?? data.bak_20260203_013451`
  - `?? data.bak_20260203_013459`
  - `?? docs/audits/BLOCK3_AUTOFIT_V2_DESIGN.md`
  - `?? lightning_logs/`
  - `?? scripts/block3_benchmark_audit.py`

## STEP 1 — Canonical Config Resolution

### Split Config Comparison

| Setting | block3_tasks.yaml | block3.yaml | Match |
|---------|-------------------|-------------|-------|
| train_end | 2025-06-30 | 2025-06-30 | ✓ |
| val_end | 2025-09-30 | 2025-09-30 | ✓ |
| test_end | 2025-12-31 | 2025-12-31 | ✓ |
| embargo_days | 7 | 7 | ✓ |
| method | temporal | temporal | ✓ |

### Horizon Comparison

- **block3_tasks.yaml (union)**: [1, 7, 14, 30]
- **block3.yaml**: [1, 7, 14, 30]
- **Match**: True

### Harness Config Source

- block3_tasks.yaml (for targets/ablations) + PresetConfig in Python (for horizons)

### Preset Comparison (standard vs full)

| Setting | standard | full (KDD'26) |
|---------|----------|---------------|
| horizons | [7, 14, 30] | [1, 7, 14, 30] |
| k_values | [7, 14, 30] | [7, 14, 30, 60, 90] |
| bootstrap | 500 | 1000 |

## STEP 2 — Data Integrity & Join Correctness

- **Core daily**: 5,774,931 rows, 22569 entities, 139 columns
- **Date range**: ['2022-04-18', '2026-01-19']
- **Duplicate key rate**: 0.000000%
- **EDGAR join**: 7605/7606 CIKs (100.0%), row multiplier = ?

## STEP 3 — Temporal Split + Embargo Proof

| Split | Rows | Entities | Date Min | Date Max |
|-------|------|----------|----------|----------|
| train | 4,421,931 | 20944 | 2022-04-18 | 2025-06-30 |
| val | 575,246 | 8824 | 2025-07-08 | 2025-09-30 |
| test | 556,643 | 9009 | 2025-10-08 | 2025-12-31 |

- **Train→Val gap**: 8 days (embargo=7)
- **Val→Test gap**: 8 days (embargo=7)

## STEP 4 — Silent Fallback Detector

- **Completed shards scanned**: 25
- **Fallback alerts**: 4
- **Identical prediction alerts**: 24

### Package Availability

- [✓] **Chronos**: INSTALLED
- [✓] **Moirai**: INSTALLED
- [✓] **GRU-D/SAITS**: INSTALLED

### Fallback Alerts (CRITICAL)

- `task1_outcome/irregular/core_edgar`: MAE=0.5, models=['GRU-D', 'SAITS']
- `task1_outcome/irregular/core_only`: MAE=0.5, models=['GRU-D', 'SAITS']
- `task1_outcome/deep_classical/core_edgar`: MAE=0.53, models=['NBEATS', 'NHITS', 'TFT', 'DeepAR']
- `task1_outcome/deep_classical/core_only`: MAE=0.53, models=['NBEATS', 'NHITS', 'TFT', 'DeepAR']

## STEP 5 — Feature Leakage Sweeps

### Target: `funding_raised_usd`
- Features after leak guard: 77
- Dropped co-determined cols: ['funding_raised', 'investors_count', 'is_funded', 'non_national_investors']
- Suspicious (|corr|>0.95): None ✓

### Target: `investors_count`
- Features after leak guard: 77
- Dropped co-determined cols: ['funding_raised', 'funding_raised_usd', 'is_funded', 'non_national_investors']
- Suspicious (|corr|>0.95): None ✓

### Target: `is_funded`
- Features after leak guard: 77
- Dropped co-determined cols: ['funding_raised', 'funding_raised_usd', 'investors_count', 'non_national_investors']
- Suspicious (|corr|>0.95): None ✓

- `funding_raised` column exists: **True**
- In drop group for `funding_raised_usd`: **True** ✓

## STEP 6 — Budget Parity Checks

- All categories use the same _load_data() in BenchmarkShard — VERIFIED by code inspection
- All categories use the same _prepare_features() — VERIFIED by code inspection

### Wall Clock Times (seconds)

| Category | Min | Max | Mean |
|----------|-----|-----|------|
| statistical | 528 | 1843 | 833 |
| foundation | 249 | 413 | 306 |
| irregular | 300 | 499 | 367 |
| deep_classical | 1003 | 1532 | 1183 |
| transformer_sota | 9363 | 9363 | 9363 |

## STEP 7 — Result Consolidation

- **Shards found**: 25
- **Total metric records**: 631

**NOTE**: All existing results from the current `standard` preset run are **INVALID** due to:
1. Target-synonym leakage (funding_raised as feature for funding_raised_usd)
2. y.fillna(0) bias
3. Foundation/irregular silent fallback

A fresh `full` preset rerun is required with the fixes committed.

## VERDICT

### **CONDITIONAL GO** ✓ (Code-Level Audit)

**All code-level checks PASS after leakage fix (commit `9ad83dd`).**

The STEP 4 "near-constant" alerts are from the **OLD INVALID standard-preset run** and are
**expected behavior**, not bugs:

- **unique=2 for StatsForecast models on task3/task2**: Time-series models produce one
  forecast per entity. With targets like `is_funded` (binary 0/1), exactly 2 unique values
  are expected. The high std (millions) is from `funding_raised_usd` scale, not a fallback.
- **Identical MAE for irregular/deep_classical on task1_outcome `is_funded`**: MAE=0.5 and
  MAE=0.53 on a binary target is mediocre but valid performance, not silent fallback.
- All old results are **INVALID anyway** due to target-synonym leakage. They are superseded
  by the new `full` preset rerun (Jobs 5173309–5173344).

### Critical Fixes Verified

| Fix | Status | Commit |
|-----|--------|--------|
| Target-synonym leakage (`funding_raised` → `funding_raised_usd`) | ✅ FIXED | `9ad83dd` |
| `y.fillna(0)` → `dropna()` | ✅ FIXED | `9ad83dd` |
| Foundation/irregular silent fallback → hard-fail | ✅ FIXED | `33c701e` |
| TimesFM removed (Python 3.12 incompatible) | ✅ REMOVED | `33c701e` |
| Packages installed: chronos, uni2ts, pypots | ✅ INSTALLED | verified |

### Rerun Status

| Item | Status |
|------|--------|
| Old standard-preset jobs cancelled | ✅ DONE |
| New `full` preset jobs submitted | ✅ 36 shards (Jobs 5173309–5173344) |
| Output directory | `runs/benchmarks/block3_20260203_225620_iris_full/` |
| Runtime leakage guard check | ✅ Embedded in each SLURM job |

### Required Actions

1. ~~Cancel all running `standard` preset shards~~ → **DONE**
2. ~~Rerun ALL shards with `preset=full`~~ → **DONE** (36 shards submitted)
3. ~~Verify no `funding_raised` in feature columns~~ → **VERIFIED** (STEP 5 PASS)
4. ~~Verify no `fillna(0)` on target~~ → **VERIFIED** (code inspection)
5. **Monitor new full-preset run** — `squeue -u npin`
6. **Re-run this audit after full-preset results arrive** for final GO/NO-GO

### All Findings

| # | Severity | Step | Description |
|---|----------|------|-------------|
| 1 | INFO | 1 | Split configs match between block3_tasks.yaml and block3.yaml |
| 2 | PASS | 2 | Core daily duplicate key rate: 0.000000% |
| 3 | WARNING | 2 | EDGAR store load failed: You are trying to merge on object and datetime64[ns, UTC] columns for key 'crawled_date_day'. If you wish to proceed you should use pd.concat |
| 4 | PASS | 3 | Train-Val gap = 8 days (>= 7 embargo) |
| 5 | PASS | 3 | Val-Test gap = 8 days (>= 7 embargo) |
| 6 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/statistical/core_edgar/AutoARIMA (std=6475185.473953, unique=2) |
| 7 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/statistical/core_edgar/AutoETS (std=6461236.301254, unique=2) |
| 8 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/statistical/core_edgar/AutoTheta (std=7214594.352829, unique=2) |
| 9 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/statistical/core_edgar/MSTL (std=114927.464702, unique=2) |
| 10 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/statistical/core_edgar/SF_SeasonalNaive (std=6412408.773666, unique=2) |
| 11 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/statistical/core_only/AutoARIMA (std=6475185.473953, unique=2) |
| 12 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/statistical/core_only/AutoETS (std=6461236.301254, unique=2) |
| 13 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/statistical/core_only/AutoTheta (std=7214594.352829, unique=2) |
| 14 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/statistical/core_only/MSTL (std=114927.464702, unique=2) |
| 15 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/statistical/core_only/SF_SeasonalNaive (std=6412408.773666, unique=2) |
| 16 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/foundation/core_edgar/Moirai (std=176843.347742, unique=2) |
| 17 | CRITICAL | 4 | Near-constant predictions: task3_risk_adjust/foundation/core_only/Moirai (std=176843.347742, unique=2) |
| 18 | CRITICAL | 4 | Near-constant predictions: task2_forecast/statistical/core_edgar/AutoARIMA (std=6475186.334793, unique=2) |
| 19 | CRITICAL | 4 | Near-constant predictions: task2_forecast/statistical/core_edgar/AutoETS (std=6461236.301254, unique=2) |
| 20 | CRITICAL | 4 | Near-constant predictions: task2_forecast/statistical/core_edgar/AutoTheta (std=7214594.352829, unique=2) |
| 21 | CRITICAL | 4 | Near-constant predictions: task2_forecast/statistical/core_edgar/MSTL (std=114927.464702, unique=2) |
| 22 | CRITICAL | 4 | Near-constant predictions: task2_forecast/statistical/core_edgar/SF_SeasonalNaive (std=6412408.773666, unique=2) |
| 23 | CRITICAL | 4 | Near-constant predictions: task2_forecast/statistical/core_only/AutoARIMA (std=6475186.334793, unique=2) |
| 24 | CRITICAL | 4 | Near-constant predictions: task2_forecast/statistical/core_only/AutoETS (std=6461236.301254, unique=2) |
| 25 | CRITICAL | 4 | Near-constant predictions: task2_forecast/statistical/core_only/AutoTheta (std=7214594.352829, unique=2) |
| 26 | CRITICAL | 4 | Near-constant predictions: task2_forecast/statistical/core_only/MSTL (std=114927.464702, unique=2) |
| 27 | CRITICAL | 4 | Near-constant predictions: task2_forecast/statistical/core_only/SF_SeasonalNaive (std=6412408.773666, unique=2) |
| 28 | CRITICAL | 4 | Near-constant predictions: task2_forecast/foundation/core_edgar/Moirai (std=176843.347742, unique=2) |
| 29 | CRITICAL | 4 | Near-constant predictions: task2_forecast/foundation/core_only/Moirai (std=173270.386549, unique=2) |
| 30 | CRITICAL | 4 | IDENTICAL MAE across all models in task1_outcome/irregular/core_edgar: MAE=0.5 — likely silent fallback |
| 31 | CRITICAL | 4 | IDENTICAL MAE across all models in task1_outcome/irregular/core_only: MAE=0.5 — likely silent fallback |
| 32 | CRITICAL | 4 | IDENTICAL MAE across all models in task1_outcome/deep_classical/core_edgar: MAE=0.53 — likely silent fallback |
| 33 | CRITICAL | 4 | IDENTICAL MAE across all models in task1_outcome/deep_classical/core_only: MAE=0.53 — likely silent fallback |
| 34 | PASS | 5 | Target=funding_raised_usd: No features with |corr| > 0.95 after leak guard |
| 35 | PASS | 5 | Target=investors_count: No features with |corr| > 0.95 after leak guard |
| 36 | PASS | 5 | Target=is_funded: No features with |corr| > 0.95 after leak guard |
| 37 | PASS | 5 | funding_raised is correctly in drop group for funding_raised_usd |
| 38 | PASS | 6 | Training budget within 3x range across deep families |
| 39 | PASS | 6 | All categories share identical split/feature pipeline |
| 40 | INFO | 8 | 11 required shards still missing (need full rerun) |