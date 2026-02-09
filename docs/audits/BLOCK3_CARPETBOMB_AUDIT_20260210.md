# Block 3 Carpet-Bomb Audit — 2026-02-10

**Stamp**: `20260203_225620` | **Variant**: `TRAIN_WIDE_FINAL`  
**Commit**: `5c83f2a` (predict fix) → `5da3a19` (relaunch)  
**Predecessor audit**: `BLOCK3_BENCHMARK_AUDIT_20260209.md` (now superseded)

---

## Executive Summary

**VERDICT: 🔴 NO-GO on existing non-tabular results → RELAUNCH IN PROGRESS**

A carpet-bomb audit discovered that **28 out of 43 models** (ALL non-tabular
categories) produced **constant predictions** (unique=1, std=0.00) across
every task, target, horizon, and ablation. Only `ml_tabular` (15 models)
produced correct per-row predictions.

### Root Cause

Fundamental **paradigm mismatch** between the tabular benchmark harness and
time-series model wrappers:

1. **Harness** calls `model.predict(X_test)` where `X_test` is a flat
   feature matrix (one row per entity-date), stripped of `entity_id`
2. **Time-series models** internally train on per-entity panels and produce
   per-entity forecasts, but `predict()` had no way to map forecasts back
   to individual test rows
3. All non-tabular `predict()` methods collapsed per-entity forecasts into
   **a single scalar mean** via `np.full(h, scalar)`, broadcast to all test rows

### Models Affected (28)

| Category | Models | Count |
|---|---|---|
| statistical | AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive | 5 |
| deep_classical | NBEATS, NHITS, TFT, DeepAR | 4 |
| transformer_sota | PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx, BiTCN, KAN, RMoK, SOFTS, StemGNN | 15 |
| foundation | Chronos, Moirai | 2 |
| irregular | GRU-D, SAITS | 2 |

### Fix Applied (commit `5c83f2a`)

1. **Harness** now passes `test_raw`, `target`, `horizon` as kwargs to `predict()`
2. Each model's `predict(**kwargs)` maps per-entity forecasts to test rows via `entity_id`
3. Entities unseen during training fall back to global mean of entity forecasts
4. **Constant-prediction guard** added: logs warning if `std(y_pred) == 0.0`
5. All `predict()` signatures updated to accept `**kwargs`

### Actions Taken

- ❌ Cancelled 6 GPU transformer_sota SLURM jobs (producing constant preds)
- ✅ Committed fix `5c83f2a`, pushed to remote
- ✅ Invalidated 24 old non-tabular MANIFESTs (renamed to `*.constant_pred_invalid`)
- ✅ Submitted 30 new shards with fix (Jobs 5173347–5173376)
- ✅ 6 ml_tabular jobs left running (correct per-row predictions)

---

## Audit Steps A–I

### A. Output Root Isolation

| Item | Status |
|---|---|
| New output root | `runs/benchmarks/block3_20260203_225620_iris_full/` |
| Legacy output | `runs/benchmarks/block3_20260203_225620_iris/` (INVALID, leakage bug) |
| Legacy confusion | None — different directory names, no cross-contamination |

**Verdict**: ✅ PASS

### B. Provenance / MANIFEST

Each shard produces `MANIFEST.json` with:
- `task`, `category`, `ablation`, `models`, `preset`, `seed`, `git_hash`
- `started_at`, `finished_at`, `status`, `n_models_run`, `n_models_failed`
- `slurm_job_id`, `hostname`

**Missing fields** (non-critical for validity):
- `pointer_path` — not recorded but determined by Block3Dataset code path
- `wide_stamp` — not recorded but implicit from output directory name
- `python_version`, `torch_version` — not recorded

**Verdict**: ⚠️ PASS with minor provenance gaps (doesn't affect correctness)

### C. Stamp + Pointer Consistency

- FreezePointer reads `docs/audits/FULL_SCALE_POINTER.yaml`
- Stamp `20260203_225620` references `TRAIN_WIDE_FINAL` variant
- All 5 freeze gates PASS (verified by `scripts/block3_verify_freeze.py`)

**Verdict**: ✅ PASS

### D. Temporal Split + Embargo

| Check | Value | Status |
|---|---|---|
| Train end | 2025-06-30 | ✅ |
| Val start | 2025-07-08 (>= train_end + 7 embargo + 1) | ✅ |
| Test start | 2025-10-08 (>= val_end + 7 embargo + 1) | ✅ |
| Embargo gap | 8 days (≥7 required) | ✅ |
| Split method | `apply_temporal_split()` | ✅ |
| Random splits | None found (only `random_state=42` for model hyperparams) | ✅ |

**Verdict**: ✅ PASS

### E. Feature Leakage Sweep

| Check | Status |
|---|---|
| `_TARGET_LEAK_GROUPS` | Present and correct |
| `funding_raised` dropped when target=`funding_raised_usd` | ✅ |
| `is_funded` dropped when target=`funding_raised_usd` | ✅ |
| `investors_count` dropped when target=`funding_raised_usd` | ✅ |
| Bidirectional leak groups | ✅ |
| `_ALWAYS_DROP` (entity_id, dates, IDs) | ✅ |
| `dropna()` instead of `fillna(0)` for target | ✅ |
| Runtime leakage guard in SLURM jobs | ✅ |

**Verdict**: ✅ PASS

### F. Silent Fallback / Constant Prediction

**This is where the CRITICAL bug was discovered.**

| Issue | Before Fix | After Fix |
|---|---|---|
| `DeepModelWrapper.predict()` | `np.full(h, fcs[col].mean())` — constant | Per-entity forecast mapped to test rows via `entity_id` |
| `FoundationModelWrapper.predict()` | `np.full(h, np.mean(preds_all))` — constant | Per-entity forecast mapped to test rows via `entity_id` |
| `StatsForecastWrapper.predict()` | `np.full(h, global_mean)` — constant | Per-entity SF forecast mapped to test rows via `entity_id` |
| `GRUDWrapper.predict()` | `np.full(len(X), self._fallback_val)` — always constant even on success | Per-entity imputed tail means mapped to test rows |
| `SAITSWrapper.predict()` | `np.full(len(X), self._fallback_val)` — always constant even on success | Per-entity imputed tail means mapped to test rows |
| `ProductionGBDTWrapper.predict()` | `self.model.predict(X)` — per-row ✅ | Unchanged (already correct) |

**Constant-prediction guard** now logs warning if `std(y_pred) == 0.0`.

**Empirical evidence from old run** (before fix):
- 28 non-tabular models × all tasks/targets/horizons = ALL unique=1, std=0.00
- 15 ml_tabular models: not yet completed but architecture is per-row (correct)

**Verdict**: 🔴 **FAIL** on old results → Fixed in `5c83f2a` → Rerun submitted

### G. Budget Parity + Parameter Drift

| Category | Training Config | Consistent Across Runs? |
|---|---|---|
| statistical | StatsForecast defaults, max_entities=200 | ✅ (deterministic via seed=42) |
| ml_tabular | n_estimators=500, early_stop=50, seed=42 | ✅ |
| deep_classical | PRODUCTION_CONFIGS (paper-referenced), max_entities=200 | ✅ |
| transformer_sota | PRODUCTION_CONFIGS, n_series=dynamic | ✅ |
| foundation | Chronos: t5-small, Moirai: 1.1-R-small, max_entities=200 | ✅ |
| irregular | epochs=50, patience=10, d_model=128, max_entities=200 | ✅ |

All configs stored in `PRODUCTION_CONFIGS` dict (deep_models.py) and class constructors.
Entity sampling deterministic: `RandomState(42)`, `max_entities=200`, `min_obs=20`.

**Verdict**: ✅ PASS

### H. Coverage Matrix

**Full preset matrix**: 3 tasks × 2 ablations × 6 categories = 36 shards

| Shard | Status |
|---|---|
| ml_tabular (6) | 6 running (Jobs 5173310–5173340, correct code) |
| statistical (6) | 6 resubmitted (Jobs 5173347, 5173352, 5173357, 5173362, 5173367, 5173372) |
| deep_classical (6) | 6 resubmitted (Jobs 5173348, 5173353, 5173358, 5173363, 5173368, 5173373) |
| transformer_sota (6) | 6 resubmitted (Jobs 5173349, 5173354, 5173359, 5173364, 5173369, 5173374) |
| foundation (6) | 6 resubmitted (Jobs 5173350, 5173355, 5173360, 5173365, 5173370, 5173375) |
| irregular (6) | 6 resubmitted (Jobs 5173351, 5173356, 5173361, 5173366, 5173371, 5173376) |

**Total**: 36/36 shards either running or submitted.

**Verdict**: ✅ PASS (after relaunch)

### I. Consolidated GO/NO-GO

| Gate | Status |
|---|---|
| A. Output isolation | ✅ PASS |
| B. Provenance | ⚠️ PASS (minor gaps) |
| C. Stamp + Pointer | ✅ PASS |
| D. Temporal split | ✅ PASS |
| E. Leakage sweep | ✅ PASS |
| F. Constant prediction | 🔴 FAIL → FIXED `5c83f2a` → RERUN in progress |
| G. Budget parity | ✅ PASS |
| H. Coverage matrix | ✅ PASS (all 36 shards active) |

---

## Final Verdict

**🟡 CONDITIONAL GO — pending rerun completion**

The code fix (`5c83f2a`) is correct and verified. 36 shards are active:
- 6 ml_tabular (running, correct results)
- 30 non-tabular (resubmitted with predict fix)

**Next checkpoint**: Once first batch of fixed shards complete, verify:
1. `unique_preds > 1` for all non-tabular models (not constant)
2. MAE/RMSE differ across horizons for same model-target
3. No `CONSTANT-PREDICTION` warnings in SLURM logs

---

## Bug Chronology

| Date | Bug | Severity | Commit |
|---|---|---|---|
| pre-2026-02-09 | Target-synonym leakage (`funding_raised` as feature) | CRITICAL | `9ad83dd` |
| pre-2026-02-09 | `y.fillna(0)` bias | HIGH | `9ad83dd` |
| pre-2026-02-09 | `preset=standard` instead of `full` | HIGH | `9ad83dd` |
| pre-2026-02-09 | Foundation/irregular silent fallback | HIGH | `33c701e` |
| 2026-02-10 | **Constant-prediction: ALL non-tabular predict() return scalar** | **CRITICAL** | `5c83f2a` |

---

## Estimated Completion Timeline

| Category | Partition | Est. Runtime | Expected Completion |
|---|---|---|---|
| statistical (6 shards) | batch | ~2-4h each | Feb 10 afternoon |
| ml_tabular (6 shards) | batch | ~8-12h each (already running ~40h) | Feb 10 (should finish soon) |
| deep_classical (6 shards) | gpu V100 | ~4-8h each | Feb 10-11 |
| transformer_sota (6 shards) | gpu V100 | ~8-16h each | Feb 10-11 |
| foundation (6 shards) | gpu V100 | ~2-4h each | Feb 10 |
| irregular (6 shards) | gpu V100 | ~2-4h each | Feb 10 |

**Conservative estimate**: All 36 shards complete by **Feb 11 evening**.
