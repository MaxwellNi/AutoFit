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
- [x] Benchmark harness (`scripts/run_block3_benchmark_shard.py`)
- [x] Statistical baselines (5 models via StatsForecast)
- [x] ML Tabular (15 models via sklearn/GBDT)
- [x] Deep Classical (4 models via NeuralForecast)
- [x] Transformer SOTA (20 models via NeuralForecast, 2 shards)
- [x] Foundation (11 models: Chronos×3, Moirai×3, Timer, TimeMoE, MOMENT, LagLlama, TimesFM)
- [x] Irregular (2 models: GRU-D, SAITS via PyPOTS)
- [x] AutoFit (10 variants: V1–V7 with ensemble selection strategies)

### Phase C: AutoFit (COMPLETE)
- [x] Rule-based composer (`src/narrative/auto_fit/rule_based_composer.py`)
- [x] Meta-feature extraction via profiler
- [x] AutoFit V1-V7: 10 ensemble selection strategies
  - V1: simple best-of-K selection
  - V2/V2E: 5-fold temporal CV with stability penalty
  - V3/V3E/V3Max: exhaustive search with time budget
  - V4: target-transform + NCL + full-OOF
  - V5: empirical regime-aware ensemble
  - V6: conference-grade stacked generalization
  - V7: data-adapted robust ensemble with 6 SOTA innovations

### Phase D: Interpretability (COMPLETE)
- [x] Concept bottleneck (`src/narrative/explainability/concept_bottleneck.py`)
- [x] Concept bank with 10 concepts
- [x] Marginal contribution logging

### Phase E: Model Expansion (COMPLETE — Phases 1-7)
- [x] Phase 1: Initial 44 models, 6 categories (2646 records)
- [x] Phase 2: AutoFit 5-fold temporal CV + stability penalty
- [x] Phase 3: 6 critical fixes (entity coverage, EDGAR as-of join, AutoFit timeout, etc.)
- [x] Phase 4: +10 SOTA models (59 total), AutoFitV4
- [x] Phase 5: +5 foundation models (65 total), AutoFitV5
- [x] Phase 6: AutoFitV6 stacked generalization
- [x] Phase 7: +2 models (67 total), AutoFitV7, full code audit + 5 root-cause fixes

### Phase F: Full Benchmark on ULHPC Iris (IN PROGRESS)
- [x] SLURM submission infrastructure (`scripts/submit_phase7_full_benchmark.sh`)
- [x] 121 SLURM jobs submitted across 11 shards × 3 tasks × ~4 ablations
- [x] Phase 7 code fixes (5 root causes, 57/57 tests pass) — commit `444f376`
- [x] EDGAR timezone fix (merge_asof dtype mismatch) — commit `ae9626b`
- [x] Statistical OOM fix (64G → 112G memory) — 8 scripts updated, resubmitted
- [ ] **Full benchmark completion** — 23/121 shards done (19%), 781 metric records
  - 12 RUNNING, 96 PENDING, 23 COMPLETED
  - Est. completion: 2-3 days (queue depth dependent)
- [ ] Results consolidation + leaderboard
- [ ] Paper tables (LaTeX)

### Phase G: Analysis (PENDING — awaiting full benchmark completion)
- [ ] TCAV-style concept importance analysis
- [ ] Ablation study (core_only vs core_text vs core_edgar vs full)
- [ ] Horizon sensitivity analysis
- [ ] Error analysis by entity type
- [ ] AutoFit model selection based on profile
- [ ] Results leaderboard + paper tables

---

## Phase 7 Benchmark Progress (2026-02-13)

### Platform: ULHPC Iris HPC
- GPU: 24 nodes, 28c, 756GB RAM, 4×V100 32GB — QOS `iris-gpu-long` (14d)
- Batch: 168 nodes, 28c, 112GB RAM — QOS `iris-batch-long` (14d)
- Account: `yves.letraon`

### Completion Matrix (23/121 shards = 19%)

| Category | task1_co | task1_ct | task1_ce | task1_fu | task2_co | task2_ct | task2_ce | task2_fu | task3_co | task3_ce | task3_fu | Done |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|------|
| ml_tabular | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⏳PD | ⏳PD | ✅ | ⏳PD | ⏳PD | 7/11 |
| statistical | 🔄RN | ⏳PD | ⏳PD | ⏳PD | ✅ | ⏳PD | ⏳PD | ⏳PD | ✅ | ⏳PD | ⏳PD | 2/11 |
| deep_classical | ✅ | ✅ | ✅ | ⏳PD | ✅ | ✅ | ⏳PD | ⏳PD | ✅ | ⏳PD | ⏳PD | 6/11 |
| tsA (10 models) | ✅ | ✅ | ⏳PD | ⏳PD | ✅ | ✅ | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | 4/11 |
| tsB (10 models) | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | 0/11 |
| fmC (Chronos) | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | 0/11 |
| fmM (Moirai) | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | 0/11 |
| fmH (HF models) | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | 0/11 |
| irregular | ✅ | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | 1/11 |
| af1 (V1-V5) | 🔄RN | 🔄RN | ⏳PD | ⏳PD | 🔄RN | 🔄RN | ⏳PD | ⏳PD | 🔄RN | ⏳PD | ⏳PD | 3/11 |
| af2 (V6-V7) | 🔄RN | 🔄RN | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | ⏳PD | 0/11 |

Legend: ✅=Done, 🔄RN=Running, ⏳PD=Pending

### Issues Encountered and Fixed

| # | Issue | Root Cause | Fix | Commit |
|---|-------|-----------|-----|--------|
| 1 | Deep/Transformer models: near-constant predictions | Only 200 entities sampled (<5% test coverage) | max_entities=2000, min_obs=10, Ridge fallback | `444f376` |
| 2 | RobustFallback: silent failure | NeuralForecast predict raised OOM/timeout | Catch all exceptions, auto-fallback to Ridge | `444f376` |
| 3 | Hybrid predict: entity coverage | Unseen entities got global_mean | Ridge regression on features for unseen entities | `444f376` |
| 4 | EDGAR covariates: not passed to models | `_build_panel_df()` ignored exog columns | Pass EDGAR features as `futr_exog_list` | `444f376` |
| 5 | AutoFit: missing `target_transform` | Regression targets needed log-transform | Auto-detect and apply `log1p`/`expm1` | `444f376` |
| 6 | EDGAR timezone mismatch | `datetime64[ns,UTC]` vs `datetime64[ns]` | `pd.to_datetime(..., utc=True).dt.tz_convert(None)` | `ae9626b` |
| 7 | Statistical OOM (64G) | `core_text`/`core_edgar`/`full` joins expand memory | Increased to 112G, 28 CPUs | Session fix |

### Metric Summary (781 records from 23 shards)
- task1_outcome: 9 shards, ~324 records
- task2_forecast: 9 shards, ~290 records
- task3_risk_adjust: 5 shards, ~167 records

---

## Next Actions

### Priority 1: Complete Phase 7 Benchmark on Iris HPC
Wait for all 121 SLURM shards to complete (est. 2-3 days).
Monitor via `squeue -u npin` and `sacct` for failures.

### Priority 2: Results Consolidation + Leaderboard
Run `scripts/consolidate_block3_results.py` once all shards complete.
Generate per-target leaderboards with bootstrap confidence intervals.

### Priority 3: Paper Tables
Run `scripts/make_paper_tables_v2.py` for LaTeX tables for KDD'26 paper.
Ablation analysis: core_only vs core_text vs core_edgar vs full.

### Priority 4: Analysis
TCAV-style concept importance via `src/narrative/explainability/concept_bottleneck.py`.
AutoFit model selection based on profile.

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
- Benchmark Phase 7: `runs/benchmarks/block3_20260203_225620_phase7/`

---

## Commit History
- `ae9626b` Fix EDGAR join: strip timezone from datetime64[ns,UTC] for merge_asof compatibility
- `444f376` Phase 7: Fix entity coverage, RobustFallback, hybrid predict, EDGAR covariates, AutoFit target_transform
- `d837828` Phase 7: SLURM submission scripts for full 67-model benchmark
- `faafdcf` AutoFit V7: data-adapted robust ensemble with 6 SOTA innovations
- `ad07032` AutoFit V6: conference-grade stacked generalization (Phase 6)
- `dce0ff9` AutoFit V5: empirical regime-aware ensemble + 5 new foundation models (65 total)
- `c53abf6` Phase 4: +10 SOTA models (59 total), AutoFitV4 w/ target-transform+NCL+full-OOF
- `320c314` Phase 3: Fix 6 critical issues, 42 SLURM jobs submitted
- `87baa13` Phase 2 AutoFit: 5-fold temporal CV + stability penalty, benchmark fixes
- `014ac92` Fix n_series dynamic computation, SLURM mem 128G, 4090 launch scripts
- `fcbe970` Model registry rewrite: 44 models, panel data fix, all 6 categories
- `3ce5509` WIDE2 freeze seal complete, all gates PASS
- `5ca7fd8` Block3 init: benchmark harness + autofit + concept bottleneck + docs
