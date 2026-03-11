# Agent Context

## Mission
Block 3 modeling on the finalized WIDE2 freeze (`TRAIN_WIDE_FINAL`). The freeze is complete with all gates PASS. No feature backfilling is required.

## Current State (as of 2026-03-08)
- **WIDE2 Freeze: COMPLETE** (stamp `20260203_225620`)
- All 5 gates PASS:
  - `pointer_valid`: PASS
  - `column_manifest`: PASS
  - `raw_cardinality_coverage`: PASS
  - `freeze_candidates`: PASS
  - `offer_day_coverage_exact`: PASS
- Block 3 entry verified via `scripts/block3_verify_freeze.py`

### Phase 9 Fair Benchmark — ULHPC Iris HPC (IN PROGRESS)

Phase 9 is a complete re-benchmark after fixing 4 critical experimental bugs
(TSLib per-entity prediction, foundation prediction_length, Moirai entity cap,
ml_tabular single-horizon). All prior Phase 7/8 results are **DEPRECATED**.

- **Target Model Count**: 114 models across 8 categories
  - `ml_tabular` (15): multi-horizon, NaN passthrough for tree models
  - `statistical` (15): original 5 + 10 new (Croston, Holt, AutoCES, etc.)
  - `deep_classical` (9): NBEATS/NHITS/TFT/DeepAR + GRU/LSTM/TCN/MLP/DilatedRNN
  - `transformer_sota` (23): PatchTST, iTransformer, TimesNet, TSMixer, etc.
  - `foundation` (11): Chronos, ChronosBolt, Chronos2, Moirai, etc.
  - `irregular` (4): GRU-D, SAITS, BRITS, CSDI
  - `tslib_sota` (34): TimeFilter, WPMixer, MSGNet, Crossformer, SCINet, CFPT, DeformableTST, ModernTCN, PathFormer, SEMPO, TimePerceiver, TimeBridge, TQNet, PIR, CARD, PDF, TimeRecipe, DUET, SRSNet, etc.
  - `autofit` (4): V734, V735, V736, V739 (validation-based)
- **Platform**: ULHPC Iris — GPU (V100 32GB, ~756GB RAM), bigmem (3TB, 112 CPUs), QOS `normal` (2-day wall)
- **Ablations**: 4 per task — `core_only`, `core_text`, `core_edgar`, `full` (task3: 3, no core_text)
- **Tasks**: `task1_outcome`, `task2_forecast`, `task3_risk_adjust`
- **Canonical Output Dir**: `runs/benchmarks/block3_phase9_fair/`
- **Validated Progress**: 66 metrics.json files, 5,083 valid records, 50 models materialized
- **SLURM Scripts**: `.slurm_scripts/phase9/` (66 scripts + submission helper)
- **Live Results**: See [docs/BLOCK3_RESULTS.md](docs/BLOCK3_RESULTS.md)
- **Full Status**: See [docs/BLOCK3_MODEL_STATUS.md](docs/BLOCK3_MODEL_STATUS.md)

## Hard Constraints
- Only commit/push: `scripts/`, `src/`, `configs/`, `docs/`.
- Never commit anything under `runs/`.
- Block 3 must read **only** `docs/audits/FULL_SCALE_POINTER.yaml` (via `FreezePointer` class).
- **Freeze artifacts are read-only**: never modify files under `runs/*_20260203_225620/`.

## Execution Contract
See `docs/BLOCK3_EXECUTION_CONTRACT.md` for the full execution contract.
Preflight check: `python3 scripts/assert_block3_execution_contract.py --entrypoint <script>`

## Block 3 Architecture

### Data Flow
```
docs/audits/FULL_SCALE_POINTER.yaml
        │
        ▼
FreezePointer (src/narrative/data_preprocessing/block3_dataset.py)
        │
        ├─> offers_core_daily (parquet)
        ├─> offers_text (parquet)
        └─> edgar_feature_store (parquet)
                │
                ▼
        Block3Dataset (lazy loading + explicit joins)
                │
                ▼
        BenchmarkHarness (scripts/run_block3_benchmark_shard.py)
                │
                ├─> Statistical (5):       AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive
                ├─> ML Tabular (15):       LightGBM, XGBoost, CatBoost, RandomForest, ...
                ├─> Deep Classical (9):    NBEATS, NHITS, TFT, DeepAR, GRU, LSTM, TCN, MLP, DilatedRNN
                ├─> Transformer SOTA (23): PatchTST, iTransformer, TimesNet, TSMixer, Informer, ...
                ├─> Foundation (11):       Chronos, ChronosBolt, Moirai, MoiraiLarge, Timer, ...
                ├─> Irregular (4):         GRU-D, SAITS, BRITS, CSDI
                ├─> TSLib SOTA (20):       TimeFilter, WPMixer, MSGNet, Crossformer, SCINet, ...
                └─> AutoFit (3):           V734, V735, V736
```

### Join Keys
- `entity_id` + `crawled_date_day`: offers_core ↔ offers_text
- `cik` + `crawled_date_day`: offers_core ↔ edgar_store

### AutoFit Meta-Features
From `scripts/block3_profile_data.py`:
- `nonstationarity_score`, `periodicity_score`, `multiscale_score`
- `long_memory_score`, `irregular_score`, `heavy_tail_score`
- `exog_strength`, `edgar_strength`, `text_strength`, `missing_rate`

## Key Scripts (Block 3)
- Freeze verification: `scripts/block3_verify_freeze.py`
- Data profiling: `scripts/block3_profile_data.py`
- Benchmark harness (shard): `scripts/run_block3_benchmark_shard.py`
- SLURM submission (Phase 9): `.slurm_scripts/phase9/submit_all_phase9.sh`
- Results aggregator: `scripts/aggregate_block3_results.py`
- Results consolidator: `scripts/consolidate_block3_results.py`
- Paper tables: `scripts/make_paper_tables_v2.py`
- Dataset interface: `src/narrative/data_preprocessing/block3_dataset.py`
- AutoFit composer: `src/narrative/auto_fit/rule_based_composer.py`
- Concept bottleneck: `src/narrative/explainability/concept_bottleneck.py`

## Model Source Files (Block 3)
- Deep + Transformer + Foundation: `src/narrative/block3/models/deep_models.py`
- Statistical (StatsForecast): `src/narrative/block3/models/statistical.py`
- Irregular (PyPOTS): `src/narrative/block3/models/irregular_models.py`
- Traditional ML (sklearn): `src/narrative/block3/models/traditional_ml.py`
- AutoFit wrapper: `src/narrative/block3/models/autofit_wrapper.py`
- RL policy (contextual bandit): `src/narrative/block3/models/rl_policy.py`
- Multi-agent coordination: `src/narrative/block3/models/multi_agent_ensemble.py`
- Unified registry: `src/narrative/block3/models/registry.py`
- Base classes: `src/narrative/block3/models/base.py`

## Block 3 Configuration
- Config file: `configs/block3.yaml`
- Targets: `total_amount_sold`, `number_investors`, `days_to_close`
- Horizons: [7, 14, 30, 60]
- Context lengths: [30, 60, 90]
- Metrics: RMSE, MAE, MAPE, SMAPE, CRPS (probabilistic)

## Commit Trail
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

## Pending Work (Block 3)
1. ✅ Freeze verification script
2. ✅ Unified dataset interface
3. ✅ Data profiling for AutoFit
4. ✅ Benchmark harness with 6 baseline categories
5. ✅ AutoFit rule-based composer
6. ✅ Concept bottleneck for interpretability
7. ✅ Model registry expansion: 127 models across 8 categories (Phase 4-8)
8. ✅ Panel data fix: all categories receive entity-panel kwargs
9. ✅ Deep/Transformer models: 22 NeuralForecast models (panel-aware, +xLSTM/TimeLLM/DeepNPTS)
10. ✅ Statistical models: entity-sampled panel via StatsForecast
11. ✅ Foundation models: 11 models (Chronos family + Moirai family + Timer/TimeMoE/MOMENT/LagLlama/TimesFM)
12. ✅ Irregular models: GRU-D, SAITS, BRITS, CSDI via PyPOTS
13. ✅ AutoFit V734/V735/V736 (older versions dropped)
14. ✅ Benchmark harness updated for all 8 categories + 4 ablations
15. ✅ Phase 7 code fixes (5 root causes across 4 files, 57/57 tests pass)
16. ✅ EDGAR timezone fix (merge_asof dtype mismatch)
17. ✅ Statistical OOM fix (64G → 112G memory)
18. ✅ V73 factored contextual bandit RL policy (Thompson Sampling / LinUCB)
19. ✅ V73 multi-agent coordination (Recon/Scout/Composer/Critic blackboard protocol)
20. ✅ V72 root cause analysis (6 root causes, GPU gate fix)
21. ✅ Phase 7/8 benchmark — deprecated (4 critical bugs found)
22. ✅ Phase 9 clean results: 5,083 valid records, 50 models in `block3_phase9_fair/`
23. ⏳ Phase 9 re-runs: 66 SLURM scripts ready, targeting 100 models × 104 conditions
24. ⏳ TCAV-style concept importance analysis
25. ⏳ Final leaderboard + paper tables (after Phase 9 completes)
