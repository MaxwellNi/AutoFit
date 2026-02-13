# AutoFit v2: Data-Adaptive Model Selection via Meta-Feature–Gated Mixture of Experts

**Status**: IMPLEMENTED (D1–D2 complete, D3–D6 ready for extension)
**Target**: KDD'26 Full Paper

---

## 1. Problem Statement

Financial time-series from SEC EDGAR offerings exhibit extreme heterogeneity:
- Heavy-tailed funding distributions (kurtosis > 50 for some entities)
- Irregular sampling (daily for active offerings, gaps of weeks otherwise)
- Strong non-stationarity (regulatory and market regime shifts)
- Mixed exogenous signals (EDGAR filings, text sentiment, macro indicators)

No single model family dominates across all entities and tasks.
**AutoFit v2** automatically selects and combines expert model families
based on dataset-level meta-features.

---

## 2. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      AutoFit v2 Pipeline                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐     ┌──────────────┐     ┌────────────────┐  │
│  │  Raw Panel   │────▶│ Meta-Feature │────▶│  MoE Router    │  │
│  │  DataFrame   │     │  Extractor   │     │  (Gating Net)  │  │
│  └─────────────┘     │  (25+ dims)  │     │  Sparse Top-K  │  │
│                       └──────────────┘     └───────┬────────┘  │
│                                                    │           │
│                            ┌───────────────────────┼──────┐    │
│                            ▼           ▼           ▼      │    │
│                       ┌─────────┐ ┌─────────┐ ┌─────────┐│    │
│                       │Tabular  │ │Deep TS  │ │Foundatn ││    │
│                       │ Expert  │ │ Expert  │ │ Expert  ││    │
│                       │LGB/CB/XG│ │NHITS/TFT│ │Chronos/ ││    │
│                       └────┬────┘ └────┬────┘ │Moirai   ││    │
│                            │           │      └────┬────┘│    │
│                            └───────────┼───────────┘     │    │
│                                        ▼                 │    │
│                               ┌────────────────┐         │    │
│                               │  Task Head     │         │    │
│                               │  (Task 1/2/3)  │         │    │
│                               └────────────────┘         │    │
│                                        │                 │    │
│                               ┌────────▼────────┐        │    │
│                               │ Weighted Pred.  │        │    │
│                               │ + Audit Trail   │        │    │
│                               └─────────────────┘        │    │
│                                                          │    │
│   Optional: ASHA Search ─────────────────────────────────┘    │
│   (within each active expert, budget-controlled)              │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Meta-Features (Section A)

25+ meta-features across 7 groups:

| Group | Features | Purpose |
|-------|----------|---------|
| A1: Missingness | `missing_rate_global`, `per_entity_mean/max`, `n_cols_above_50pct` | Expert suitability (trees handle NaN natively) |
| A2: Irregularity | `sampling_interval_cv`, `pct_gaps_gt_7d/30d`, `median_obs` | Route to irregular-aware models (GRU-D) |
| A3: Multi-scale | `acf_lag7/30/90_mean`, `multiscale_score` | Detect periodicity and long memory |
| A4: Heavy-tail | `kurtosis_mean/max`, `tail_index_proxy`, `pct_outliers_3σ` | Select robust losses, favour trees |
| A5: Non-stationarity | `nonstationarity_score`, `rolling_mean_drift` | Decomposition backbones, regime models |
| A6: Exog strength | `exog_strength`, `edgar_strength`, `text_strength` | Penalise foundation models (can't use exog) |
| A7: Leakage | `leakage_suspects`, `leakage_max_corr` | Safety gate: flag >0.95 correlation columns |

### Backward Compatibility
`MetaFeaturesV2.to_v1_compat()` maps to the 10 v1 meta-feature names used by `rule_based_composer.py`.

---

## 4. Expert Categories

| Expert | Category | Models | When Preferred |
|--------|----------|--------|----------------|
| `tabular` | `ml_tabular` | LightGBM, CatBoost, XGBoost, RF, HistGB | Heavy tails, moderate missing, always-strong baseline |
| `deep_ts` | `deep_classical` | NHITS, NBEATS, TFT, DeepAR | Non-stationary, multi-scale, sufficient data |
| `transformer` | `transformer_sota` | PatchTST, iTransformer, TimesNet, TSMixer | Long memory, multi-scale, large data |
| `foundation` | `foundation` | Chronos, Moirai, TimesFM | Few-shot, periodic, no strong exog |
| `statistical` | `statistical` | AutoARIMA, AutoETS, AutoTheta, MSTL | Strong periodicity, near-stationary |
| `irregular` | `irregular` | GRU-D, SAITS | High sampling CV, many gaps, high missingness |

---

## 5. Gating Modes

| Mode | Behaviour | Use Case |
|------|-----------|----------|
| `hard` | argmax → single expert | Ablation, interpretability |
| `soft` | Full softmax over all experts | Maximum flexibility |
| `sparse` | Top-K with renormalization (default K=2) | Paper default: best tradeoff |

---

## 6. Task-Specific Heads

| Task | Head | Post-processing |
|------|------|-----------------|
| Task 1 (Outcome) | `Task1Head` | Sigmoid for `is_funded`; clip ≥ 0 for financial amounts |
| Task 2 (Forecast) | `Task2Head` | Quantile intervals from residual bootstrap |
| Task 3 (Risk Adjust) | `Task3Head` | Mean-variance calibration to training distribution |

---

## 7. ASHA Budget Search (D6)

Within each active expert, ASHA selects the best specific model:

```
Rung 0: 10% entities → evaluate all candidates → keep top ⌈N/η⌉
Rung 1: 30% entities → re-evaluate survivors → keep top ⌈N/η²⌉
Rung 2: 100% entities → final evaluation → winner
```

- **Budget**: wall-clock seconds (default 1800s)
- **η**: 3 (halving factor)
- **Metric**: MAE (Task 1/3), CRPS (Task 2)
- **Output**: Full JSON audit trail

---

## 8. Zero-Leakage Guarantees

1. **Meta-features**: A1/A2/A4 use only X features (never target)
2. **A3/A5**: Use target for autocorrelation/stationarity but compute on *training split only*
3. **A6**: Mutual-info between exog and target is valid (no temporal leak)
4. **A7**: Explicitly flags suspiciously-correlated columns (>0.95)
5. **Temporal split**: All evaluation uses `unified_protocol.apply_temporal_split` with embargo
6. **Test coverage**: 44 assertions across 2 test files verify all guarantees

---

## 9. Module Map

```
src/narrative/auto_fit/
├── meta_features_v2.py      # A1-A7 meta-feature computation
├── router.py                 # MoE gating network
├── search_budget.py          # ASHA v2 budget search (replaces budget_search.py)
├── autofit_v2.py            # Main entry point
├── rule_based_composer.py   # v1 (retained for backward compat)
├── two_stage_selector.py    # v1 (retained for backward compat)
└── diagnose_dataset.py      # v1 (retained for backward compat)

src/narrative/models/autofit/
├── __init__.py
├── experts.py               # Expert wrappers (6 categories)
└── heads.py                 # Task-specific prediction heads

test/
├── test_meta_features_v2.py          # 22 assertions on meta-features
└── test_autofit_v2_no_leakage.py     # 22 assertions on pipeline
```

---

## 10. Evaluation Protocol

### Comparison Targets
AutoFit v2 must beat:
1. **Single-best model** (LightGBM) on each task
2. **Oracle ensemble** (retrospective best combination) — soft upper bound
3. **v1 rule-based composer** — direct predecessor

### Metrics per Task

| Task | Primary | Secondary |
|------|---------|-----------|
| Task 1 | MAE, RMSE | AUC (is_funded), NDCG@10 (ranking) |
| Task 2 | CRPS, MAE | Quantile coverage, MASE |
| Task 3 | Worst-group MAE | ECE, performance gap |

### Statistical Significance
- Bootstrap CI (n=1000) on primary metric
- Paired permutation test (p < 0.05)

---

## 11. Implementation Progress (D1–D6)

| Step | Status | Description |
|------|--------|-------------|
| D1 | ✅ DONE | `meta_features_v2.py` + 22 tests |
| D2 | ✅ DONE | `autofit_v2.py`, `router.py`, expert wrappers, task heads, 22 tests |
| D3 | ⏳ TODO | Robust losses (Huber for heavy-tail), Platt calibration |
| D4 | ⏳ TODO | Task 2 extension: quantile + coverage in ASHA objective |
| D5 | ⏳ TODO | Task 3 extension: Group DRO / worst-group scoring |
| D6 | ⏳ TODO | Foundation experts + full ASHA with real Block3Dataset |

---

## 12. CLI Usage

```bash
# Smoke test (meta-features only)
PYTHONPATH=src python -m narrative.auto_fit.meta_features_v2 \
    --pointer docs/audits/FULL_SCALE_POINTER.yaml \
    --target funding_raised_usd \
    --output-dir runs/autofit_v2/smoke

# Full pipeline (routing + audit, no ASHA)
PYTHONPATH=src python -m narrative.auto_fit.autofit_v2 \
    --pointer docs/audits/FULL_SCALE_POINTER.yaml \
    --task task1_outcome \
    --target funding_raised_usd \
    --gating-mode sparse --top-k 2 \
    --output-dir runs/autofit_v2/task1

# Run tests
PYTHONPATH=src python -m pytest test/test_meta_features_v2.py test/test_autofit_v2_no_leakage.py -v
```
