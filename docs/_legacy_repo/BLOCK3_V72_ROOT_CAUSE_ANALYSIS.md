# AutoFit V7.2 Root Cause Analysis

## Executive Summary

V7.2 achieved **0 rank-1 wins across 104 benchmark conditions** with a median rank of 22 and a global performance gap of -1.84% vs baseline. This document identifies and traces the 6 root causes to their exact code locations.

## Performance Context

| Metric | V7.2 Value |
|--------|------------|
| Rank-1 wins | 0 / 104 |
| Median rank | 22 |
| Best rank | 3 |
| Global improvement vs baseline | -1.84% (NEGATIVE) |
| Pilot pass | **FAIL** |
| Best AutoFit version (by wins) | V1 (30/104) |

### Champion Model Distribution (ground truth)

| Model | Wins | Category |
|-------|------|----------|
| NBEATS | 39 | deep_classical |
| PatchTST | 24 | transformer_sota |
| NHITS | 23 | deep_classical |
| KAN | 7 | transformer_sota |
| Chronos | 6 | foundation |
| NBEATSx | 4 | transformer_sota |
| DLinear | 1 | transformer_sota |

**62/104 wins from deep_classical, 36/104 from transformer_sota, 6/104 from foundation.**
All 104 champions are GPU-dependent models.

---

## Root Cause #1 (CRITICAL): GPU Gate Drops All Champion-Class Models

**Location**: `src/narrative/block3/models/autofit_wrapper.py`, function `_fit_single_candidate`, line ~345

```python
_GPU_CATEGORIES = {"deep_classical", "transformer_sota", "foundation"}
cat = _get_model_category(model_name)
if cat in _GPU_CATEGORIES and not torch.cuda.is_available():
    return None  # SILENTLY SKIPPED
```

**Evidence**: V72 SLURM jobs were submitted with `partition="batch"` and `partition="bigmem"` — both CPU-only. No `--gres=gpu` was ever requested.

**Impact**: All 20+ deep_classical, transformer_sota, and foundation models (NBEATS, NHITS, PatchTST, KAN, Chronos, NBEATSx, etc.) were **silently dropped** from the candidate pool. V72 trained exclusively on ~9 ML tabular models (LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, HistGradientBoosting, + count specialists).

**Consequence**: The entire ensemble selection, champion anchor routing, and stacking architecture operated on the **wrong model pool**. Since all 104 champions are GPU models, V72 could never win.

---

## Root Cause #2: Dead Champion Anchor Templates

**Location**: `_champion_template_for_lane` (line ~3802) and champion anchor routing (line ~5490)

The champion templates specify GPU-dependent models as preferred anchors:
- Count lane: `["NBEATS", "NHITS", "KAN", "PatchTST", ...]`
- Binary lane: `["PatchTST", "NHITS", "NBEATSx", ...]`
- Heavy-tail lane: `["NHITS", "PatchTST", "Chronos", ...]`

Since all deep models were GPU-gated out (Root Cause #1), the anchor routing code:
```python
available_template = [nm for nm in template_candidates if nm in oof_clean]
```
always returned empty for the intended champions. Anchors degraded to:
- **RandomForest**: used as anchor 28 times
- **LightGBMTweedie**: used as anchor 20 times

---

## Root Cause #3: Horizon Invariance

**Evidence**: V72 produces **identical MAE** across all 4 horizons (h=1, 7, 14, 30) for the same task/ablation/target. For `funding_raised_usd`, MAE = 396,381.29 across ALL horizons.

**Mechanism**: ML tabular models (the only survivors after GPU gate) do not internally encode horizon information. The `horizon` parameter is passed to them but they ignore it since they are not panel-aware. Only `_PANEL_CATEGORIES = {"deep_classical", "transformer_sota", "foundation", "statistical", "irregular"}` models use the horizon parameter — but all were dropped.

---

## Root Cause #4: Count Lane 99.5% Clip Rate

**Location**: `_build_lane_postprocess_state` (line ~3672) and `_apply_lane_postprocess`

For `investors_count` (count lane), `count_safe_mode=True` triggers:
1. `np.rint()` rounding → changes virtually every float prediction
2. `np.clip(0, upper)` with `upper = np.percentile(y, 99)` → too aggressive for right-skewed data

Result: `lane_clip_rate = 0.9951` (99.5% of predictions modified). The postprocessing itself introduces systematic bias.

---

## Root Cause #5: V7.2 Worse Than V1

| AutoFit | Condition Wins |
|---------|---------------|
| V1 | 30 / 104 |
| V7 | 28 / 104 |
| V71 | 16 / 104 |
| V3 | 13 / 104 |
| **V72** | **8 / 104** |
| V3Max | 5 / 104 |
| V6 | 4 / 104 |

V72's added complexity (sparse MoE, multi-seed meta-learner, champion anchoring, 2-layer stacking) all **degrade** performance when the model pool consists only of correlated tabular models. The meta-learner overfits to tabular noise; the Caruana ensemble finds no diversity.

---

## Root Cause #6: OOF Guard Fallback Rate

For `funding_raised_usd`: OOF guard triggered in **16/44 conditions (36%)**. When the blended ensemble of 3-4 correlated tabular models performs worse than the single best, the ensemble collapses to a single model (usually RandomForest or LightGBM).

---

## Resolution Path

All root causes trace back to **Root Cause #1**: the SLURM partition assignment. Fixing requires:

1. **Run V73 on GPU partition** (`--gres=gpu:1`, `partition=gpu`)
2. This automatically resolves RC#1 (GPU gate passes), RC#2 (anchors available), RC#3 (panel models use horizon), RC#5 (diversity restored), RC#6 (OOF guard unnecessary with strong candidates)
3. RC#4 (clip rate) requires separate code fix in `_build_lane_postprocess_state`

---

## Appendix: V72 Routing Diagnostics

From 108 V72 benchmark records:
- **Lane distribution**: heavy_tail 36, count 36, binary 12, general 24
- **Anchor models used**: RandomForest (28), LightGBMTweedie (20) — **zero deep models**
- **OOF guard triggers**: 16 / 108 (15%)
- **Train time**: mean 7862s (~2.2 hours), range [2957s, 14255s]
- **Policy actions**: all `offline_rule_v72|lane=X|hb=Y|ablation=Z|miss=low|qs=N|budget=96`
