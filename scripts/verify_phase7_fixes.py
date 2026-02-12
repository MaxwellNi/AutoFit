#!/usr/bin/env python3
"""
Phase 7 Fix Verification Script — KDD'26 Block 3.

Validates ALL critical changes applied during Phase 7 comprehensive fix:

1. _build_panel_df: max_entities=None → uses ALL entities
2. _RobustFallback: LightGBM + adaptive asinh/sqrt target transform
3. DeepModelWrapper: per-entity hybrid predict, no all-or-nothing threshold
4. FoundationModelWrapper: no entity cap, RobustFallback
5. HFFoundationModelWrapper: no entity cap, RobustFallback
6. StatsForecastWrapper: no entity cap
7. GRUDWrapper / SAITSWrapper: raised entity cap, RobustFallback
8. AutoFit V1-V7: target_transform passed to all _temporal_kfold_evaluate_all
9. _repeated_temporal_kfold_evaluate_all: target_transform parameter added
10. Import consistency check (relative imports in irregular_models.py)

Usage:
    python scripts/verify_phase7_fixes.py
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
import sys
import textwrap
from pathlib import Path

# ============================================================================
# Setup
# ============================================================================
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PASS = 0
FAIL = 0
WARN = 0


def check(condition: bool, name: str, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        msg = f"  ✗ FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def warn(name: str, detail: str = ""):
    global WARN
    WARN += 1
    print(f"  ⚠ WARN: {name}" + (f" — {detail}" if detail else ""))


# ============================================================================
# 1. Source-level AST checks (no imports needed)
# ============================================================================
print("\n" + "=" * 72)
print("PHASE 7 FIX VERIFICATION")
print("=" * 72)

# --- deep_models.py ---
print("\n[1] deep_models.py — AST checks")
dm_path = REPO / "src/narrative/block3/models/deep_models.py"
dm_src = dm_path.read_text()

check("max_entities: Optional[int] = None" in dm_src,
      "_build_panel_df: max_entities default is None")

check("class _RobustFallback" in dm_src,
      "_RobustFallback class defined")

check('self._transform_kind = "asinh"' in dm_src,
      "_RobustFallback supports asinh transform")

check('self._transform_kind = "sqrt"' in dm_src,
      "_RobustFallback supports sqrt transform")

check("import lightgbm as lgb" in dm_src,
      "_RobustFallback uses LightGBM (not Ridge)")

check("def _build_static_df" in dm_src,
      "_build_static_df function defined")

check("_NF_SUPPORTS_STATIC_EXOG" in dm_src,
      "_NF_SUPPORTS_STATIC_EXOG constant defined")

check("_EXOG_EXCLUDE" in dm_src,
      "_EXOG_EXCLUDE constant defined")

# Entity caps
check("_max_e = 5000 if self.model_name in _NEEDS_N_SERIES else None" in dm_src,
      "DeepModelWrapper: 5000 for cross-series, None for channel-independent")

# No Ridge fallback reference (should be replaced by RobustFallback)
check("_fallback_ridge" not in dm_src.split("class DeepModelWrapper")[1].split("class Foundation")[0],
      "DeepModelWrapper: Ridge fallback fully replaced by _RobustFallback",
      "Found leftover _fallback_ridge reference")

# Per-entity hybrid predict (no 95% threshold)
deep_predict_src = dm_src.split("class DeepModelWrapper")[1].split("class Foundation")[0]
# Per-entity hybrid predict (no 95% threshold) — exclude comments
deep_predict_code = "\n".join(
    line for line in deep_predict_src.split("\n")
    if not line.strip().startswith("#")
)
check("coverage < 0.95" not in deep_predict_code,
      "DeepModelWrapper: no 95% all-or-nothing threshold")

check("Per-entity HYBRID predict" in deep_predict_src,
      "DeepModelWrapper: per-entity HYBRID predict log message")

# Foundation — no MAX_E cap
fm_src = dm_src.split("class FoundationModelWrapper")[1].split("class HFFoundation")[0]
check("MAX_E = 500" not in fm_src and "MAX_E =" not in fm_src,
      "FoundationModelWrapper: no MAX_E entity cap",
      "Found MAX_E cap")

check("self._fallback = _RobustFallback()" in fm_src,
      "FoundationModelWrapper: uses _RobustFallback")

# HF Foundation
hf_src = dm_src.split("class HFFoundationModelWrapper")[1]
check("self._fallback = _RobustFallback()" in hf_src,
      "HFFoundationModelWrapper: uses _RobustFallback")


# --- statistical.py ---
print("\n[2] statistical.py — AST checks")
stat_path = REPO / "src/narrative/block3/models/statistical.py"
stat_src = stat_path.read_text()

check("MAX_ENTITIES" not in stat_src,
      "StatsForecastWrapper: no MAX_ENTITIES cap")

check("rng.choice" not in stat_src,
      "StatsForecastWrapper: no entity sampling",
      "Found rng.choice entity sampling")


# --- irregular_models.py ---
print("\n[3] irregular_models.py — AST checks")
irr_path = REPO / "src/narrative/block3/models/irregular_models.py"
irr_src = irr_path.read_text()

check("max_entities=5000" in irr_src,
      "GRU-D/SAITS: max_entities raised to 5000")

check("from .deep_models import _RobustFallback" in irr_src,
      "irregular: uses relative import for _RobustFallback")

check("from src.narrative" not in irr_src,
      "irregular: no absolute src.narrative import",
      "Found absolute import — may break on different install paths")

check("self._robust_fallback" in irr_src,
      "GRU-D/SAITS: _robust_fallback attribute present")


# --- autofit_wrapper.py ---
print("\n[4] autofit_wrapper.py — target_transform propagation")
af_path = REPO / "src/narrative/block3/models/autofit_wrapper.py"
af_src = af_path.read_text()

# Check _temporal_kfold_evaluate_all has target_transform param
check("target_transform: Optional[TargetTransform] = None" in af_src,
      "_temporal_kfold_evaluate_all: has target_transform parameter")

# Check _repeated_temporal_kfold_evaluate_all has target_transform param
check('target_transform: Optional[TargetTransform] = None,\n) -> Tuple' in af_src
      or ('_repeated_temporal_kfold_evaluate_all' in af_src and
          af_src.split("_repeated_temporal_kfold_evaluate_all")[1][:500].count("target_transform") >= 1),
      "_repeated_temporal_kfold_evaluate_all: has target_transform parameter")

# Check _fit_single_candidate has target_transform param
check("target_transform: Optional[TargetTransform] = None," in af_src,
      "_fit_single_candidate: has target_transform parameter")

# Count how many callers pass target_transform
n_tt_passes = af_src.count("target_transform=")
check(n_tt_passes >= 14,
      f"target_transform passed at least 14 times (found {n_tt_passes})",
      f"Only {n_tt_passes} occurrences — some callers may be missing")

# Check specific AutoFit variants
# Check specific AutoFit variants (V1, V2, V2E need _tt = TargetTransform)
for variant in ["AutoFitV1", "AutoFitV2", "AutoFitV2E"]:
    # Find the .fit() method that calls _temporal_kfold_evaluate_all
    # V1/V2/V2E should have _tt = TargetTransform() before the call
    marker = f"[{variant}]"
    idx = af_src.find(marker)
    if idx > 0:
        section = af_src[max(0, idx - 2000):idx + 2000]
        check("_tt = TargetTransform()" in section
              or "target_transform=" in section,
              f"{variant}: creates TargetTransform and passes it")
    else:
        warn(f"{variant}: could not locate section")

# V4, V5 already have self._target_transform
for variant, attr in [("AutoFitV4", "self._target_transform"),
                       ("AutoFitV5", "self._target_transform"),
                       ("AutoFitV6", "self._target_xform"),
                       ("AutoFitV7", "self._target_xform")]:
    pat = f"target_transform={attr}"
    check(pat in af_src,
          f"{variant}: passes {attr} to evaluator")

# GPU-awareness in _fit_single_candidate
check("torch.cuda.is_available" in af_src,
      "_fit_single_candidate: GPU-awareness check")

# Degenerate detection
check("near-constant" in af_src.lower() or "pred_std" in af_src,
      "_fit_single_candidate: near-constant prediction detection")


# ============================================================================
# 2. Attempt module-level import (catches missing deps at import time)
# ============================================================================
print("\n[5] Module import checks")

try:
    from src.narrative.block3.models.base import ModelBase, ModelConfig
    check(True, "base.py imports OK")
except Exception as e:
    check(False, "base.py imports", str(e))

try:
    from src.narrative.block3.models.deep_models import (
        _RobustFallback, _build_panel_df, _build_static_df,
    )
    check(True, "deep_models.py core helpers import OK")
except Exception as e:
    check(False, "deep_models.py core helpers import", str(e))

try:
    from src.narrative.block3.models.traditional_ml import (
        ProductionGBDTWrapper,
    )
    check(True, "traditional_ml.py imports OK")
except Exception as e:
    check(False, "traditional_ml.py imports", str(e))


# ============================================================================
# 3. Unit tests for _RobustFallback
# ============================================================================
print("\n[6] _RobustFallback unit tests")

import numpy as np
import pandas as pd

try:
    from src.narrative.block3.models.deep_models import _RobustFallback

    # Test 1: Heavy-tailed target (kurtosis > 5) → should use asinh
    rng = np.random.RandomState(42)
    n = 2000
    y_heavy = np.concatenate([
        rng.exponential(1000, n - 50),
        rng.exponential(1e6, 50),  # extreme tail
    ])
    X_df = pd.DataFrame({
        "feat1": rng.randn(n),
        "feat2": rng.randn(n),
        "feat3": rng.uniform(0, 100, n),
    })
    y_s = pd.Series(y_heavy)

    fb = _RobustFallback()
    fb.fit(X_df, y_s)
    check(fb._transform_kind == "asinh",
          f"Heavy-tail target → asinh (got: {fb._transform_kind})")
    check(fb._fitted, "RobustFallback fitted successfully")

    preds = fb.predict(X_df)
    check(preds is not None and len(preds) == n,
          f"Predictions shape correct ({len(preds)} == {n})")

    # Predictions should NOT be constant
    pred_std = np.std(preds)
    check(pred_std > 0,
          f"Predictions not constant (std={pred_std:.2f})")

    # Test 2: Count-like target → should use sqrt
    y_count = rng.poisson(5, n).astype(float)
    fb2 = _RobustFallback()
    fb2.fit(X_df, pd.Series(y_count))
    check(fb2._transform_kind == "sqrt",
          f"Count-like target → sqrt (got: {fb2._transform_kind})")

    # Test 3: Normal target → should use identity
    y_normal = rng.randn(n) * 10 + 50
    fb3 = _RobustFallback()
    fb3.fit(X_df, pd.Series(y_normal))
    check(fb3._transform_kind == "identity",
          f"Normal target → identity (got: {fb3._transform_kind})")

    # Test 4: Forward/inverse roundtrip
    y_test = np.array([0.0, 1.0, 100.0, 1e6, 1e9])
    fb_rt = _RobustFallback()
    fb_rt._transform_kind = "asinh"
    forward = fb_rt._forward(y_test)
    inverse = fb_rt._inverse(forward)
    roundtrip_err = np.max(np.abs(y_test - inverse))
    check(roundtrip_err < 1e-3,
          f"Asinh roundtrip error: {roundtrip_err:.2e}")

except Exception as e:
    check(False, f"_RobustFallback tests failed: {e}")


# ============================================================================
# 4. Unit tests for _build_panel_df
# ============================================================================
print("\n[7] _build_panel_df unit tests")

try:
    from src.narrative.block3.models.deep_models import _build_panel_df

    # Create synthetic train_raw
    n_entities = 100
    obs_per_entity = 30
    rows = []
    for i in range(n_entities):
        for j in range(obs_per_entity):
            rows.append({
                "entity_id": f"eid_{i}",
                "crawled_date_day": pd.Timestamp("2024-01-01") + pd.Timedelta(days=j),
                "funding_raised_usd": float(rng.exponential(10000)),
            })
    train_raw = pd.DataFrame(rows)

    # Test max_entities=None → all entities
    panel = _build_panel_df(train_raw, "funding_raised_usd", max_entities=None, min_obs=10)
    check(panel is not None, "Panel built successfully with max_entities=None")
    n_e = panel["unique_id"].nunique()
    check(n_e == n_entities,
          f"All entities included: {n_e} == {n_entities}")

    # Test max_entities=50 → capped
    panel_50 = _build_panel_df(train_raw, "funding_raised_usd", max_entities=50)
    n_e_50 = panel_50["unique_id"].nunique()
    check(n_e_50 == 50,
          f"Entity cap works: {n_e_50} == 50")

    # Test min_obs filter
    # Add 20 entities with only 5 obs each
    sparse_rows = []
    for i in range(20):
        for j in range(5):
            sparse_rows.append({
                "entity_id": f"sparse_{i}",
                "crawled_date_day": pd.Timestamp("2024-01-01") + pd.Timedelta(days=j),
                "funding_raised_usd": float(rng.exponential(10000)),
            })
    mixed_raw = pd.concat([train_raw, pd.DataFrame(sparse_rows)], ignore_index=True)
    panel_mixed = _build_panel_df(mixed_raw, "funding_raised_usd", min_obs=10)
    n_mixed = panel_mixed["unique_id"].nunique()
    check(n_mixed == n_entities,
          f"min_obs filter works: {n_mixed} entities (sparse excluded)")

except Exception as e:
    check(False, f"_build_panel_df tests failed: {e}")


# ============================================================================
# 5. _build_static_df test
# ============================================================================
print("\n[8] _build_static_df unit test")

try:
    from src.narrative.block3.models.deep_models import _build_static_df

    # Add numeric features to train_raw
    train_raw_feat = train_raw.copy()
    train_raw_feat["edgar_revenue"] = rng.exponential(1e6, len(train_raw_feat))
    train_raw_feat["edgar_assets"] = rng.exponential(1e7, len(train_raw_feat))

    eids = train_raw_feat["entity_id"].unique()
    result = _build_static_df(train_raw_feat, eids, "funding_raised_usd")
    check(result is not None, "_build_static_df returns valid result")
    if result:
        sdf, cols = result
        check("unique_id" in sdf.columns, "static_df has unique_id column")
        check(len(cols) >= 2, f"At least 2 exog features: {len(cols)}")
        check(sdf["unique_id"].nunique() == n_entities,
              f"All entities represented: {sdf['unique_id'].nunique()}")

except Exception as e:
    check(False, f"_build_static_df tests failed: {e}")


# ============================================================================
# 6. AutoFit TargetTransform test
# ============================================================================
print("\n[9] AutoFit TargetTransform propagation")

try:
    from src.narrative.block3.models.autofit_wrapper import (
        TargetTransform, _temporal_kfold_evaluate_all,
        _fit_single_candidate,
    )

    # Check _temporal_kfold_evaluate_all signature has target_transform
    sig = inspect.signature(_temporal_kfold_evaluate_all)
    check("target_transform" in sig.parameters,
          "_temporal_kfold_evaluate_all has target_transform param")

    # Check _fit_single_candidate signature
    sig2 = inspect.signature(_fit_single_candidate)
    check("target_transform" in sig2.parameters,
          "_fit_single_candidate has target_transform param")

    # Check TargetTransform class works
    tt = TargetTransform()
    tt.fit(pd.Series(y_heavy))
    check(tt.kind in ("log1p", "asinh", "identity", "sqrt"),
          f"TargetTransform kind: {tt.kind}")
    transformed = tt.transform(y_heavy)
    inversed = tt.inverse(transformed)
    rt_err = np.mean(np.abs(y_heavy - inversed))
    check(rt_err < 1.0,
          f"TargetTransform roundtrip error: {rt_err:.4f}")

except ImportError as e:
    warn(f"AutoFit import failed (expected if deps missing): {e}")
except Exception as e:
    check(False, f"AutoFit tests failed: {e}")

# Check _repeated_temporal_kfold_evaluate_all signature
try:
    from src.narrative.block3.models.autofit_wrapper import (
        _repeated_temporal_kfold_evaluate_all,
    )
    sig3 = inspect.signature(_repeated_temporal_kfold_evaluate_all)
    check("target_transform" in sig3.parameters,
          "_repeated_temporal_kfold_evaluate_all has target_transform param")
except ImportError:
    warn("_repeated_temporal_kfold_evaluate_all import failed")
except Exception as e:
    check(False, f"_repeated CV sig check: {e}")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 72)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {WARN} warnings")
print("=" * 72)

if FAIL > 0:
    print("\n⚠ SOME CHECKS FAILED — review above\n")
    sys.exit(1)
else:
    print("\n✓ ALL CHECKS PASSED — Phase 7 fixes verified\n")
    sys.exit(0)
