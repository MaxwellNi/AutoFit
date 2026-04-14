#!/usr/bin/env python3
"""Minimal unit tests for AutoFit V7.3 thin-increment behaviour.

Tests verify:
1. V73 class construction and inheritance chain.
2. Count two-part head helpers (`_fit_count_two_part_head`, `_apply_count_two_part_head`).
3. Binary calibration joint-gate selector (`_select_binary_calibration_with_gate`).
4. V73 is retained in source but blocked from the current active registry.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return importlib.import_module("narrative.block3.models.autofit_wrapper")


# ---------- Test 1: V73 construction, hierarchy & config ----------

def test_v73_class_inherits_v72():
    module = _load_module()
    v73 = module.AutoFitV73Wrapper(top_k=4)
    assert isinstance(v73, module.AutoFitV72Wrapper)
    assert isinstance(v73, module.AutoFitV71Wrapper)
    assert v73.config.name == "AutoFitV73"
    assert v73.config.params["strategy"] == "v73_gpu_full_spectrum"
    assert v73.config.params["version"] == "7.3"
    # V7.3 state fields exist and have correct defaults
    assert v73._v73_variant_id == ""
    assert v73._reuse_decision == "rerun_required"
    assert v73._reuse_source_run == ""
    assert v73._count_two_part_state == {}
    assert v73._pinball_q90 == 0.0


def test_v73_factory_function():
    module = _load_module()
    v73 = module.get_autofit_v73(top_k=3)
    assert v73.config.name == "AutoFitV73"
    assert isinstance(v73, module.AutoFitV73Wrapper)


# ---------- Test 2: Count two-part head helpers ----------

def test_fit_count_two_part_head_with_zeros():
    module = _load_module()

    # Synthetic: 40% zeros, positive values ~ 10-50
    np.random.seed(42)
    n = 200
    y = np.zeros(n)
    y[80:] = np.random.uniform(10, 50, 120)
    blend = np.random.uniform(5, 40, n)  # noisy predictions

    state = module._fit_count_two_part_head(blend, y, count_distribution_family="auto")

    assert state["fitted"] is True
    assert state["family"] == "two_part_active"  # >30% zeros → active
    assert 0.3 <= state["zero_prob"] <= 0.5
    assert 0.5 <= state["positive_scale"] <= 2.0
    assert "n_samples" in state["diagnostics"]


def test_fit_count_two_part_head_passthrough_few_zeros():
    module = _load_module()

    np.random.seed(123)
    n = 200
    y = np.random.uniform(5, 100, n)
    y[:10] = 0  # only 5% zeros
    blend = np.random.uniform(5, 90, n)

    state = module._fit_count_two_part_head(blend, y, count_distribution_family="auto")

    assert state["fitted"] is True
    assert state["family"] == "passthrough"  # <30% zeros → passthrough


def test_apply_count_two_part_head_active():
    module = _load_module()

    state = {
        "fitted": True,
        "family": "two_part_active",
        "positive_scale": 1.5,
        "zero_prob": 0.4,
    }
    preds = np.array([10.0, 20.0, 0.0, -5.0])
    adjusted = module._apply_count_two_part_head(preds, state)

    # Scale correction: multiply by 1.5, clip to >=0
    np.testing.assert_allclose(adjusted, [15.0, 30.0, 0.0, 0.0])


def test_apply_count_two_part_head_passthrough():
    module = _load_module()

    state = {"fitted": True, "family": "passthrough"}
    preds = np.array([10.0, 20.0, -5.0])
    adjusted = module._apply_count_two_part_head(preds, state)
    np.testing.assert_array_equal(adjusted, preds)


def test_apply_count_two_part_head_unfitted():
    module = _load_module()

    state = {"fitted": False}
    preds = np.array([10.0, 20.0])
    adjusted = module._apply_count_two_part_head(preds, state)
    np.testing.assert_array_equal(adjusted, preds)


def test_fit_count_two_part_head_insufficient_data():
    module = _load_module()

    # Too few samples
    state = module._fit_count_two_part_head(
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
    )
    assert state["fitted"] is False


# ---------- Test 3: Binary calibration joint-gate ----------

def test_binary_joint_gate_selects_best_brier():
    module = _load_module()

    candidates = {
        "platt": {"brier": 0.10, "logloss": 0.30, "ece": 0.03},
        "isotonic": {"brier": 0.12, "logloss": 0.28, "ece": 0.04},
        "identity": {"brier": 0.15, "logloss": 0.40, "ece": 0.05},
    }
    best = module._select_binary_calibration_with_gate(candidates, ece_threshold=0.05)
    assert best == "platt"  # best brier + ECE within threshold


def test_binary_joint_gate_identity_force_on_high_ece():
    module = _load_module()

    candidates = {
        "platt": {"brier": 0.10, "logloss": 0.30, "ece": 0.50},
        "identity": {"brier": 0.15, "logloss": 0.40, "ece": 0.05},
    }
    best = module._select_binary_calibration_with_gate(candidates, ece_threshold=0.05)
    assert best == "identity"  # platt ECE too far from identity


def test_binary_joint_gate_empty_candidates():
    module = _load_module()
    best = module._select_binary_calibration_with_gate({})
    assert best == "identity"


# ---------- Test 4: Registration ----------

def test_v73_in_autofit_models():
    module = _load_module()
    assert "AutoFitV73" in module.AUTOFIT_MODELS
    v73 = module.AUTOFIT_MODELS["AutoFitV73"]()
    assert v73.config.name == "AutoFitV73"


def test_v73_blocked_from_current_registry():
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    registry = importlib.import_module("narrative.block3.models.registry")
    assert "AutoFitV73" not in registry.MODEL_CATEGORIES.get("autofit", [])
    with pytest.raises(ValueError, match="Retired AutoFit model blocked from current registry"):
        registry.get_model("AutoFitV73")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
