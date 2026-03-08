#!/usr/bin/env python3
"""AutoFitV7.1 leakage-safe routing tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from narrative.block3.models.autofit_wrapper import (
    AutoFitV71Wrapper,
    _apply_lane_postprocess,
    _blend_weights_for_lane,
    _champion_anchor_candidates,
    _build_lane_postprocess_state,
    _infer_target_lane,
    _quick_screen_threshold_for_lane,
    _safe_inverse_transform,
)


def test_target_lane_detection_binary_count_heavy_tail():
    y_binary = pd.Series([0, 1] * 100)
    lane_binary = _infer_target_lane({}, y_binary)
    assert lane_binary == "binary"

    y_count = pd.Series(np.random.RandomState(42).poisson(lam=4.0, size=500))
    lane_count = _infer_target_lane({}, y_count)
    assert lane_count == "count"

    # Heavy-tail, non-negative, non-count-like
    heavy = np.random.RandomState(7).lognormal(mean=5.0, sigma=2.0, size=2000)
    y_heavy = pd.Series(heavy)
    lane_heavy = _infer_target_lane({}, y_heavy)
    assert lane_heavy == "heavy_tail"


def test_dynamic_thresholds_and_weights_are_lane_specific():
    assert _quick_screen_threshold_for_lane("heavy_tail") == 0.95
    assert _quick_screen_threshold_for_lane("count") == 0.85
    assert _quick_screen_threshold_for_lane("binary") == 0.90

    assert _blend_weights_for_lane("heavy_tail") == (0.75, 0.25)
    assert _blend_weights_for_lane("count") == (0.55, 0.45)
    assert _blend_weights_for_lane("binary") == (0.50, 0.50)


def test_candidate_pool_lane_specific_and_deduplicated():
    wrapper = AutoFitV71Wrapper()
    count_pool = wrapper._candidate_pool_for_lane("count")
    assert "XGBoostPoisson" in count_pool
    assert "LightGBMTweedie" in count_pool
    assert len(count_pool) == len(set(count_pool))

    binary_pool = wrapper._candidate_pool_for_lane("binary")
    assert "TabPFNClassifier" in binary_pool
    assert len(binary_pool) == len(set(binary_pool))

    heavy_pool = wrapper._candidate_pool_for_lane("heavy_tail")
    assert "TabPFNRegressor" in heavy_pool
    assert len(heavy_pool) == len(set(heavy_pool))


def test_count_lane_postprocess_is_non_negative_integer_and_deterministic():
    y_train = pd.Series(np.random.RandomState(9).poisson(lam=6.0, size=800).astype(float))
    state1 = _build_lane_postprocess_state("count", y_train)
    state2 = _build_lane_postprocess_state("count", y_train)
    assert state1 == state2

    raw = np.array([-3.2, 0.49, 1.51, np.nan, np.inf, 12.9], dtype=float)
    out = _apply_lane_postprocess(raw, state1)
    assert np.isfinite(out).all()
    assert (out >= 0).all()
    assert np.allclose(out, np.rint(out))


def test_binary_lane_postprocess_is_probability_bounded():
    y_train = pd.Series([0, 1] * 30)
    state = _build_lane_postprocess_state("binary", y_train)
    raw = np.array([-0.1, 0.2, 1.3, np.nan], dtype=float)
    out = _apply_lane_postprocess(raw, state)
    assert np.isfinite(out).all()
    assert (out >= 0).all()
    assert (out <= 1).all()


def test_count_safe_mode_ignores_single_extreme_outlier_for_upper_bound():
    rng = np.random.RandomState(123)
    y = rng.poisson(lam=6.0, size=2000).astype(float)
    y[0] = 1e9
    y_train = pd.Series(y)

    safe_state = _build_lane_postprocess_state("count", y_train, count_safe_mode=True)
    legacy_state = _build_lane_postprocess_state("count", y_train, count_safe_mode=False)

    assert safe_state["upper"] is not None
    assert legacy_state["upper"] is not None
    assert safe_state["upper"] < legacy_state["upper"]
    assert safe_state["upper"] < 1e7


def test_safe_inverse_transform_guards_nonfinite_and_extreme_values():
    class _DummyTransform:
        kind = "log1p"

        def inverse(self, y):
            return np.expm1(y)

    state = {"lower": 0.0, "upper": 200.0, "round_to_int": True}
    raw = np.array([0.0, np.nan, np.inf, 1000.0], dtype=float)
    out, hits = _safe_inverse_transform(raw, _DummyTransform(), state)
    assert hits >= 3
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_anchor_candidates_are_defined_per_lane():
    assert "NBEATS" in _champion_anchor_candidates("count")
    assert "PatchTST" in _champion_anchor_candidates("binary")
    assert "Chronos" in _champion_anchor_candidates("heavy_tail")
