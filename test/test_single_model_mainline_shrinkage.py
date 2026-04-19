#!/usr/bin/env python3
"""Tests for P5 adaptive shrinkage gate in investors lane."""
from __future__ import annotations

import numpy as np
import pytest

from narrative.block3.models.single_model_mainline.shrinkage_utils import (
    apply_shrinkage,
    fit_shrinkage_gate,
    predict_shrinkage_alpha,
    shrinkage_diagnostics,
)


def _make_data(n=200, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    design = rng.randn(n, 10).astype(np.float32)
    anchor = np.abs(rng.randn(n)) * 5
    # Learned pred = anchor + noise; noise varies by sample
    noise_scale = np.abs(design[:, 0]) + 0.1  # heteroscedastic
    learned = anchor + noise_scale * rng.randn(n)
    learned = np.clip(learned, 0, None)
    target = anchor + 0.3 * rng.randn(n)
    target = np.clip(target, 0, None)
    return design, learned, anchor, target


class TestShrinkageUtils:
    def test_fit_returns_model_and_diagnostics(self):
        design, learned, anchor, target = _make_data()
        model, diag = fit_shrinkage_gate(design, learned, anchor, target)
        assert model is not None
        assert diag["shrinkage_status"] == "converged"
        assert "mean_oracle_alpha" in diag
        assert "shrunk_mae" in diag
        assert diag["base_mae"] >= 0
        assert diag["shrunk_mae"] >= 0

    def test_fit_too_few_samples(self):
        design, learned, anchor, target = _make_data(n=10)
        model, diag = fit_shrinkage_gate(design, learned, anchor, target)
        assert model is None
        assert diag["shrinkage_status"] == "too_few_samples"

    def test_fit_learned_equals_anchor(self):
        n = 50
        design = np.random.randn(n, 5).astype(np.float32)
        anchor = np.ones(n) * 3.0
        learned = anchor.copy()  # identical
        target = np.ones(n) * 4.0
        model, diag = fit_shrinkage_gate(design, learned, anchor, target)
        assert model is None
        assert diag["shrinkage_status"] == "learned_equals_anchor"

    def test_predict_alpha_range(self):
        design, learned, anchor, target = _make_data()
        model, _ = fit_shrinkage_gate(design, learned, anchor, target)
        alpha = predict_shrinkage_alpha(model, design)
        assert alpha.shape == (len(design),)
        assert np.all(alpha >= 0.0)
        assert np.all(alpha <= 1.0)

    def test_predict_alpha_no_model_returns_default(self):
        n = 30
        design = np.random.randn(n, 5).astype(np.float32)
        alpha = predict_shrinkage_alpha(None, design, default_alpha=0.3)
        assert np.allclose(alpha, 0.3)

    def test_apply_shrinkage_strength_zero_returns_learned(self):
        learned = np.array([1.0, 2.0, 3.0])
        anchor = np.array([10.0, 20.0, 30.0])
        alpha = np.array([0.5, 0.5, 0.5])
        result = apply_shrinkage(learned, anchor, alpha, strength=0.0)
        np.testing.assert_array_almost_equal(result, learned)

    def test_apply_shrinkage_strength_one_blends(self):
        learned = np.array([1.0, 2.0, 3.0])
        anchor = np.array([10.0, 20.0, 30.0])
        alpha = np.array([1.0, 0.0, 0.5])
        result = apply_shrinkage(learned, anchor, alpha, strength=1.0)
        expected = np.array([10.0, 2.0, 16.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_shrinkage_reduces_mae_on_noisy_data(self):
        """Shrinkage should reduce MAE when learned predictions are noisy."""
        design, learned, anchor, target = _make_data(n=500, rng_seed=123)
        model, diag = fit_shrinkage_gate(design, learned, anchor, target)
        assert model is not None
        # On training data, shrinkage should not increase MAE
        assert diag["shrunk_mae"] <= diag["base_mae"] + 1e-6

    def test_diagnostics_keys(self):
        alpha = np.random.rand(100)
        diag = shrinkage_diagnostics(alpha)
        assert "shrinkage_mean_alpha" in diag
        assert "shrinkage_median_alpha" in diag
        assert "shrinkage_frac_gt_0.5" in diag


class TestInvestorsLaneShrinkageIntegration:
    """Test that shrinkage gate integrates correctly with InvestorsLaneRuntime."""

    def test_shrinkage_gate_can_fit_and_predict(self):
        from narrative.block3.models.single_model_mainline.lanes.investors_lane import (
            InvestorsLaneRuntime,
        )

        rng = np.random.RandomState(42)
        n = 200
        lane_state = rng.randn(n, 10).astype(np.float32)
        y = np.clip(rng.poisson(2, n).astype(np.float64), 0, None)
        aux = rng.randn(n, 3).astype(np.float32)
        anchor = np.clip(rng.randn(n) * 2 + 2, 0, None)

        runtime = InvestorsLaneRuntime(random_state=42)
        runtime.fit(
            lane_state, y,
            aux_features=aux,
            anchor=anchor,
            enable_hurdle_head=True,
            enable_shrinkage_gate=True,
            shrinkage_strength=0.8,
            horizon=7,
            task_name="task2_forecast",
        )
        assert runtime._fitted
        assert runtime._use_shrinkage_gate

        pred = runtime.predict(lane_state, aux_features=aux, anchor=anchor)
        assert pred.shape == (n,)
        assert np.all(np.isfinite(pred))
        assert np.all(pred >= 0.0)

    def test_shrinkage_gate_disabled_on_h1(self):
        from narrative.block3.models.single_model_mainline.lanes.investors_lane import (
            InvestorsLaneRuntime,
        )

        rng = np.random.RandomState(42)
        n = 100
        lane_state = rng.randn(n, 10).astype(np.float32)
        y = np.clip(rng.poisson(2, n).astype(np.float64), 0, None)

        runtime = InvestorsLaneRuntime(random_state=42)
        runtime.fit(
            lane_state, y,
            enable_shrinkage_gate=True,
            horizon=1,
            task_name="task2_forecast",
        )
        assert not runtime._use_shrinkage_gate  # disabled on h1

    def test_shrinkage_diagnostics_exposed(self):
        from narrative.block3.models.single_model_mainline.lanes.investors_lane import (
            InvestorsLaneRuntime,
        )

        rng = np.random.RandomState(42)
        n = 200
        lane_state = rng.randn(n, 10).astype(np.float32)
        y = np.clip(rng.poisson(2, n).astype(np.float64), 0, None)
        aux = rng.randn(n, 3).astype(np.float32)
        anchor = np.clip(rng.randn(n) * 2 + 2, 0, None)

        runtime = InvestorsLaneRuntime(random_state=42)
        runtime.fit(
            lane_state, y,
            aux_features=aux,
            anchor=anchor,
            enable_hurdle_head=True,
            enable_shrinkage_gate=True,
            shrinkage_strength=0.8,
            horizon=14,
            task_name="task2_forecast",
        )
        diag = runtime.describe_shrinkage()
        assert diag["shrinkage_enabled"]
        assert "shrinkage_converged" in diag
