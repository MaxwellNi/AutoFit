#!/usr/bin/env python3
"""
Tests for AutoFit v2 — No-leakage & integration tests.

Covers:
- Meta-features never use future label information
- Router determinism
- ASHA respects budget
- End-to-end pipeline with synthetic data
- Expert factory creates valid objects
- Task heads produce correct output shapes
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from narrative.auto_fit.meta_features_v2 import MetaFeaturesV2, compute_meta_features
from narrative.auto_fit.router import (
    DEFAULT_EXPERTS,
    GatingMode,
    GatingResult,
    MetaFeatureRouter,
    select_expert_models,
)
from narrative.auto_fit.search_budget import ASHACandidate, run_asha
from narrative.auto_fit.autofit_v2 import autofit_v2, AutoFitV2Result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_panel(n_entities=20, obs_per_entity=60, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_entities):
        eid = f"e_{i:04d}"
        dates = pd.date_range("2024-01-01", periods=obs_per_entity, freq="D")
        target = rng.randn(obs_per_entity).cumsum() + 1000
        feat1 = rng.randn(obs_per_entity) * 10
        feat2 = rng.randn(obs_per_entity) * 5
        for j in range(obs_per_entity):
            rows.append({
                "entity_id": eid,
                "crawled_date_day": dates[j],
                "funding_raised_usd": target[j],
                "feat1": feat1[j],
                "feat2": feat2[j],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# No-leakage tests
# ---------------------------------------------------------------------------

class TestNoLeakage:
    """Ensure meta-features don't leak label information improperly."""

    def test_meta_features_independent_of_target_shuffle(self):
        """
        Assert 1: Shuffling the target column should NOT change the core
        meta-features (A1-A5) that don't depend on target.
        
        Only A6 (exog_strength) and A7 (leakage) may change.
        """
        df = _make_panel()
        mf_orig = compute_meta_features(df, seed=42)

        df_shuffled = df.copy()
        rng = np.random.RandomState(99)
        df_shuffled["funding_raised_usd"] = rng.permutation(
            df_shuffled["funding_raised_usd"].values
        )
        mf_shuf = compute_meta_features(df_shuffled, seed=42)

        # A1: Missingness must be identical
        assert mf_orig.missing_rate_global == mf_shuf.missing_rate_global
        # A2: Irregularity must be identical
        assert mf_orig.sampling_interval_cv == mf_shuf.sampling_interval_cv
        # A4: Kurtosis from feature cols (not target) is identical
        assert abs(mf_orig.kurtosis_mean - mf_shuf.kurtosis_mean) < 1e-6

    def test_meta_features_no_target_in_feature_cols(self):
        """
        Assert 2: target_col is NOT among the feature columns used for
        A1/A2/A4 computations.
        """
        df = _make_panel()
        mf = compute_meta_features(df, target_col="funding_raised_usd")
        # n_feature_cols should not count the target
        expected = len([c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in {"entity_id", "crawled_date_day", "funding_raised_usd"}])
        assert mf.n_feature_cols == expected

    def test_temporal_ordering_not_violated(self):
        """
        Assert 3: compute_meta_features sorts by entity+date but never
        accesses future rows relative to current.
        """
        # If there were a future leak, adding a future-only feature would
        # contaminate non-future meta-features. We verify it doesn't.
        df = _make_panel()
        mf1 = compute_meta_features(df, seed=42)
        
        # Add a column that is only non-null in the last 10 rows per entity
        df2 = df.copy()
        df2["future_only"] = np.nan
        for eid, grp in df2.groupby("entity_id"):
            idx = grp.index[-10:]
            df2.loc[idx, "future_only"] = 999.0
        
        mf2 = compute_meta_features(df2, seed=42)
        # Core features should be very similar (the only difference is
        # extra NaN column affects missing_rate)
        assert abs(mf1.sampling_interval_cv - mf2.sampling_interval_cv) < 1e-10


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

class TestRouter:
    def test_router_determinism(self):
        """Assert 4: Same meta-features → same routing."""
        mf = MetaFeaturesV2(
            missing_rate_global=0.1,
            acf_lag7_mean=0.3,
            nonstationarity_score=0.5,
            kurtosis_mean=4.0,
            n_rows=100000,
            n_entities=5000,
        )
        router = MetaFeatureRouter(gating_mode=GatingMode.SPARSE, top_k=2)
        g1 = router.route(mf)
        g2 = router.route(mf)
        assert g1.expert_weights == g2.expert_weights
        assert g1.top_experts == g2.top_experts

    def test_router_weights_sum_to_one(self):
        """Assert 5: Expert weights sum to 1.0 in sparse mode."""
        mf = MetaFeaturesV2(n_rows=10000, n_entities=500)
        router = MetaFeatureRouter(gating_mode=GatingMode.SPARSE, top_k=3)
        g = router.route(mf)
        total = sum(g.expert_weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_hard_mode_single_expert(self):
        """Assert 6: Hard mode selects exactly one expert with weight 1.0."""
        mf = MetaFeaturesV2(n_rows=10000, n_entities=500)
        router = MetaFeatureRouter(gating_mode=GatingMode.HARD)
        g = router.route(mf)
        active = [n for n, w in g.expert_weights.items() if w > 0]
        assert len(active) == 1
        assert g.expert_weights[active[0]] == 1.0

    def test_heavy_tail_favours_tabular(self):
        """Assert 7: High kurtosis boosts tabular expert."""
        mf_normal = MetaFeaturesV2(kurtosis_mean=2.0, n_rows=50000)
        mf_heavy = MetaFeaturesV2(kurtosis_mean=15.0, n_rows=50000)
        router = MetaFeatureRouter(gating_mode=GatingMode.SOFT)
        g_normal = router.route(mf_normal)
        g_heavy = router.route(mf_heavy)
        assert g_heavy.expert_weights["tabular"] > g_normal.expert_weights["tabular"]

    def test_strong_periodicity_favours_statistical(self):
        """Assert 8: High ACF7 boosts statistical expert."""
        mf_low = MetaFeaturesV2(acf_lag7_mean=0.1, n_rows=50000)
        mf_high = MetaFeaturesV2(acf_lag7_mean=0.8, n_rows=50000)
        router = MetaFeatureRouter(gating_mode=GatingMode.SOFT)
        g_low = router.route(mf_low)
        g_high = router.route(mf_high)
        assert g_high.expert_weights["statistical"] > g_low.expert_weights["statistical"]

    def test_irregular_data_favours_irregular_expert(self):
        """Assert 9: High sampling CV boosts irregular expert."""
        mf_reg = MetaFeaturesV2(sampling_interval_cv=0.01, n_rows=50000)
        mf_irreg = MetaFeaturesV2(sampling_interval_cv=1.5, pct_gaps_gt_7d=0.5,
                                   missing_rate_global=0.5, n_rows=50000)
        router = MetaFeatureRouter(gating_mode=GatingMode.SOFT)
        g_reg = router.route(mf_reg)
        g_irreg = router.route(mf_irreg)
        assert g_irreg.expert_weights["irregular"] > g_reg.expert_weights["irregular"]


# ---------------------------------------------------------------------------
# ASHA tests
# ---------------------------------------------------------------------------

class TestASHA:
    def test_asha_respects_budget(self):
        """Assert 10: ASHA terminates within budget."""
        import time

        def slow_eval(name, config, frac):
            time.sleep(0.05)  # 50ms per eval
            return np.random.rand()

        candidates = [
            ASHACandidate(model_name=f"M{i}", expert_name="test")
            for i in range(5)
        ]
        result = run_asha(candidates, slow_eval, budget_seconds=0.5, seed=42)
        assert result.total_time_seconds < 2.0  # generous upper bound

    def test_asha_returns_winner(self):
        """Assert 11: ASHA always returns a winner."""
        scores = {"M0": 0.5, "M1": 0.3, "M2": 0.8}
        def eval_fn(name, config, frac):
            return scores.get(name, 1.0)

        candidates = [
            ASHACandidate(model_name=name, expert_name="test")
            for name in scores
        ]
        result = run_asha(candidates, eval_fn, budget_seconds=10, seed=42)
        assert result.winner.model_name == "M1"  # lowest score

    def test_asha_halving(self):
        """Assert 12: After first rung, some candidates are eliminated."""
        scores = {f"M{i}": float(i) for i in range(9)}
        def eval_fn(name, config, frac):
            return scores[name]

        candidates = [
            ASHACandidate(model_name=name, expert_name="test")
            for name in scores
        ]
        result = run_asha(candidates, eval_fn, eta=3, budget_seconds=60, seed=42)
        # 9 → ⌈9/3⌉=3 → ⌈3/3⌉=1
        assert result.n_candidates_final <= 3


# ---------------------------------------------------------------------------
# End-to-end pipeline tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_autofit_v2_smoke(self):
        """Assert 13: Pipeline runs on synthetic data without error."""
        df = _make_panel(n_entities=10, obs_per_entity=40)
        result = autofit_v2(
            df,
            target_col="funding_raised_usd",
            task="task1_outcome",
            gating_mode=GatingMode.SPARSE,
            top_k=2,
            run_asha_search=False,
        )
        assert isinstance(result, AutoFitV2Result)
        assert result.final_model != ""
        assert result.total_time_seconds > 0

    def test_autofit_v2_output_dict(self):
        """Assert 14: to_dict() produces valid JSON-serializable output."""
        df = _make_panel(n_entities=10, obs_per_entity=40)
        result = autofit_v2(df, run_asha_search=False)
        d = result.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 100

    def test_autofit_v2_saves_audit(self):
        """Assert 15: Pipeline saves audit artifacts to output_dir."""
        df = _make_panel(n_entities=10, obs_per_entity=40)
        with tempfile.TemporaryDirectory() as tmp:
            result = autofit_v2(
                df,
                run_asha_search=False,
                output_dir=Path(tmp),
            )
            assert result.audit_path is not None
            assert result.audit_path.exists()
            # Meta-features JSON should also exist
            mf_json = Path(tmp) / "meta_features_v2.json"
            assert mf_json.exists()


# ---------------------------------------------------------------------------
# Expert factory tests
# ---------------------------------------------------------------------------

class TestExpertFactory:
    def test_create_all_experts(self):
        """Assert 16: All 6 expert categories can be instantiated."""
        from narrative.models.autofit.experts import create_expert, EXPERT_CLASSES
        for expert_name in EXPERT_CLASSES:
            expert = create_expert(expert_name, model_name="TestModel")
            assert not expert.is_fitted

    def test_invalid_expert_raises(self):
        """Assert 17: Unknown expert name raises ValueError."""
        from narrative.models.autofit.experts import create_expert
        with pytest.raises(ValueError, match="Unknown expert"):
            create_expert("nonexistent", "SomeModel")


# ---------------------------------------------------------------------------
# Task head tests
# ---------------------------------------------------------------------------

class TestTaskHeads:
    def test_task1_head_binary(self):
        """Assert 18: Task1Head produces proba for is_funded."""
        from narrative.models.autofit.heads import create_task_head
        head = create_task_head("task1_outcome")
        raw = np.array([2.0, -1.0, 0.5])
        out = head.postprocess(raw, target="is_funded")
        assert "proba" in out
        assert all(0 <= p <= 1 for p in out["proba"])

    def test_task1_head_regression_non_negative(self):
        """Assert 19: Task1Head clips funding predictions to >= 0."""
        from narrative.models.autofit.heads import create_task_head
        head = create_task_head("task1_outcome")
        raw = np.array([-100.0, 500.0, -50.0])
        out = head.postprocess(raw, target="funding_raised_usd")
        assert all(p >= 0 for p in out["point"])

    def test_task2_head_quantiles(self):
        """Assert 20: Task2Head produces quantile estimates."""
        from narrative.models.autofit.heads import create_task_head
        head = create_task_head("task2_forecast")
        raw = np.array([100.0, 200.0, 300.0])
        y_train = np.random.randn(100) * 50 + 200
        out = head.postprocess(raw, target="funding_raised_usd", y_train=y_train)
        assert "lower" in out
        assert "upper" in out

    def test_task3_head_calibration(self):
        """Assert 21: Task3Head produces calibrated predictions."""
        from narrative.models.autofit.heads import create_task_head
        head = create_task_head("task3_risk_adjust")
        raw = np.array([10.0, 20.0, 30.0])
        y_train = np.array([100.0, 200.0, 300.0, 150.0, 250.0] * 10)
        out = head.postprocess(raw, target="funding_raised_usd", y_train=y_train)
        assert "calibrated" in out
        # Calibrated should have same mean as training
        assert abs(np.mean(out["calibrated"]) - np.mean(y_train)) < 1.0


# ---------------------------------------------------------------------------
# Select expert models
# ---------------------------------------------------------------------------

class TestSelectExpertModels:
    def test_selects_from_top_experts(self):
        """Assert 22: select_expert_models returns models from active experts."""
        mf = MetaFeaturesV2(n_rows=50000, n_entities=2000, kurtosis_mean=8.0)
        router = MetaFeatureRouter(gating_mode=GatingMode.SPARSE, top_k=2)
        gating = router.route(mf)
        selected = select_expert_models(gating, DEFAULT_EXPERTS, max_models_per_expert=2)
        assert len(selected) > 0
        # Each selected model has a positive weight
        for model, expert, weight in selected:
            assert weight > 0
            assert isinstance(model, str)
