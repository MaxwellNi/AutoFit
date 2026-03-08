#!/usr/bin/env python3
"""Reproducibility tests for AutoFitV7.1 helper components."""
from __future__ import annotations

import numpy as np
import pandas as pd

from narrative.block3.models.autofit_wrapper import _build_regime_retrieval_features


def _make_feature_frame(n=400):
    rng = np.random.RandomState(42)
    X = pd.DataFrame(
        {
            "a": rng.randn(n),
            "b": rng.randn(n) * 2.0,
            "c": rng.randn(n) * 0.5,
            "d": rng.randn(n),
        }
    )
    # Inject deterministic missingness pattern
    X.loc[X.index % 7 == 0, "b"] = np.nan
    X.loc[X.index % 11 == 0, "c"] = np.nan
    return X


def test_regime_retrieval_fit_is_deterministic():
    X = _make_feature_frame()

    X1, state1 = _build_regime_retrieval_features(X, fit=True)
    X2, state2 = _build_regime_retrieval_features(X, fit=True)

    for col in [
        "__regime_min_dist__",
        "__regime_mean_dist__",
        "__regime_best_bucket__",
        "__regime_margin__",
    ]:
        np.testing.assert_allclose(X1[col].values, X2[col].values, rtol=1e-9, atol=1e-9)

    assert state1.get("enabled") == state2.get("enabled")
    if state1.get("enabled"):
        np.testing.assert_allclose(state1["centers"], state2["centers"], rtol=1e-9, atol=1e-9)
        assert state1["cols"] == state2["cols"]


def test_regime_retrieval_predict_reuses_train_state():
    X_train = _make_feature_frame(300)
    X_test = _make_feature_frame(120)

    _, state = _build_regime_retrieval_features(X_train, fit=True)
    Xp1, _ = _build_regime_retrieval_features(X_test, fit=False, state=state)
    Xp2, _ = _build_regime_retrieval_features(X_test, fit=False, state=state)

    for col in [
        "__regime_min_dist__",
        "__regime_mean_dist__",
        "__regime_best_bucket__",
        "__regime_margin__",
    ]:
        np.testing.assert_allclose(Xp1[col].values, Xp2[col].values, rtol=1e-9, atol=1e-9)
