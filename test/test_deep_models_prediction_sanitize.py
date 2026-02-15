#!/usr/bin/env python3
"""Numerical-stability tests for deep model fallback predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from narrative.block3.models.base import ModelConfig
from narrative.block3.models.deep_models import DeepModelWrapper, _RobustFallback


class _DummyFallbackModel:
    def predict(self, X):
        return np.array([np.nan] * len(X), dtype=float)


def test_robust_fallback_predict_replaces_non_finite():
    fb = _RobustFallback()
    fb._model = _DummyFallbackModel()
    fb._feature_cols = ["f1"]
    fb._transform_kind = "identity"
    fb._safe_fill = 7.0
    fb._fitted = True

    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0]})
    y_pred = fb.predict(X)

    assert y_pred is not None
    assert np.isfinite(y_pred).all()
    assert np.allclose(y_pred, 7.0)


class _DummyNF:
    def predict(self):
        return pd.DataFrame(
            {
                "unique_id": ["e1"] * 7,
                "ds": pd.date_range("2025-01-01", periods=7, freq="D"),
                "VanillaTransformer": np.ones(7, dtype=float),
            }
        )


class _DummyHybridFallback:
    _fitted = True

    def predict(self, X):
        return np.array([np.nan, np.inf, -np.inf, np.nan], dtype=float)


def test_deep_model_hybrid_predict_sanitizes_non_finite_fallback():
    cfg = ModelConfig(name="VanillaTransformer", model_type="forecasting", params={})
    model = DeepModelWrapper(cfg, "VanillaTransformer")
    model._fitted = True
    model._use_fallback = False
    model._horizon = 7
    model._nf = _DummyNF()
    model._fallback = _DummyHybridFallback()

    X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0]})
    test_raw = pd.DataFrame(
        {
            "entity_id": ["e1", "e2", "e2", "e1"],
            "funding_raised_usd": [1.0, 2.0, 3.0, 4.0],
        }
    )

    y_pred = model.predict(
        X,
        test_raw=test_raw,
        target="funding_raised_usd",
        horizon=7,
    )

    assert y_pred.shape == (4,)
    assert np.isfinite(y_pred).all()
