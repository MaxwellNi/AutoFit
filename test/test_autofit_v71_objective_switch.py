#!/usr/bin/env python3
"""Objective switching tests for AutoFitV7.1 meta learner."""
from __future__ import annotations

import numpy as np
import pandas as pd

from narrative.block3.models.autofit_wrapper import _build_multi_seed_meta_learner


def _make_meta_dataset(n=240):
    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "f3": rng.randn(n),
    })
    return X


def test_binary_objective_uses_classifier_meta_learner():
    X = _make_meta_dataset()
    y = pd.Series(([0, 1] * (len(X) // 2))[: len(X)])
    oof = {"m1": np.clip(np.random.RandomState(1).rand(len(X)), 0.01, 0.99)}

    learners, cols, mae = _build_multi_seed_meta_learner(
        X,
        y,
        oof,
        n_seeds=1,
        objective_mode="binary",
        use_huber=False,
    )
    assert len(learners) == 1
    assert hasattr(learners[0], "predict_proba")
    assert len(cols) >= 1
    assert mae >= 0.0


def test_count_objective_uses_poisson_meta_learner():
    X = _make_meta_dataset()
    y = pd.Series(np.random.RandomState(2).poisson(lam=3.5, size=len(X)).astype(float))
    oof = {"m1": np.maximum(0.0, np.random.RandomState(3).normal(loc=3.0, scale=1.0, size=len(X)))}

    learners, cols, mae = _build_multi_seed_meta_learner(
        X,
        y,
        oof,
        n_seeds=1,
        objective_mode="count",
        use_huber=False,
    )
    assert len(learners) == 1
    assert hasattr(learners[0], "predict")
    assert len(cols) >= 1
    assert mae >= 0.0


def test_huber_objective_path_runs():
    X = _make_meta_dataset()
    y = pd.Series(np.random.RandomState(4).lognormal(mean=2.0, sigma=1.0, size=len(X)))
    oof = {"m1": np.random.RandomState(5).lognormal(mean=2.0, sigma=1.0, size=len(X))}

    learners, cols, mae = _build_multi_seed_meta_learner(
        X,
        y,
        oof,
        n_seeds=1,
        objective_mode="huber",
        use_huber=True,
    )
    assert len(learners) == 1
    assert len(cols) >= 1
    assert mae >= 0.0
