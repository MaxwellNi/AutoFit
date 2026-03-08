#!/usr/bin/env python3
"""Coverage and fairness guard tests for benchmark harness."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _load_benchmark_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_block3_benchmark_shard.py"
    spec = importlib.util.spec_from_file_location("run_block3_benchmark_shard", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _make_shard(module):
    shard = module.BenchmarkShard.__new__(module.BenchmarkShard)
    shard.category = "ml_tabular"
    shard.task = "task1_outcome"
    shard.ablation = "core_only"
    shard.preset_config = SimpleNamespace(n_bootstrap=0)
    shard.seed = 42
    shard.git_hash = "test"
    shard.predictions = []
    return shard


class _DummyModel:
    def __init__(self, pred_fn, use_fallback=False):
        self._pred_fn = pred_fn
        self._use_fallback = use_fallback

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X, **kwargs):
        return self._pred_fn(len(X))


def _mock_data(n=200):
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "entity_id": np.arange(n),
            "crawled_date_day": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "cik": ["c" + str(i % 9) for i in range(n)],
            "funding_raised_usd": rng.lognormal(mean=2.0, sigma=0.5, size=n),
            "feat1": rng.randn(n),
            "feat2": rng.randn(n),
        }
    )
    return df


def test_run_model_fairness_guard_length_mismatch(monkeypatch):
    module = _load_benchmark_module()
    shard = _make_shard(module)

    train = _mock_data(120)
    test = _mock_data(80)

    monkeypatch.setattr(module, "check_model_available", lambda name: True)
    monkeypatch.setattr(
        module,
        "get_model",
        lambda name: _DummyModel(lambda n: np.zeros(n - 1, dtype=float)),
    )

    with pytest.raises(RuntimeError, match="FAIRNESS GUARD"):
        shard.run_model("Dummy", train, train, test, "funding_raised_usd", 7)


def test_run_model_fairness_guard_low_coverage(monkeypatch):
    module = _load_benchmark_module()
    shard = _make_shard(module)

    train = _mock_data(120)
    test = _mock_data(100)

    monkeypatch.setattr(module, "check_model_available", lambda name: True)

    def _pred_with_nans(n):
        y = np.zeros(n, dtype=float)
        y[:5] = np.nan  # 95% coverage -> should fail (<0.98)
        return y

    monkeypatch.setattr(module, "get_model", lambda name: _DummyModel(_pred_with_nans))

    with pytest.raises(RuntimeError, match="FAIRNESS GUARD"):
        shard.run_model("Dummy", train, train, test, "funding_raised_usd", 7)


def test_run_model_records_coverage_and_fallback(monkeypatch):
    module = _load_benchmark_module()
    shard = _make_shard(module)

    train = _mock_data(120)
    test = _mock_data(80)

    monkeypatch.setattr(module, "check_model_available", lambda name: True)
    monkeypatch.setattr(
        module,
        "get_model",
        lambda name: _DummyModel(lambda n: np.linspace(0.1, 0.9, n), use_fallback=True),
    )

    result = shard.run_model("Dummy", train, train, test, "funding_raised_usd", 7)
    assert result is not None
    assert result.fairness_pass is True
    assert result.prediction_coverage_ratio >= 0.98
    assert result.n_missing_predictions == 0
    assert result.fallback_fraction == 1.0
    assert result.effective_eval_rows > 0
    assert result.lane_clip_rate == 0.0
    assert result.inverse_transform_guard_hits == 0
    assert result.anchor_models_used == []
    assert result.policy_action_id is None
    assert result.oof_guard_triggered is False


def test_comparability_filter_drops_low_coverage_and_unfair():
    from scripts.aggregate_block3_results import apply_comparability_filter

    df = pd.DataFrame(
        [
            {"model_name": "A", "prediction_coverage_ratio": 0.99, "fairness_pass": True},
            {"model_name": "B", "prediction_coverage_ratio": 0.97, "fairness_pass": True},
            {"model_name": "C", "prediction_coverage_ratio": 1.00, "fairness_pass": False},
        ]
    )
    out = apply_comparability_filter(df, min_coverage=0.98, fairness_only=True)
    assert len(out) == 1
    assert out.iloc[0]["model_name"] == "A"
