#!/usr/bin/env python3
"""
Tests for AutoFit v2 meta-features (Section D1).

15+ assertions covering:
- Contract: output type, required keys, value ranges
- Determinism: same input → same output
- Edge cases: empty DF, single entity, all-NaN columns
- Leakage detection: planted leaky column is flagged
- v1 backward compatibility
- Heavy-tail detection: planted heavy-tail column
- Missingness accounting
- Irregularity detection
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from narrative.auto_fit.meta_features_v2 import (
    MetaFeaturesV2,
    compute_meta_features,
    save_meta_features_report,
    _safe_acf,
    _hill_estimator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_panel(
    n_entities: int = 50,
    obs_per_entity: int = 90,
    seed: int = 42,
    add_leaky_col: bool = False,
    add_heavy_tail_col: bool = False,
    missing_frac: float = 0.0,
    irregular: bool = False,
) -> pd.DataFrame:
    """Generate a synthetic panel DataFrame for testing."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_entities):
        eid = f"entity_{i:04d}"
        if irregular:
            # Random gaps
            days_offset = np.sort(rng.choice(range(obs_per_entity * 3), size=obs_per_entity, replace=False))
        else:
            days_offset = np.arange(obs_per_entity)
        dates = pd.date_range("2024-01-01", periods=obs_per_entity * 3, freq="D")[days_offset]
        target = rng.randn(obs_per_entity).cumsum() + rng.randn() * 100
        feat1 = rng.randn(obs_per_entity) * 10
        feat2 = rng.randn(obs_per_entity) * 5
        edgar_x = rng.randn(obs_per_entity) * 2
        text_x = rng.randn(obs_per_entity)

        for j in range(obs_per_entity):
            row = {
                "entity_id": eid,
                "crawled_date_day": dates[j],
                "funding_raised_usd": target[j],
                "feat1": feat1[j],
                "feat2": feat2[j],
                "edgar_score": edgar_x[j],
                "text_sentiment": text_x[j],
            }
            if add_leaky_col:
                row["leaky_col"] = target[j] + rng.randn() * 0.001
            if add_heavy_tail_col:
                # Pareto-like heavy tail
                row["heavy_tail"] = float(rng.pareto(1.5)) * 1000
            rows.append(row)

    df = pd.DataFrame(rows)

    if missing_frac > 0:
        mask = rng.rand(len(df)) < missing_frac
        df.loc[mask, "feat1"] = np.nan
        mask2 = rng.rand(len(df)) < missing_frac
        df.loc[mask2, "feat2"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Tests — Contract
# ---------------------------------------------------------------------------

class TestMetaFeaturesContract:
    """Output contract: type, keys, ranges."""

    def test_returns_dataclass(self):
        """Assert 1: compute_meta_features returns MetaFeaturesV2."""
        df = _make_panel(n_entities=5, obs_per_entity=30)
        mf = compute_meta_features(df)
        assert isinstance(mf, MetaFeaturesV2)

    def test_required_keys_in_dict(self):
        """Assert 2: to_dict() has all 25+ expected keys."""
        df = _make_panel(n_entities=5, obs_per_entity=30)
        mf = compute_meta_features(df)
        d = mf.to_dict()
        required = {
            "missing_rate_global", "missing_rate_per_entity_mean",
            "sampling_interval_cv", "acf_lag7_mean", "acf_lag30_mean",
            "multiscale_score", "kurtosis_mean", "tail_index_proxy",
            "nonstationarity_score", "exog_strength", "edgar_strength",
            "text_strength", "leakage_suspects", "leakage_max_corr",
            "n_entities", "n_rows", "n_feature_cols", "computed_at",
        }
        assert required.issubset(d.keys()), f"Missing keys: {required - d.keys()}"

    def test_missing_rate_in_01(self):
        """Assert 3: missing_rate_global ∈ [0, 1]."""
        df = _make_panel(n_entities=5, obs_per_entity=30)
        mf = compute_meta_features(df)
        assert 0.0 <= mf.missing_rate_global <= 1.0

    def test_nonstationarity_in_01(self):
        """Assert 4: nonstationarity_score ∈ [0, 1]."""
        df = _make_panel(n_entities=10, obs_per_entity=60)
        mf = compute_meta_features(df)
        assert 0.0 <= mf.nonstationarity_score <= 1.0

    def test_acf_values_bounded(self):
        """Assert 5: ACF means are in [0, 1]."""
        df = _make_panel(n_entities=10, obs_per_entity=100)
        mf = compute_meta_features(df)
        assert 0.0 <= mf.acf_lag7_mean <= 1.0
        assert 0.0 <= mf.acf_lag30_mean <= 1.0


# ---------------------------------------------------------------------------
# Tests — Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        """Assert 6: Identical input → identical output."""
        df = _make_panel(n_entities=20, obs_per_entity=50, seed=99)
        mf1 = compute_meta_features(df, seed=42)
        mf2 = compute_meta_features(df, seed=42)
        d1, d2 = mf1.to_dict(), mf2.to_dict()
        # Exclude timestamp
        d1.pop("computed_at", None)
        d2.pop("computed_at", None)
        assert d1 == d2


# ---------------------------------------------------------------------------
# Tests — Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_dataframe(self):
        """Assert 7: Empty DataFrame returns zero-valued MetaFeaturesV2."""
        df = pd.DataFrame(columns=["entity_id", "crawled_date_day", "funding_raised_usd"])
        mf = compute_meta_features(df)
        assert mf.n_rows == 0
        assert mf.n_entities == 0

    def test_single_entity(self):
        """Assert 8: Single-entity panel works without error."""
        df = _make_panel(n_entities=1, obs_per_entity=60)
        mf = compute_meta_features(df)
        assert mf.n_entities == 1
        assert mf.n_rows == 60

    def test_all_nan_column(self):
        """Assert 9: Column of all NaN doesn't crash."""
        df = _make_panel(n_entities=5, obs_per_entity=30)
        df["all_nan_col"] = np.nan
        mf = compute_meta_features(df)
        assert mf.missing_rate_global > 0  # at least the NaN column


# ---------------------------------------------------------------------------
# Tests — Leakage Detection
# ---------------------------------------------------------------------------

class TestLeakageDetection:
    def test_leaky_column_detected(self):
        """Assert 10: Planted leaky column (corr>0.99 with target) is flagged."""
        df = _make_panel(n_entities=20, obs_per_entity=100, add_leaky_col=True)
        mf = compute_meta_features(df)
        assert "leaky_col" in mf.leakage_suspects, (
            f"Expected 'leaky_col' in suspects, got {mf.leakage_suspects}"
        )
        assert mf.leakage_max_corr > 0.95

    def test_no_false_leakage_on_clean_data(self):
        """Assert 11: Clean random features have no leakage suspects."""
        df = _make_panel(n_entities=20, obs_per_entity=100, seed=7)
        mf = compute_meta_features(df)
        # Random features shouldn't correlate > 0.95 with target
        assert len(mf.leakage_suspects) == 0


# ---------------------------------------------------------------------------
# Tests — Heavy Tail
# ---------------------------------------------------------------------------

class TestHeavyTail:
    def test_heavy_tail_detected(self):
        """Assert 12: Pareto-distributed column elevates kurtosis_mean."""
        df_clean = _make_panel(n_entities=10, obs_per_entity=60, seed=1)
        mf_clean = compute_meta_features(df_clean)

        df_heavy = _make_panel(n_entities=10, obs_per_entity=60, seed=1, add_heavy_tail_col=True)
        mf_heavy = compute_meta_features(df_heavy)

        assert mf_heavy.kurtosis_mean > mf_clean.kurtosis_mean


# ---------------------------------------------------------------------------
# Tests — Missingness
# ---------------------------------------------------------------------------

class TestMissingness:
    def test_high_missing_rate_detected(self):
        """Assert 13: 40% injected missingness shows up."""
        df = _make_panel(n_entities=10, obs_per_entity=60, missing_frac=0.4)
        mf = compute_meta_features(df)
        assert mf.missing_rate_global > 0.1  # at least some of the columns

    def test_zero_missing_rate(self):
        """Assert 14: Clean data has ~0 missingness."""
        df = _make_panel(n_entities=10, obs_per_entity=60, missing_frac=0.0)
        mf = compute_meta_features(df)
        assert mf.missing_rate_global < 0.01


# ---------------------------------------------------------------------------
# Tests — v1 Compatibility
# ---------------------------------------------------------------------------

class TestV1Compat:
    def test_v1_keys_present(self):
        """Assert 15: to_v1_compat() returns all expected v1 keys."""
        df = _make_panel(n_entities=10, obs_per_entity=60)
        mf = compute_meta_features(df)
        v1 = mf.to_v1_compat()
        v1_keys = {
            "missing_rate", "periodicity_score", "long_memory_score",
            "multiscale_score", "nonstationarity_score", "irregular_score",
            "heavy_tail_score", "exog_strength", "edgar_strength", "text_strength",
        }
        assert v1_keys == set(v1.keys())

    def test_v1_values_numeric(self):
        """Assert 16: All v1 values are float."""
        df = _make_panel(n_entities=10, obs_per_entity=60)
        mf = compute_meta_features(df)
        for k, v in mf.to_v1_compat().items():
            assert isinstance(v, float), f"{k} is {type(v)}"


# ---------------------------------------------------------------------------
# Tests — Irregularity
# ---------------------------------------------------------------------------

class TestIrregularity:
    def test_irregular_panel_detected(self):
        """Assert 17: Irregular sampling has higher CV than regular."""
        df_regular = _make_panel(n_entities=10, obs_per_entity=60, irregular=False)
        mf_regular = compute_meta_features(df_regular)

        df_irreg = _make_panel(n_entities=10, obs_per_entity=60, irregular=True)
        mf_irreg = compute_meta_features(df_irreg)

        assert mf_irreg.sampling_interval_cv > mf_regular.sampling_interval_cv


# ---------------------------------------------------------------------------
# Tests — Report I/O
# ---------------------------------------------------------------------------

class TestReportIO:
    def test_json_and_md_written(self):
        """Assert 18: save_meta_features_report creates both files."""
        df = _make_panel(n_entities=5, obs_per_entity=30)
        mf = compute_meta_features(df)
        with tempfile.TemporaryDirectory() as tmp:
            j, m = save_meta_features_report(mf, Path(tmp), stamp="test")
            assert j.exists()
            assert m.exists()
            # JSON is valid
            data = json.loads(j.read_text())
            assert "n_entities" in data
            # MD has header
            md = m.read_text()
            assert "# AutoFit v2 Meta-Features Report" in md

    def test_json_roundtrip(self):
        """Assert 19: JSON → dict → values match."""
        df = _make_panel(n_entities=5, obs_per_entity=30)
        mf = compute_meta_features(df)
        with tempfile.TemporaryDirectory() as tmp:
            j, _ = save_meta_features_report(mf, Path(tmp))
            data = json.loads(j.read_text())
            assert data["n_entities"] == mf.n_entities
            assert abs(data["missing_rate_global"] - mf.missing_rate_global) < 1e-8


# ---------------------------------------------------------------------------
# Tests — Internal helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_safe_acf_short_series(self):
        """Assert 20: _safe_acf returns NaN for too-short series."""
        assert np.isnan(_safe_acf(np.array([1, 2, 3]), lag=7))

    def test_hill_estimator_degenerate(self):
        """Assert 21: _hill_estimator returns 0 for too-short input."""
        assert _hill_estimator(np.array([1.0, 2.0])) == 0.0

    def test_hill_estimator_heavy(self):
        """Assert 22: Hill estimator > 0 for Pareto draws."""
        rng = np.random.RandomState(42)
        x = rng.pareto(1.5, size=1000)
        h = _hill_estimator(x)
        assert h > 0.1
