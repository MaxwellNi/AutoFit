#!/usr/bin/env python3
"""Unit tests for AutoFit V7.3.2 Structural Oracle Router.

Tests verify:
1. Target type detection (binary / count / heavy_tail / general)
2. Horizon bucket mapping
3. Ablation class mapping
4. Oracle table completeness and correctness
5. Oracle lookup behavior
6. Fallback candidate selection
7. V732 registration in AUTOFIT_MODELS and model registry
8. Champion pool completeness
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return importlib.import_module("narrative.block3.models.autofit_wrapper")


# ---------- Test 1: Target type detection ----------

def test_detect_target_type_binary():
    module = _load_module()
    y = pd.Series(np.random.choice([0.0, 1.0], size=200))
    assert module.AutoFitV732Wrapper._detect_target_type(y) == "binary"


def test_detect_target_type_count():
    module = _load_module()
    y = pd.Series(np.random.randint(0, 200, size=500).astype(float))
    assert module.AutoFitV732Wrapper._detect_target_type(y) == "count"


def test_detect_target_type_heavy_tail():
    module = _load_module()
    np.random.seed(42)
    y = pd.Series(np.abs(np.random.exponential(scale=100000, size=500)))
    y.iloc[:10] = np.random.exponential(scale=10000000, size=10)
    result = module.AutoFitV732Wrapper._detect_target_type(y)
    assert result == "heavy_tail", f"Expected heavy_tail, got {result}"


def test_detect_target_type_general():
    module = _load_module()
    y = pd.Series(np.random.normal(0, 1, size=200))
    assert module.AutoFitV732Wrapper._detect_target_type(y) == "general"


def test_detect_target_type_small_sample():
    module = _load_module()
    y = pd.Series([1.0, 2.0, 3.0])
    assert module.AutoFitV732Wrapper._detect_target_type(y) == "general"


# ---------- Test 2: Horizon bucket ----------

def test_horizon_bucket_short():
    module = _load_module()
    assert module.AutoFitV732Wrapper._horizon_bucket(1) == "short"


def test_horizon_bucket_medium():
    module = _load_module()
    assert module.AutoFitV732Wrapper._horizon_bucket(7) == "medium"
    assert module.AutoFitV732Wrapper._horizon_bucket(3) == "medium"


def test_horizon_bucket_long():
    module = _load_module()
    assert module.AutoFitV732Wrapper._horizon_bucket(14) == "long"
    assert module.AutoFitV732Wrapper._horizon_bucket(30) == "long"


# ---------- Test 3: Ablation class mapping ----------

def test_ablation_class_temporal():
    module = _load_module()
    assert module.AutoFitV732Wrapper._ablation_class("core_only") == "temporal"
    assert module.AutoFitV732Wrapper._ablation_class("core_text") == "temporal"
    assert module.AutoFitV732Wrapper._ablation_class("unknown") == "temporal"


def test_ablation_class_exogenous():
    module = _load_module()
    assert module.AutoFitV732Wrapper._ablation_class("core_edgar") == "exogenous"
    assert module.AutoFitV732Wrapper._ablation_class("full") == "exogenous"


# ---------- Test 4: Oracle table completeness ----------

def test_oracle_table_has_24_entries():
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    assert len(table) == 24, f"Expected 24 entries, got {len(table)}"


def test_oracle_table_covers_all_target_types():
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    target_types = {k[0] for k in table.keys()}
    assert target_types == {"heavy_tail", "count", "binary"}, (
        f"Expected {{heavy_tail, count, binary}}, got {target_types}"
    )


def test_oracle_table_covers_all_horizons():
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    horizons = {k[1] for k in table.keys()}
    assert horizons == {1, 7, 14, 30}, f"Expected {{1,7,14,30}}, got {horizons}"


def test_oracle_table_covers_both_ablation_classes():
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    abl_classes = {k[2] for k in table.keys()}
    assert abl_classes == {"temporal", "exogenous"}, (
        f"Expected {{temporal, exogenous}}, got {abl_classes}"
    )


def test_oracle_table_all_values_are_tuples_of_two():
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    for key, val in table.items():
        assert isinstance(val, tuple), f"Value for {key} is not a tuple"
        assert len(val) == 2, f"Value for {key} has {len(val)} elements, expected 2"


def test_oracle_table_all_models_in_champion_pool():
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    pool = set(module.AutoFitV732Wrapper._CHAMPION_POOL)
    for key, (primary, runner_up) in table.items():
        assert primary in pool, f"Primary {primary} for {key} not in champion pool"
        assert runner_up in pool, f"Runner-up {runner_up} for {key} not in champion pool"


# ---------- Test 5: Oracle lookup correctness ----------

def test_oracle_lookup_heavy_tail_h1_temporal():
    module = _load_module()
    result = module.AutoFitV732Wrapper._oracle_lookup("heavy_tail", 1, "core_only")
    assert result == ("NBEATS", "NHITS")


def test_oracle_lookup_heavy_tail_h7_temporal():
    module = _load_module()
    result = module.AutoFitV732Wrapper._oracle_lookup("heavy_tail", 7, "core_text")
    assert result == ("NHITS", "NBEATS")


def test_oracle_lookup_heavy_tail_h14_chronos():
    module = _load_module()
    result = module.AutoFitV732Wrapper._oracle_lookup("heavy_tail", 14, "core_only")
    assert result == ("Chronos", "NHITS")


def test_oracle_lookup_heavy_tail_h30_chronos():
    module = _load_module()
    result = module.AutoFitV732Wrapper._oracle_lookup("heavy_tail", 30, "full")
    assert result == ("Chronos", "NHITS")


def test_oracle_lookup_count_h1_temporal_kan():
    module = _load_module()
    result = module.AutoFitV732Wrapper._oracle_lookup("count", 1, "core_only")
    assert result == ("KAN", "NBEATS")


def test_oracle_lookup_count_h7_nbeats():
    module = _load_module()
    result = module.AutoFitV732Wrapper._oracle_lookup("count", 7, "core_edgar")
    assert result == ("NBEATS", "NHITS")


def test_oracle_lookup_binary_temporal_deepnpts():
    """Binary + temporal -> DeepNPTS for all horizons."""
    module = _load_module()
    for h in (1, 7, 14, 30):
        result = module.AutoFitV732Wrapper._oracle_lookup("binary", h, "core_only")
        assert result is not None, f"No oracle entry for binary/h={h}/core_only"
        assert result[0] == "DeepNPTS", (
            f"Expected DeepNPTS for binary/h={h}/core_only, got {result[0]}"
        )


def test_oracle_lookup_binary_exogenous_h1_patchtst():
    module = _load_module()
    result = module.AutoFitV732Wrapper._oracle_lookup("binary", 1, "core_edgar")
    assert result[0] == "PatchTST"


def test_oracle_lookup_binary_exogenous_h7_nhits():
    module = _load_module()
    result = module.AutoFitV732Wrapper._oracle_lookup("binary", 7, "full")
    assert result[0] == "NHITS"


def test_oracle_lookup_general_returns_none():
    """General target type has no oracle entry -> returns None."""
    module = _load_module()
    result = module.AutoFitV732Wrapper._oracle_lookup("general", 7, "core_only")
    assert result is None


def test_oracle_lookup_unseen_horizon_returns_none():
    """Unseen horizon (e.g., h=60) has no oracle entry -> returns None."""
    module = _load_module()
    result = module.AutoFitV732Wrapper._oracle_lookup("heavy_tail", 60, "core_only")
    assert result is None


# ---------- Test 6: Structural oracle -- component justification ----------

def test_nbeats_dominates_h1():
    """At h=1, NBEATS should be primary for heavy_tail (basis expansion)."""
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    for abl_cls in ("temporal", "exogenous"):
        key = ("heavy_tail", 1, abl_cls)
        assert table[key][0] == "NBEATS", f"Expected NBEATS at h=1/{abl_cls}"


def test_chronos_dominates_long_horizon():
    """At h>=14, Chronos should be primary for heavy_tail (pre-trained prior)."""
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    for h in (14, 30):
        for abl_cls in ("temporal", "exogenous"):
            key = ("heavy_tail", h, abl_cls)
            assert table[key][0] == "Chronos", (
                f"Expected Chronos at h={h}/{abl_cls}, got {table[key][0]}"
            )


def test_nhits_at_h7():
    """At h=7, NHITS should be primary for heavy_tail (weekly periodicity)."""
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    for abl_cls in ("temporal", "exogenous"):
        key = ("heavy_tail", 7, abl_cls)
        assert table[key][0] == "NHITS", (
            f"Expected NHITS at h=7/{abl_cls}, got {table[key][0]}"
        )


def test_kan_at_count_h1():
    """At h=1 count, KAN should be primary (B-spline activations)."""
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    for abl_cls in ("temporal", "exogenous"):
        key = ("count", 1, abl_cls)
        assert table[key][0] == "KAN", (
            f"Expected KAN at count/h=1/{abl_cls}, got {table[key][0]}"
        )


def test_nbeats_at_count_long():
    """At h>=7 count, NBEATS should be primary (Fourier seasonality)."""
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    for h in (7, 14, 30):
        for abl_cls in ("temporal", "exogenous"):
            key = ("count", h, abl_cls)
            assert table[key][0] == "NBEATS", (
                f"Expected NBEATS at count/h={h}/{abl_cls}, got {table[key][0]}"
            )


def test_deepnpts_for_binary_temporal():
    """Binary + no EDGAR -> DeepNPTS (non-parametric weighting)."""
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    for h in (1, 7, 14, 30):
        key = ("binary", h, "temporal")
        assert table[key][0] == "DeepNPTS", (
            f"Expected DeepNPTS at binary/h={h}/temporal, got {table[key][0]}"
        )


def test_patchtst_for_binary_exogenous_h1_h14():
    """Binary + EDGAR at h=1,14 -> PatchTST (patching + attention)."""
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    for h in (1, 14):
        key = ("binary", h, "exogenous")
        assert table[key][0] == "PatchTST", (
            f"Expected PatchTST at binary/h={h}/exogenous, got {table[key][0]}"
        )


# ---------- Test 7: Fallback routing ----------

def test_fallback_routing_has_3_buckets():
    module = _load_module()
    fb = module.AutoFitV732Wrapper._FALLBACK_ROUTING
    assert set(fb.keys()) == {"short", "medium", "long"}


def test_fallback_routing_each_has_at_least_3():
    module = _load_module()
    fb = module.AutoFitV732Wrapper._FALLBACK_ROUTING
    for bucket, candidates in fb.items():
        assert len(candidates) >= 3, f"Fallback {bucket} has < 3 candidates"


# ---------- Test 8: V732 registration ----------

def test_v732_in_autofit_models():
    module = _load_module()
    assert "AutoFitV732" in module.AUTOFIT_MODELS


def test_v732_factory():
    module = _load_module()
    v732 = module.get_autofit_v732(top_k=3)
    assert v732.config.name == "AutoFitV732"
    assert isinstance(v732, module.AutoFitV732Wrapper)
    assert v732._top_k == 3


def test_v732_config_params():
    module = _load_module()
    v732 = module.AutoFitV732Wrapper(top_k=4, val_fraction=0.15)
    assert v732.config.params["version"] == "7.3.2"
    assert v732.config.params["strategy"] == "structural_oracle_router"
    assert v732._val_fraction == 0.15


# ---------- Test 9: Champion pool completeness ----------

def test_champion_pool_has_8_models():
    module = _load_module()
    assert len(module.AutoFitV732Wrapper._CHAMPION_POOL) == 8


def test_champion_pool_models():
    module = _load_module()
    expected = {"NBEATS", "Chronos", "NHITS", "KAN", "DeepNPTS",
                "PatchTST", "NBEATSx", "DLinear"}
    actual = set(module.AutoFitV732Wrapper._CHAMPION_POOL)
    assert actual == expected, f"Pool mismatch: {actual ^ expected}"


# ---------- Test 10: Oracle / runner-up pairs never duplicate ----------

def test_oracle_primary_differs_from_runner_up():
    module = _load_module()
    table = module.AutoFitV732Wrapper._ORACLE_TABLE
    for key, (primary, runner_up) in table.items():
        assert primary != runner_up, (
            f"Primary == runner_up ({primary}) for {key}"
        )


# ---------- Test 11: Determinism check -- same conditions -> same oracle ----------

def test_oracle_deterministic():
    """Oracle lookup is purely functional -- calling twice gives same result."""
    module = _load_module()
    cls = module.AutoFitV732Wrapper
    for target_type in ("heavy_tail", "count", "binary"):
        for h in (1, 7, 14, 30):
            for abl in ("core_only", "core_text", "core_edgar", "full"):
                r1 = cls._oracle_lookup(target_type, h, abl)
                r2 = cls._oracle_lookup(target_type, h, abl)
                assert r1 == r2, f"Non-deterministic for ({target_type}, {h}, {abl})"
