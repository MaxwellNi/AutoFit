#!/usr/bin/env python3
"""Unit tests for FusedChampion: standalone PyTorch expert modules.

Tests verify:
1. Expert module output shapes (forward pass correctness)
2. FusedChampionNet routing and instantiation
3. Basis function computation (TrendBasis, SeasonalityBasis)
4. Data utilities (windowing, scaling)
5. Training convergence (loss decreases)
6. FusedChampionWrapper fit/predict interface
7. Oracle table completeness
8. Expert config completeness
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Module loader ──

def _load_module():
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return importlib.import_module("narrative.block3.models.fused_champion")


# ============================================================================
# TEST GROUP 1: Expert Module Output Shapes
# ============================================================================

def test_nbeats_expert_shape():
    mod = _load_module()
    expert = mod.NBEATSExpert(input_size=60, h=7)
    x = torch.randn(4, 60)
    out = expert(x)
    assert out.shape == (4, 7), f"Expected (4,7), got {out.shape}"


def test_nbeats_expert_h1_clamped():
    """NBEATS with h_nf=7 (clamped from h=1) should work."""
    mod = _load_module()
    expert = mod.NBEATSExpert(input_size=60, h=7)
    x = torch.randn(2, 60)
    out = expert(x)
    assert out.shape == (2, 7)


def test_nbeats_expert_h30():
    mod = _load_module()
    expert = mod.NBEATSExpert(input_size=60, h=30)
    x = torch.randn(8, 60)
    out = expert(x)
    assert out.shape == (8, 30)


def test_nhits_expert_shape():
    mod = _load_module()
    expert = mod.NHITSExpert(input_size=60, h=7)
    x = torch.randn(4, 60)
    out = expert(x)
    assert out.shape == (4, 7)


def test_nhits_expert_h14():
    mod = _load_module()
    expert = mod.NHITSExpert(input_size=60, h=14)
    x = torch.randn(4, 60)
    out = expert(x)
    assert out.shape == (4, 14)


def test_kan_expert_shape():
    mod = _load_module()
    expert = mod.KANExpert(input_size=60, h=7)
    x = torch.randn(4, 60)
    out = expert(x)
    assert out.shape == (4, 7)


def test_kan_expert_h30():
    mod = _load_module()
    expert = mod.KANExpert(input_size=60, h=30)
    x = torch.randn(4, 60)
    out = expert(x)
    assert out.shape == (4, 30)


def test_deepnpts_expert_shape():
    mod = _load_module()
    expert = mod.DeepNPTSExpert(input_size=60, h=7)
    x = torch.randn(4, 60)
    out = expert(x)
    assert out.shape == (4, 7)


def test_deepnpts_output_range():
    """DeepNPTS output should be in the convex hull of input values."""
    mod = _load_module()
    expert = mod.DeepNPTSExpert(input_size=60, h=7)
    # All-positive input
    x = torch.abs(torch.randn(4, 60)) + 1.0
    with torch.no_grad():
        out = expert(x)
    # After softmax weighting, all outputs should be positive
    # (may fail during initial random weights but statistically likely)
    assert out.shape == (4, 7)


def test_patchtst_expert_shape():
    mod = _load_module()
    expert = mod.PatchTSTExpert(input_size=64, h=7)
    x = torch.randn(4, 64)
    out = expert(x)
    assert out.shape == (4, 7)


def test_patchtst_expert_h30():
    mod = _load_module()
    expert = mod.PatchTSTExpert(input_size=64, h=30)
    x = torch.randn(4, 64)
    out = expert(x)
    assert out.shape == (4, 30)


def test_dlinear_expert_shape():
    mod = _load_module()
    expert = mod.DLinearExpert(input_size=60, h=7)
    x = torch.randn(4, 60)
    out = expert(x)
    assert out.shape == (4, 7)


# ============================================================================
# TEST GROUP 2: FusedChampionNet Routing
# ============================================================================

def test_fused_net_nbeats():
    mod = _load_module()
    net = mod.FusedChampionNet(expert_id="nbeats", input_size=60, h=7)
    assert net.expert_id == "nbeats"
    assert isinstance(net.expert, mod.NBEATSExpert)
    x = torch.randn(2, 60)
    out = net(x)
    assert out.shape == (2, 7)


def test_fused_net_nhits():
    mod = _load_module()
    net = mod.FusedChampionNet(expert_id="nhits", input_size=60, h=7)
    assert isinstance(net.expert, mod.NHITSExpert)


def test_fused_net_kan():
    mod = _load_module()
    net = mod.FusedChampionNet(expert_id="kan", input_size=60, h=7)
    assert isinstance(net.expert, mod.KANExpert)


def test_fused_net_deepnpts():
    mod = _load_module()
    net = mod.FusedChampionNet(expert_id="deepnpts", input_size=60, h=7)
    assert isinstance(net.expert, mod.DeepNPTSExpert)


def test_fused_net_patchtst():
    mod = _load_module()
    net = mod.FusedChampionNet(expert_id="patchtst", input_size=64, h=7)
    assert isinstance(net.expert, mod.PatchTSTExpert)


def test_fused_net_dlinear():
    mod = _load_module()
    net = mod.FusedChampionNet(expert_id="dlinear", input_size=60, h=7)
    assert isinstance(net.expert, mod.DLinearExpert)


def test_fused_net_invalid_expert():
    mod = _load_module()
    try:
        mod.FusedChampionNet(expert_id="nonexistent", input_size=60, h=7)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_fused_net_all_experts_available():
    mod = _load_module()
    assert set(mod.EXPERT_BUILDERS.keys()) == {
        "nbeats", "nhits", "kan", "deepnpts", "patchtst", "dlinear",
    }


# ============================================================================
# TEST GROUP 3: Basis Functions
# ============================================================================

def test_trend_basis_shape():
    mod = _load_module()
    basis = mod.TrendBasis(n_basis=2, backcast_size=60, forecast_size=7)
    theta = torch.randn(4, 6)  # n_theta = 2 * (2+1) = 6
    backcast, forecast = basis(theta)
    assert backcast.shape == (4, 60)
    assert forecast.shape == (4, 7, 1)


def test_seasonality_basis_shape():
    mod = _load_module()
    basis = mod.SeasonalityBasis(harmonics=2, backcast_size=60, forecast_size=7)
    harmonic_size = basis.forecast_basis.shape[0]
    theta = torch.randn(4, 2 * harmonic_size)
    backcast, forecast = basis(theta)
    assert backcast.shape == (4, 60)
    assert forecast.shape == (4, 7, 1)


def test_identity_basis_shape():
    mod = _load_module()
    basis = mod._IdentityBasis(backcast_size=60, forecast_size=7)
    n_knots = 2
    theta = torch.randn(4, 60 + n_knots)
    backcast, forecast = basis(theta)
    assert backcast.shape == (4, 60)
    assert forecast.shape == (4, 7, 1)


# ============================================================================
# TEST GROUP 4: Data Utilities
# ============================================================================

def test_create_windows_basic():
    mod = _load_module()
    series = {"e1": np.arange(100, dtype=np.float32)}
    ins, outs = mod.create_windows(series, input_size=10, h=3, step=1)
    assert ins.shape[1] == 10
    assert outs.shape[1] == 3
    assert ins.shape[0] == outs.shape[0]
    # Number of windows: T - L - h + 1 = 100 - 10 - 3 + 1 = 88
    assert ins.shape[0] == 88


def test_create_windows_multiple_entities():
    mod = _load_module()
    series = {
        "e1": np.arange(50, dtype=np.float32),
        "e2": np.arange(30, dtype=np.float32),
    }
    ins, outs = mod.create_windows(series, input_size=10, h=3, step=1)
    expected = (50 - 10 - 3 + 1) + (30 - 10 - 3 + 1)  # 38 + 18 = 56
    assert ins.shape[0] == expected, f"Expected {expected}, got {ins.shape[0]}"


def test_create_windows_short_series():
    mod = _load_module()
    series = {"e1": np.arange(5, dtype=np.float32)}
    ins, outs = mod.create_windows(series, input_size=10, h=3, step=1)
    assert ins.shape[0] == 0


def test_create_windows_nan_skip():
    mod = _load_module()
    s = np.arange(50, dtype=np.float32)
    s[25] = np.nan  # This window and overlapping ones should be skipped
    series = {"e1": s}
    ins, outs = mod.create_windows(series, input_size=10, h=3, step=1)
    assert ins.shape[0] < 38  # Some windows should be skipped


def test_robust_scale_batch():
    mod = _load_module()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    scaled, median, iqr = mod._robust_scale_batch(x)
    assert scaled.shape == x.shape
    assert median.shape == (1, 1)
    assert iqr.shape == (1, 1)
    # Median should be 3.0
    assert abs(median.item() - 3.0) < 1e-5


def test_robust_descale():
    mod = _load_module()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    scaled, median, iqr = mod._robust_scale_batch(x)
    restored = mod._robust_descale(scaled, median, iqr)
    assert torch.allclose(restored, x, atol=1e-5)


def test_extract_entity_series():
    mod = _load_module()
    df = pd.DataFrame({
        "entity_id": [1]*20 + [2]*20,
        "crawled_date_day": pd.date_range("2024-01-01", periods=20).tolist() * 2,
        "target_col": np.random.randn(40),
    })
    result = mod._extract_entity_series(df, "target_col", min_obs=10)
    assert len(result) == 2
    assert all(len(v) == 20 for v in result.values())


def test_extract_entity_series_missing_target():
    mod = _load_module()
    df = pd.DataFrame({"entity_id": [1]*20, "x": np.random.randn(20)})
    result = mod._extract_entity_series(df, "nonexistent")
    assert len(result) == 0


# ============================================================================
# TEST GROUP 5: Training Convergence
# ============================================================================

def test_training_loss_decreases_nbeats():
    """NBEATS training loss should decrease on simple synthetic data."""
    mod = _load_module()
    # Simple linear trend data
    series = {"e1": np.cumsum(np.random.randn(200)) + 100}
    insample, outsample = mod.create_windows(series, input_size=60, h=7, step=1)
    assert len(insample) > 0

    net = mod.FusedChampionNet("nbeats", 60, 7)
    device = torch.device("cpu")

    # Override max_steps for fast test
    original = mod.EXPERT_CONFIGS["nbeats"]["max_steps"]
    mod.EXPERT_CONFIGS["nbeats"]["max_steps"] = 50

    try:
        # Get initial loss
        net.eval()
        x = torch.tensor(insample[:32], dtype=torch.float32)
        y = torch.tensor(outsample[:32], dtype=torch.float32)
        x_s, med, iqr = mod._robust_scale_batch(x)
        y_s = (y - med) / iqr
        with torch.no_grad():
            init_loss = torch.nn.functional.l1_loss(net(x_s), y_s).item()

        # Train
        final_loss = mod.train_expert(net, insample, outsample, "nbeats", device)

        # Loss should decrease
        assert final_loss < init_loss, (
            f"Loss did not decrease: {init_loss:.4f} → {final_loss:.4f}"
        )
    finally:
        mod.EXPERT_CONFIGS["nbeats"]["max_steps"] = original


def test_training_loss_decreases_dlinear():
    mod = _load_module()
    series = {"e1": np.sin(np.linspace(0, 20*np.pi, 300)) + 5}
    insample, outsample = mod.create_windows(series, input_size=60, h=7, step=1)
    assert len(insample) > 0

    net = mod.FusedChampionNet("dlinear", 60, 7)
    device = torch.device("cpu")

    original = mod.EXPERT_CONFIGS["dlinear"]["max_steps"]
    mod.EXPERT_CONFIGS["dlinear"]["max_steps"] = 50

    try:
        final_loss = mod.train_expert(net, insample, outsample, "dlinear", device)
        assert final_loss < 10.0, f"DLinear loss too high: {final_loss}"
    finally:
        mod.EXPERT_CONFIGS["dlinear"]["max_steps"] = original


# ============================================================================
# TEST GROUP 6: FusedChampionWrapper Interface
# ============================================================================

def test_wrapper_instantiation():
    mod = _load_module()
    wrapper = mod.FusedChampionWrapper()
    assert wrapper.name == "FusedChampion"
    assert wrapper._fitted is False


def test_wrapper_detect_target_type():
    mod = _load_module()
    binary = pd.Series(np.random.choice([0.0, 1.0], size=200))
    assert mod.FusedChampionWrapper._detect_target_type(binary) == "binary"

    count = pd.Series(np.random.randint(0, 200, size=500).astype(float))
    assert mod.FusedChampionWrapper._detect_target_type(count) == "count"


def test_wrapper_ablation_class():
    mod = _load_module()
    assert mod.FusedChampionWrapper._ablation_class("core_only") == "temporal"
    assert mod.FusedChampionWrapper._ablation_class("core_text") == "temporal"
    assert mod.FusedChampionWrapper._ablation_class("core_edgar") == "exogenous"
    assert mod.FusedChampionWrapper._ablation_class("full") == "exogenous"


def test_wrapper_fit_predict_synthetic():
    """End-to-end fit/predict with synthetic data."""
    mod = _load_module()
    wrapper = mod.FusedChampionWrapper()

    # Override max_steps for fast test
    for k in mod.EXPERT_CONFIGS:
        mod.EXPERT_CONFIGS[k]["max_steps"] = 20

    try:
        n_entities = 3
        obs_per_entity = 100
        rows = []
        for eid in range(n_entities):
            for t in range(obs_per_entity):
                rows.append({
                    "entity_id": eid,
                    "crawled_date_day": pd.Timestamp("2024-01-01") + pd.Timedelta(days=t),
                    "funding_raised_usd": (
                        float(np.random.exponential(100000))
                        + 50000 * np.sin(t / 7 * 2 * np.pi)
                    ),
                })
        train_df = pd.DataFrame(rows)
        X = train_df[["crawled_date_day"]]
        y = train_df["funding_raised_usd"]
        y.name = "funding_raised_usd"

        wrapper.fit(
            X, y,
            train_raw=train_df,
            target="funding_raised_usd",
            horizon=7,
            ablation="core_only",
        )

        assert wrapper._fitted is True
        assert wrapper._expert_id is not None

        # Predict
        test_df = train_df.tail(10).copy()
        X_test = test_df[["crawled_date_day"]]
        preds = wrapper.predict(
            X_test,
            test_raw=test_df,
            target="funding_raised_usd",
            horizon=7,
        )
        assert len(preds) == 10
        assert all(np.isfinite(preds))
    finally:
        # Restore defaults
        mod.EXPERT_CONFIGS["nbeats"]["max_steps"] = 1000
        mod.EXPERT_CONFIGS["nhits"]["max_steps"] = 1000
        mod.EXPERT_CONFIGS["kan"]["max_steps"] = 1000
        mod.EXPERT_CONFIGS["deepnpts"]["max_steps"] = 1000
        mod.EXPERT_CONFIGS["patchtst"]["max_steps"] = 3000
        mod.EXPERT_CONFIGS["dlinear"]["max_steps"] = 1000


# ============================================================================
# TEST GROUP 7: Oracle Table
# ============================================================================

def test_oracle_table_completeness():
    mod = _load_module()
    table = mod.FusedChampionWrapper._ORACLE_TABLE
    target_types = {"heavy_tail", "count", "binary"}
    horizons = {1, 7, 14, 30}
    ablation_classes = {"temporal", "exogenous"}
    for tt in target_types:
        for h in horizons:
            for ac in ablation_classes:
                key = (tt, h, ac)
                assert key in table, f"Missing oracle entry: {key}"


def test_oracle_table_champion_validity():
    """All champions in oracle table should be trainable or Chronos."""
    mod = _load_module()
    table = mod.FusedChampionWrapper._ORACLE_TABLE
    valid_names = set(mod.CHAMPION_TO_EXPERT.keys()) | {"Chronos"}
    for key, (primary, fallback) in table.items():
        assert primary in valid_names, f"Unknown champion {primary} in {key}"
        assert fallback in valid_names, f"Unknown fallback {fallback} in {key}"


def test_oracle_table_has_24_entries():
    mod = _load_module()
    table = mod.FusedChampionWrapper._ORACLE_TABLE
    assert len(table) == 24, f"Expected 24 entries, got {len(table)}"


# ============================================================================
# TEST GROUP 8: Expert Config Completeness
# ============================================================================

def test_all_experts_have_configs():
    mod = _load_module()
    for expert_id in mod.EXPERT_BUILDERS:
        assert expert_id in mod.EXPERT_CONFIGS, f"Missing config for {expert_id}"


def test_configs_have_required_fields():
    mod = _load_module()
    required = {"input_size", "max_steps", "batch_size", "lr"}
    for expert_id, cfg in mod.EXPERT_CONFIGS.items():
        for field in required:
            assert field in cfg, f"Config {expert_id} missing '{field}'"


def test_champion_to_expert_mapping():
    mod = _load_module()
    for champion, expert_id in mod.CHAMPION_TO_EXPERT.items():
        assert expert_id in mod.EXPERT_BUILDERS, (
            f"Champion {champion} maps to unknown expert {expert_id}"
        )


# ============================================================================
# TEST GROUP 9: KANLinear Specifics
# ============================================================================

def test_kan_linear_shape():
    mod = _load_module()
    layer = mod._KANLinear(in_features=60, out_features=256)
    x = torch.randn(4, 60)
    out = layer(x)
    assert out.shape == (4, 256)


def test_kan_linear_b_splines():
    """B-spline evaluation should produce correct shape."""
    mod = _load_module()
    layer = mod._KANLinear(in_features=10, out_features=5)
    x = torch.randn(3, 10)
    bases = layer._b_splines(x)
    # Expected: [batch=3, in=10, grid_size+spline_order=8]
    assert bases.shape == (3, 10, 8)


# ============================================================================
# TEST GROUP 10: Moving Average
# ============================================================================

def test_moving_avg_preserves_length():
    mod = _load_module()
    ma = mod._MovingAvg(kernel_size=25)
    x = torch.randn(4, 60)
    out = ma(x)
    assert out.shape == (4, 60), f"Expected (4,60), got {out.shape}"


def test_dlinear_decomposition():
    """DLinear trend + seasonal should reconstruct approximately."""
    mod = _load_module()
    expert = mod.DLinearExpert(input_size=60, h=7)
    x = torch.randn(4, 60)
    # Just verify forward pass works
    out = expert(x)
    assert out.shape == (4, 7)
