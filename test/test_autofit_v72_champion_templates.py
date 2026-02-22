#!/usr/bin/env python3
"""Unit tests for AutoFit V7.2 champion template routing helpers."""
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


def test_champion_template_count_lane_defaults():
    module = _load_module()
    tmpl = module._champion_template_for_lane("count", 7)
    assert tmpl["min_anchors"] == 1
    assert abs(float(tmpl["max_degrade_mult"]) - 1.02) < 1e-12
    assert tmpl["candidates"][:4] == ["NBEATS", "NHITS", "KAN", "PatchTST"]


def test_champion_template_binary_lane_requires_dual_anchors():
    module = _load_module()
    tmpl = module._champion_template_for_lane("binary", 14)
    assert tmpl["min_anchors"] == 2
    assert tmpl["candidates"][:2] == ["PatchTST", "NHITS"]


def test_champion_template_heavy_tail_horizon_order_changes():
    module = _load_module()
    short_tmpl = module._champion_template_for_lane("heavy_tail", 1)
    long_tmpl = module._champion_template_for_lane("heavy_tail", 30)
    assert short_tmpl["candidates"][:3] == ["NHITS", "PatchTST", "Chronos"]
    assert long_tmpl["candidates"][:3] == ["Chronos", "PatchTST", "NHITS"]


def test_champion_anchor_candidates_are_deduped():
    module = _load_module()
    cands = module._champion_anchor_candidates("heavy_tail", 30)
    assert cands[0] == "Chronos"
    assert len(cands) == len(set(cands))


def test_missingness_bucket_inference():
    module = _load_module()
    X_aug = pd.DataFrame({"__n_missing__": [1, 2, 3, 4, 5, 6]})
    bucket, stats = module._infer_missingness_bucket(X_aug, n_base_features=10)
    assert bucket in {"low", "medium", "high"}
    assert "q33" in stats and "q66" in stats and "median_ratio" in stats


def test_binary_calibrator_returns_supported_name():
    module = _load_module()
    n = 200
    y = np.array([0] * 100 + [1] * 100)
    raw = np.linspace(0.1, 0.9, n)
    cal, name, score, scores = module._fit_binary_calibrator(raw, y)
    assert name in {None, "platt", "isotonic"}
    if name is not None:
        assert cal is not None
        assert score is not None
        assert isinstance(scores, dict)


def test_discrete_time_hazard_head_fit_and_apply():
    module = _load_module()
    n = 320
    y = np.array([0] * 190 + [1] * 130)
    raw = np.linspace(0.02, 0.98, n)
    calibrator, cal_name, cal_score, diag = module._fit_discrete_time_hazard_head(
        raw,
        y,
        horizon=14,
        mode="discrete_time_hazard",
    )
    assert cal_name in {"identity", "platt", "isotonic"}
    assert "scores" in diag
    assert "ece" in diag
    pred = module._apply_discrete_time_hazard_head(
        raw,
        calibrator=calibrator,
        calibrator_name=cal_name,
        horizon=14,
        mode="discrete_time_hazard",
    )
    assert pred.shape == raw.shape
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0.0) and np.all(pred <= 1.0)
    if cal_score is not None:
        assert isinstance(cal_score, float)


def test_sparse_moe_route_keeps_anchor_and_budget():
    module = _load_module()
    selected = ["PatchTST", "NHITS", "NBEATSx", "LightGBM"]
    sorted_by_adj = [
        ("PatchTST", 0.09),
        ("NHITS", 0.11),
        ("NBEATSx", 0.12),
        ("LightGBM", 0.18),
    ]
    div_scores = {"PatchTST": 0.1, "NHITS": 0.4, "NBEATSx": 0.2, "LightGBM": 0.05}
    route = module._build_sparse_moe_route(
        selected_names=selected,
        sorted_by_adj=sorted_by_adj,
        div_scores=div_scores,
        lane="binary",
        horizon_band="mid",
        missingness_bucket="medium",
        template_candidates=["PatchTST", "NHITS"],
        required_anchors=2,
        max_experts=3,
        temperature=0.45,
        min_weight=0.05,
    )
    active = route["active_experts"]
    assert len(active) <= 3
    assert "PatchTST" in active
    assert "NHITS" in active
    w = route["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-8
