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
