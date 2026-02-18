#!/usr/bin/env python3
"""Unit tests for V7.2 hyperparameter-search artifact builder."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_v72_hyperparam_search.py"
    spec = importlib.util.spec_from_file_location("run_v72_hyperparam_search", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_target_family_mapping():
    module = _load_module()
    assert module._target_family("investors_count") == "count"
    assert module._target_family("funding_raised_usd") == "heavy_tail"
    assert module._target_family("is_funded") == "binary"


def test_plan_artifacts_include_search_budget():
    module = _load_module()
    strict_rows = [
        {
            "model_name": "AutoFitV72",
            "category": "autofit",
            "target": "investors_count",
            "mae": 44.8,
            "_source_path": "runs/benchmarks/a/metrics.json",
            "train_time_seconds": 12.0,
            "inference_time_seconds": 1.0,
        },
        {
            "model_name": "NHITS",
            "category": "deep_classical",
            "target": "investors_count",
            "mae": 44.7,
            "_source_path": "runs/benchmarks/b/metrics.json",
            "train_time_seconds": 120.0,
            "inference_time_seconds": 6.0,
        },
    ]

    ledger_rows, best_cfg_json, cost_rows = module._build_plan_artifacts(strict_rows, search_budget=96)
    assert ledger_rows
    assert all(int(r["search_budget"]) == 96 for r in ledger_rows)
    assert all(r["selection_scope"] == "train_val_oof_only" for r in ledger_rows)
    assert "targets" in best_cfg_json
    assert "investors_count" in best_cfg_json["targets"]
    assert cost_rows
