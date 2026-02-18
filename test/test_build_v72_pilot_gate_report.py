#!/usr/bin/env python3
"""Unit tests for V7.2 pilot gate report builder."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_v72_pilot_gate_report.py"
    spec = importlib.util.spec_from_file_location("build_v72_pilot_gate_report", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_pilot_gate_report_handles_no_v72_data():
    module = _load_module()
    report = module._build_report([])
    assert report["overall_pass"] is False
    assert report["checks"]["fairness_pass_100"] is False


def test_pilot_gate_report_overlap_logic():
    module = _load_module()
    rows = [
        {
            "model_name": "AutoFitV7",
            "category": "autofit",
            "task": "task1_outcome",
            "ablation": "core_edgar",
            "target": "investors_count",
            "horizon": 7,
            "mae": 50.0,
            "fairness_pass": True,
            "prediction_coverage_ratio": 1.0,
            "_source_path": "runs/benchmarks/a/metrics.json",
        },
        {
            "model_name": "AutoFitV72",
            "category": "autofit",
            "task": "task1_outcome",
            "ablation": "core_edgar",
            "target": "investors_count",
            "horizon": 7,
            "mae": 45.0,
            "fairness_pass": True,
            "prediction_coverage_ratio": 1.0,
            "_source_path": "runs/benchmarks/b/metrics.json",
        },
        {
            "model_name": "NHITS",
            "category": "deep_classical",
            "task": "task1_outcome",
            "ablation": "core_edgar",
            "target": "investors_count",
            "horizon": 7,
            "mae": 40.0,
            "fairness_pass": True,
            "prediction_coverage_ratio": 1.0,
            "_source_path": "runs/benchmarks/c/metrics.json",
        },
    ]
    report = module._build_report(rows)
    assert report["counts"]["overlap_keys_v7_v72_non_autofit"] == 1
    assert report["metrics"]["investors_count_gap_reduction_pct"] is not None
