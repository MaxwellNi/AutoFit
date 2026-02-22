#!/usr/bin/env python3
"""Schema tests for V7.2 routing telemetry in shard metrics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "run_block3_benchmark_shard.py"
    spec = importlib.util.spec_from_file_location("run_block3_benchmark_shard", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


class _DummyModel:
    def get_routing_info(self):
        return {
            "lane_family": "count",
            "champion_template": {"lane": "count", "horizon_band": "short", "primary": "NBEATS"},
            "hazard_calibration_method": "discrete_time_hazard:isotonic",
            "tail_pinball_q90": 12.34,
            "time_consistency_violation_rate": 0.0,
            "lane_clip_rate": 0.1,
            "inverse_transform_guard_hits": 2,
            "anchor_models_used": ["NBEATS", "NHITS"],
            "policy_action_id": "policy-v72",
            "oof_guard_triggered": False,
        }


def test_extract_routing_signals_includes_v72_fields():
    module = _load_module()
    signals = module.BenchmarkShard._extract_routing_signals(_DummyModel())

    assert signals["lane_family"] == "count"
    assert signals["champion_template"] == "lane=count|band=short|primary=NBEATS"
    assert signals["hazard_calibration_method"] == "discrete_time_hazard:isotonic"
    assert signals["tail_pinball_q90"] == 12.34
    assert signals["time_consistency_violation_rate"] == 0.0
    assert signals["lane_clip_rate"] == 0.1
    assert signals["inverse_transform_guard_hits"] == 2
    assert signals["anchor_models_used"] == ["NBEATS", "NHITS"]
    assert signals["policy_action_id"] == "policy-v72"
    assert signals["oof_guard_triggered"] is False

