#!/usr/bin/env python3
"""Tests for V7.3 handoff pack builder."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_v73_handoff_pack.py"
    spec = importlib.util.spec_from_file_location("build_v73_handoff_pack", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_build_reuse_manifest_marks_missing_keys(tmp_path: Path):
    module = _load_module()
    tp = tmp_path / "tp"
    _write_csv(
        tp / "condition_inventory_full.csv",
        [
            {
                "task": "task1_outcome",
                "ablation": "core_only",
                "target": "investors_count",
                "horizon": 1,
                "strict_completed": "true",
            },
            {
                "task": "task1_outcome",
                "ablation": "core_only",
                "target": "investors_count",
                "horizon": 7,
                "strict_completed": "true",
            },
        ],
        ["task", "ablation", "target", "horizon", "strict_completed"],
    )
    _write_csv(
        tp / "condition_leaderboard.csv",
        [
            {
                "task": "task1_outcome",
                "ablation": "core_only",
                "target": "investors_count",
                "horizon": 1,
                "bench_dirs": "runs/benchmarks/block3_20260203_225620_phase7_v72_completion",
                "condition_completed": "true",
                "best_model": "NBEATS",
            },
            {
                "task": "task1_outcome",
                "ablation": "core_only",
                "target": "investors_count",
                "horizon": 7,
                "bench_dirs": "runs/benchmarks/block3_20260203_225620_phase7_v72_completion",
                "condition_completed": "true",
                "best_model": "NBEATS",
            },
        ],
        ["task", "ablation", "target", "horizon", "bench_dirs", "condition_completed", "best_model"],
    )
    _write_csv(
        tp / "missing_key_manifest.csv",
        [
            {
                "task": "task1_outcome",
                "ablation": "core_only",
                "target": "investors_count",
                "horizon": 7,
            }
        ],
        ["task", "ablation", "target", "horizon"],
    )

    rows = module._build_v73_reuse_manifest(tp)
    assert len(rows) == 2
    keep = next(r for r in rows if int(r["horizon"]) == 1)
    miss = next(r for r in rows if int(r["horizon"]) == 7)
    assert keep["needs_rerun"] == "false"
    assert keep["reuse_from_run"] != ""
    assert miss["needs_rerun"] == "true"
    assert miss["reuse_reason"] == "missing_from_v72_coverage_manifest"


def test_build_champion_component_map_generates_groups(tmp_path: Path):
    module = _load_module()
    tp = tmp_path / "tp"
    _write_csv(
        tp / "condition_leaderboard.csv",
        [
            {
                "task": "task1_outcome",
                "ablation": "core_edgar",
                "target": "investors_count",
                "horizon": 1,
                "condition_completed": "true",
                "best_model": "NBEATS",
            },
            {
                "task": "task1_outcome",
                "ablation": "core_edgar",
                "target": "investors_count",
                "horizon": 14,
                "condition_completed": "true",
                "best_model": "NHITS",
            },
            {
                "task": "task1_outcome",
                "ablation": "full",
                "target": "is_funded",
                "horizon": 7,
                "condition_completed": "true",
                "best_model": "PatchTST",
            },
        ],
        ["task", "ablation", "target", "horizon", "condition_completed", "best_model"],
    )
    rows = module._build_v73_champion_component_map(tp)
    assert len(rows) >= 2
    count_row = next(r for r in rows if r["target_family"] == "count")
    assert "NBEATS" in count_row["champion_models"] or "NHITS" in count_row["champion_models"]
    assert count_row["transfer_priority"] == "critical"


def test_build_rl_policy_spec_contains_no_test_feedback(tmp_path: Path):
    module = _load_module()
    tp = tmp_path / "tp"
    tp.mkdir(parents=True, exist_ok=True)
    (tp / "truth_pack_summary.json").write_text(
        json.dumps({"strict_completed_conditions": 104, "expected_conditions": 104, "v72_missing_keys": 10}),
        encoding="utf-8",
    )
    (tp / "v72_pilot_gate_report.json").write_text(
        json.dumps({"overall_pass": False, "counts": {"overlap_keys_v7_v72_non_autofit": 80}}),
        encoding="utf-8",
    )
    spec = module._build_v73_rl_policy_spec(tp)
    assert spec["constraints"]["selection_data_scope"] == "train_val_oof_only"
    assert spec["constraints"]["test_feedback_allowed"] is False

