#!/usr/bin/env python3
"""Unit tests for build_block3_execution_status.py."""

import importlib.util
import json
import sys
from pathlib import Path



def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_block3_execution_status.py"
    spec = importlib.util.spec_from_file_location("build_block3_execution_status", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_nested_tree_expected_key_count_is_104():
    module = _load_module()
    tp = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "block3_truth_pack"
    inventory = module._read_csv(tp / "condition_inventory_full.csv")
    expected = module._expected_keys_from_inventory(inventory)
    assert len(expected) == 104

    strict_done = module._strict_done_set(inventory)
    missing = module._missing_keys_from_manifest(module._read_csv(tp / "missing_key_manifest.csv"))
    v72_done = set(expected) - missing

    tree = module._build_nested_tree(expected, strict_done, v72_done, [])
    total_tree = sum(int(tree[t]["expected_keys"]) for t in ["task1_outcome", "task2_forecast", "task3_risk_adjust"])
    assert total_tree == 104


def test_v72_coverage_matches_missing_manifest_summary():
    module = _load_module()
    tp = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "block3_truth_pack"

    inventory = module._read_csv(tp / "condition_inventory_full.csv")
    expected = module._expected_keys_from_inventory(inventory)
    missing = module._missing_keys_from_manifest(module._read_csv(tp / "missing_key_manifest.csv"))

    summary = json.loads((tp / "missing_key_manifest_summary.json").read_text(encoding="utf-8"))
    assert len(expected) == int(summary["expected_keys"])
    assert len(missing) == int(summary["missing_keys"])
    assert len(expected) - len(missing) == int(summary["v72_strict_keys"])


def test_duration_parser_formats():
    module = _load_module()
    assert module._parse_duration_hours("1-02:30:00") == 26.5
    assert module._parse_duration_hours("12:30:00") == 12.5
    assert module._parse_duration_hours("06:15") == 6.25


def test_queue_actions_keep_required_groups():
    module = _load_module()
    jobs = [
        {
            "job_id": "1",
            "job_name": "p7v72c_t1_ce",
            "state": "PENDING",
            "partition": "batch",
            "group": "autofit_v72_completion",
        },
        {
            "job_id": "2",
            "job_name": "p7r_af1_t1_ce",
            "state": "RUNNING",
            "partition": "batch",
            "group": "autofit_resubmit",
        },
        {
            "job_id": "3",
            "job_name": "p7xF_fdr_t2_fu",
            "state": "PENDING",
            "partition": "gpu",
            "group": "foundation_reference",
        },
    ]
    payload = module._build_queue_actions(jobs)
    by_name = {r["job_name"]: r for r in payload["actions"]}

    assert by_name["p7v72c_t1_ce"]["action"] == "keep_priority"
    assert by_name["p7r_af1_t1_ce"]["action"] == "keep_priority"
    assert by_name["p7xF_fdr_t2_fu"]["action"] == "deprioritize_hold_recommended"
