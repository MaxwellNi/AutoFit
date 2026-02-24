#!/usr/bin/env python3
"""Tests for key-level V7.2 completion job manifest builder."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_v72_missing_key_jobs.py"
    spec = importlib.util.spec_from_file_location("build_v72_missing_key_jobs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_build_manifest_rows_respects_memory_plan_override():
    module = _load_module()
    missing_rows = [
        {
            "task": "task2_forecast",
            "ablation": "core_text",
            "target": "investors_count",
            "horizon": 14,
            "priority_rank": 2,
            "priority_group": "P2_task2_core_only_text_full",
            "reason": "v72_strict_missing",
        }
    ]
    mem_plan = {
        "task2_forecast|core_text|investors_count|h14": {
            "memory_class": "XL",
            "predicted_peak_rss_gb": 310.0,
        }
    }
    rows = module.build_manifest_rows(
        missing_rows=missing_rows,
        mem_plan=mem_plan,
        output_root="runs/benchmarks/block3_20260203_225620_phase7_v72_completion",
        job_prefix="p7v72k",
        seed=42,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["job_name"] == "p7v72k_t2_ct_ic_h14"
    assert row["resource_class"] == "XL"
    assert row["partition"] == "bigmem"
    assert row["qos"] == "iris-bigmem-long"
    assert row["predicted_peak_rss_gb"] == 310.0


def test_default_classification_promotes_known_oom_lineage():
    module = _load_module()
    missing_rows = [
        {
            "task": "task1_outcome",
            "ablation": "core_text",
            "target": "funding_raised_usd",
            "horizon": 1,
            "priority_rank": 1,
            "priority_group": "P1_task1_core_text_full_core_edgar",
            "reason": "v72_strict_missing",
        },
        {
            "task": "task1_outcome",
            "ablation": "core_only",
            "target": "funding_raised_usd",
            "horizon": 1,
            "priority_rank": 4,
            "priority_group": "P4_other",
            "reason": "v72_strict_missing",
        },
    ]
    rows = module.build_manifest_rows(
        missing_rows=missing_rows,
        mem_plan={},
        output_root="runs/benchmarks/block3_20260203_225620_phase7_v72_completion",
        job_prefix="p7v72k",
        seed=42,
    )
    by_key = {f"{r['task']}|{r['ablation']}": r for r in rows}
    assert by_key["task1_outcome|core_text"]["resource_class"] == "XL"
    assert by_key["task1_outcome|core_only"]["resource_class"] == "L"
