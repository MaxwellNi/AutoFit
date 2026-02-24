#!/usr/bin/env python3
"""Tests for V7.2 memory estimator helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "estimate_block3_memory_requirements.py"
    spec = importlib.util.spec_from_file_location("estimate_block3_memory_requirements", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_parse_mem_units():
    module = _load_module()
    assert module._parse_req_mem_to_gb("160G") == 160.0
    assert round(module._parse_req_mem_to_gb("1024M"), 4) == 1.0
    assert module._parse_req_mem_to_gb("") is None


def test_admission_upgrade_threshold():
    module = _load_module()
    # requested L class=160G, threshold=128G
    assert module._memory_class_from_predicted(127.9) == "L"
    assert module._memory_class_from_predicted(128.1) == "XL"


def test_match_job_to_key_supports_key_level_name():
    module = _load_module()
    key = module.MissingKey(
        task="task3_risk_adjust",
        ablation="full",
        target="investors_count",
        horizon=30,
        priority_rank=3,
        priority_group="P3_task3_all_ablations",
    )
    assert module._match_job_to_key("p7v72k_t3_fu_ic_h30", key) is True
