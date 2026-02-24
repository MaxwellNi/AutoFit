#!/usr/bin/env python3
"""Tests for offline policy dataset/training scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load(name: str):
    path = Path(__file__).resolve().parent.parent / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_policy_dataset_builder_rows(tmp_path):
    build_mod = _load("build_v72_policy_dataset.py")
    tp = tmp_path / "tp"
    tp.mkdir(parents=True, exist_ok=True)
    (tp / "condition_leaderboard.csv").write_text(
        "task,ablation,target,horizon,condition_completed,best_model,autofit_gap_pct\n"
        "task1_outcome,core_edgar,investors_count,7,True,NBEATS,120.0\n"
        "task1_outcome,full,is_funded,14,True,PatchTST,90.0\n",
        encoding="utf-8",
    )
    (tp / "failure_taxonomy.csv").write_text(
        "issue_type,task,ablation,target,horizon\n"
        "v71_count_explosion,task1_outcome,core_edgar,investors_count,7\n",
        encoding="utf-8",
    )
    rows = build_mod.build_rows(tp)
    assert len(rows) == 2
    count_row = next(r for r in rows if r["target"] == "investors_count")
    assert count_row["route_key"].startswith("lane=count|")
    assert count_row["template_id"] == "count_nbeats_nhits_kan"


def test_policy_training_report_has_state_policy(tmp_path):
    train_mod = _load("train_v72_offline_policy.py")
    dataset = tmp_path / "v72_policy_dataset.csv"
    dataset.write_text(
        "route_key,template_id,candidate_subset,count_family,binary_calibration_mode,top_k,reward\n"
        "lane=count|hb=short|ablation=core_edgar|miss=mid,count_nbeats_nhits_kan,NBEATS,N/A,N/A,10,1.5\n"
        "lane=count|hb=short|ablation=core_edgar|miss=mid,count_nbeats_nhits_kan,NBEATS,N/A,N/A,10,1.0\n"
        "lane=count|hb=short|ablation=core_edgar|miss=mid,count_alt,ALT,N/A,N/A,8,0.5\n",
        encoding="utf-8",
    )
    rows = train_mod._load_rows(dataset)
    policy = train_mod._build_policy(rows)
    assert "lane=count|hb=short|ablation=core_edgar|miss=mid" in policy
    state = policy["lane=count|hb=short|ablation=core_edgar|miss=mid"]
    assert state["template_id"] == "count_nbeats_nhits_kan"
