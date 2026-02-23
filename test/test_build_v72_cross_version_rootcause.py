#!/usr/bin/env python3
"""Tests for V7/V7.1/V7.2 cross-version root-cause artifact builder."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_v72_cross_version_rootcause.py"
    spec = importlib.util.spec_from_file_location("build_v72_cross_version_rootcause", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_snapshot_has_expected_core_keys():
    module = _load_module()
    tp = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "block3_truth_pack"
    snapshot = module._build_snapshot(tp)

    assert "strict_comparable_conditions" in snapshot
    assert "v72_coverage" in snapshot
    assert "gate_status" in snapshot
    assert "fairness_certification" in snapshot
    assert int(snapshot["strict_comparable_conditions"]["expected"]) == 104


def test_rootcause_matrix_contains_required_problem_ids():
    module = _load_module()
    tp = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "block3_truth_pack"
    snapshot = module._build_snapshot(tp)
    rows = module._build_rootcause_matrix(snapshot)
    ids = {r["problem_id"] for r in rows}

    assert "count_lane_extreme_error_lineage" in ids
    assert "count_lane_median_gap_persistent" in ids
    assert "binary_lane_calibration_and_routing_gap" in ids
    assert "coverage_deficit_blocks_full_claim" in ids
    assert "audit_not_certified_due_spike_lineage" in ids


def test_frontier_fix_map_has_primary_links():
    module = _load_module()
    rows = module._build_frontier_fix_map()
    assert len(rows) >= 8
    assert all(str(r.get("primary_link", "")).startswith("https://") for r in rows)
    assert all(r.get("status") == "verified_primary" for r in rows)


def test_script_writes_expected_artifacts(tmp_path):
    module = _load_module()

    # seed minimal truth-pack files required by snapshot builder
    tp = tmp_path / "tp"
    tp.mkdir(parents=True, exist_ok=True)

    (tp / "condition_leaderboard.csv").write_text(
        "task,ablation,target,horizon,expected_condition,condition_completed,n_records,best_model,best_category,best_mae,best_non_autofit_model,best_non_autofit_category,best_non_autofit_mae,best_autofit_model,best_autofit_variant_id,best_autofit_mae,autofit_gap_pct,bench_dirs,sources\n"
        "task1_outcome,core_only,investors_count,1,True,True,2,NBEATS,deep_classical,44.0,NBEATS,deep_classical,44.0,AutoFitV72,autofitv72,120.0,172.7,run_a,metrics_a\n",
        encoding="utf-8",
    )
    (tp / "failure_taxonomy.csv").write_text(
        "issue_type,severity,model_name,category,task,ablation,target,horizon,mae,prediction_coverage_ratio,fairness_pass,evidence_source,note\n"
        "v71_count_explosion,critical,AutoFitV71,autofit,task1_outcome,core_edgar,investors_count,1,1.0,1.0,True,path,note\n",
        encoding="utf-8",
    )
    (tp / "v71_vs_v7_overlap.csv").write_text(
        "task,ablation,target,horizon,mae_v7,mae_v71,relative_gain_pct,v71_wins,source_v7,source_v71\n"
        "task1_outcome,core_only,investors_count,1,100,90,10,True,a,b\n",
        encoding="utf-8",
    )
    (tp / "autofit_lineage.csv").write_text(
        "model_name,target,n_records,conditions_covered,condition_coverage_ratio,best_mae,median_mae,p25_mae,p75_mae,worst_mae,median_gap_vs_best_non_autofit_pct\n"
        "AutoFitV72,investors_count,1,1,0.01,100,100,100,100,100,120\n",
        encoding="utf-8",
    )
    (tp / "v72_pilot_gate_report.json").write_text(
        json.dumps(
            {
                "overall_pass": False,
                "checks": {"fairness_pass_100": True},
                "metrics": {
                    "global_normalized_mae_improvement_pct": 1.0,
                    "investors_count_gap_reduction_pct": -2.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (tp / "fairness_certification_latest.json").write_text(
        json.dumps({"overall_certified": False, "label": "NOT CERTIFIED"}),
        encoding="utf-8",
    )
    (tp / "missing_key_manifest_summary.json").write_text(
        json.dumps({"v72_strict_keys": 30, "missing_keys": 74}),
        encoding="utf-8",
    )

    snapshot = module._build_snapshot(tp)
    root = module._build_rootcause_matrix(snapshot)
    frontier = module._build_frontier_fix_map()

    module._write_csv(
        tp / "v72_cross_version_rootcause_matrix.csv",
        root,
        [
            "problem_id",
            "introduced_or_observed_in",
            "still_unresolved_in_v72",
            "evidence_path",
            "impact_targets",
            "impact_scale",
            "root_mechanism",
            "fix_component",
            "gate_link",
        ],
    )
    module._write_csv(
        tp / "v72_frontier_fix_map_20260223.csv",
        frontier,
        [
            "problem",
            "source",
            "mechanism",
            "integration_point",
            "risk",
            "verification_test",
            "primary_link",
            "status",
        ],
    )

    assert (tp / "v72_cross_version_rootcause_matrix.csv").exists()
    assert (tp / "v72_frontier_fix_map_20260223.csv").exists()
