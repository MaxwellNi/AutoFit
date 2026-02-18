#!/usr/bin/env python3
"""Tests for Block3 truth pack panorama builder."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_block3_truth_pack.py"
    spec = importlib.util.spec_from_file_location("build_block3_truth_pack", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_expected_condition_count_is_104():
    module = _load_module()
    config = Path(__file__).resolve().parent.parent / "configs" / "block3_tasks.yaml"
    conditions = module._load_expected_conditions(config)
    assert len(conditions) == 104


def test_strict_legacy_layer_classification():
    module = _load_module()

    strict_row = {
        "mae": 1.0,
        "fairness_pass": True,
        "prediction_coverage_ratio": 0.99,
    }
    legacy_row = {
        "mae": 1.0,
    }
    excluded_row = {
        "mae": 1.0,
        "fairness_pass": True,
        "prediction_coverage_ratio": 0.95,
    }

    assert module.classify_record_layer(strict_row, min_coverage=0.98) == "strict_comparable"
    assert module.classify_record_layer(legacy_row, min_coverage=0.98) == "legacy_unverified"
    assert module.classify_record_layer(excluded_row, min_coverage=0.98) == "strict_excluded"


def test_autofit_step_deltas_handles_no_overlap():
    module = _load_module()
    expected_conditions = [
        ("task1_outcome", "core_only", "funding_raised_usd", 7),
        ("task1_outcome", "core_only", "investors_count", 7),
    ]
    strict_records = [
        {
            "model_name": "AutoFitV1",
            "category": "autofit",
            "mae": 100.0,
            "_condition_key": ("task1_outcome", "core_only", "funding_raised_usd", 7),
        }
    ]

    deltas = module._build_autofit_step_deltas(
        strict_records=strict_records,
        best_non_by_condition={},
        expected_conditions=expected_conditions,
    )

    row = next(
        r
        for r in deltas
        if r["from_version"] == "AutoFitV1"
        and r["to_version"] == "AutoFitV2"
        and r["target"] == "investors_count"
    )
    assert row["overlap_keys"] == 0
    assert row["median_mae_delta_pct"] is None
    assert row["median_gap_delta_pct"] is None


def test_slurm_snapshot_graceful_degradation(monkeypatch):
    module = _load_module()

    monkeypatch.setattr(module, "_run_command", lambda cmd, timeout=30: "")
    snapshot = module._capture_slurm_snapshot(slurm_since="2026-02-12")

    assert "snapshot_ts" in snapshot
    assert snapshot["running_total"] == 0
    assert snapshot["pending_total"] == 0
    assert isinstance(snapshot["prefix_status_squeue"], dict)
    assert isinstance(snapshot["prefix_status_sacct"], dict)


def test_new_master_sections_registered():
    module = _load_module()
    section_names = [name for name, _ in module.SECTION_ORDER]
    assert "AUDIT_GATES" in section_names
    assert "MODEL_FAMILY_COVERAGE_AUDIT" in section_names
    assert "TARGET_SUBTASKS" in section_names
    assert "TOP3_REPRESENTATIVE_MODELS" in section_names
    assert "FAMILY_GAP_MATRIX" in section_names
    assert "CHAMPION_TEMPLATE_LIBRARY" in section_names
    assert "HYPERPARAMETER_SEARCH_LEDGER" in section_names
    assert "BEST_CONFIG_BY_MODEL_TARGET" in section_names
    assert "COMPUTE_COST_REPORT" in section_names
    assert "V72_PILOT_GATE_REPORT" in section_names
    assert "PRIMARY_LITERATURE_MATRIX" in section_names
    assert "CITATION_CORRECTION_LOG" in section_names


def test_top3_and_family_gap_helpers():
    module = _load_module()

    condition_rows = [
        {
            "condition_completed": True,
            "target": "investors_count",
            "best_model": "NBEATS",
            "best_category": "deep_classical",
        },
        {
            "condition_completed": True,
            "target": "investors_count",
            "best_model": "NBEATS",
            "best_category": "deep_classical",
        },
        {
            "condition_completed": True,
            "target": "investors_count",
            "best_model": "KAN",
            "best_category": "transformer_sota",
        },
    ]
    top3 = module._build_top3_representative_models_by_target(condition_rows)
    assert len(top3) == 2
    assert top3[0]["model_name"] == "NBEATS"
    assert top3[0]["win_count"] == 2

    strict_records = [
        {
            "target": "investors_count",
            "category": "deep_classical",
            "model_name": "NBEATS",
            "mae": 44.0,
            "_source_path": "runs/a/metrics.json",
        },
        {
            "target": "investors_count",
            "category": "transformer_sota",
            "model_name": "KAN",
            "mae": 45.0,
            "_source_path": "runs/b/metrics.json",
        },
    ]
    gap_rows = module._build_family_gap_by_target(strict_records)
    assert len(gap_rows) == 2
    deep_row = next(r for r in gap_rows if r["category"] == "deep_classical")
    tr_row = next(r for r in gap_rows if r["category"] == "transformer_sota")
    assert deep_row["gap_vs_global_best_pct"] == 0.0
    assert tr_row["gap_vs_global_best_pct"] > 0.0


def test_champion_template_library_builder():
    module = _load_module()
    condition_rows = [
        {
            "condition_completed": True,
            "task": "task1_outcome",
            "ablation": "core_edgar",
            "target": "investors_count",
            "horizon": 1,
            "best_model": "NBEATS",
            "best_category": "deep_classical",
        },
        {
            "condition_completed": True,
            "task": "task1_outcome",
            "ablation": "core_edgar",
            "target": "investors_count",
            "horizon": 7,
            "best_model": "NHITS",
            "best_category": "deep_classical",
        },
        {
            "condition_completed": True,
            "task": "task1_outcome",
            "ablation": "core_edgar",
            "target": "investors_count",
            "horizon": 14,
            "best_model": "NBEATS",
            "best_category": "deep_classical",
        },
    ]
    failure_rows = [
        {
            "issue_type": "v71_count_explosion",
            "target": "investors_count",
        }
    ]
    rows = module._build_champion_template_library(condition_rows, failure_rows)
    assert rows
    first = rows[0]
    assert first["target_family"] == "count"
    assert first["primary_anchor"] in {"NBEATS", "NHITS"}
    assert "winner_distribution_json" in first


def test_primary_literature_and_correction_rows():
    module = _load_module()
    lit_rows = module._build_primary_literature_matrix_rows()
    corr_rows = module._build_citation_correction_rows()

    assert len(lit_rows) >= 8
    assert all("primary_link" in r for r in lit_rows)
    assert any(r.get("status") == "verified_primary" for r in lit_rows)

    assert len(corr_rows) >= 3
    assert any(r.get("status") == "hypothesis" for r in corr_rows)


def test_best_config_and_pilot_gate_row_builders():
    module = _load_module()

    payload = {
        "targets": {
            "investors_count": {
                "AutoFitV72": {
                    "target_family": "count",
                    "category": "autofit",
                    "status": "planned_with_evidence",
                    "search_budget": 96,
                    "trials_executed": 0,
                    "best_mae_observed_strict": 44.8,
                    "best_config": {"top_k": 8},
                    "search_space": {"top_k": [6, 8, 10]},
                    "evidence_path": "runs/benchmarks/x/metrics.json",
                }
            }
        }
    }
    best_rows = module._build_best_config_rows(payload, "docs/benchmarks/block3_truth_pack/best_config_by_model_target.json")
    assert len(best_rows) == 1
    assert best_rows[0]["model_name"] == "AutoFitV72"
    assert best_rows[0]["target"] == "investors_count"

    pilot_payload = {
        "generated_at_utc": "2026-02-18T00:00:00+00:00",
        "overall_pass": False,
        "counts": {"rows_total": 10},
        "metrics": {"global_normalized_mae_improvement_pct": 5.0},
        "checks": {"fairness_pass_100": True},
    }
    gate_rows = module._build_pilot_gate_rows(
        pilot_payload,
        "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
    )
    assert len(gate_rows) >= 4
    assert any(r["key"] == "overall_pass" for r in gate_rows)
