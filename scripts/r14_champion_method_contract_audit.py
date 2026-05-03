#!/usr/bin/env python3
"""Audit whether the current R14 line satisfies the champion method contract."""

from __future__ import annotations

import glob
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "research" / "champion_method_contract.json"
AUDIT_DIR = ROOT / "runs" / "audits"


REQUIRED_CONTRACT_KEYS = {
    "registry_name",
    "registry_version",
    "purpose",
    "claim_boundary",
    "main_method_spine",
    "thresholds",
    "gate_groups",
    "candidate_strategy_rank",
}


def _latest(pattern: str) -> Path | None:
    files = sorted((ROOT / "runs" / "audits").glob(pattern))
    return max(files, key=os.path.getmtime) if files else None


def _read(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit records read failures
        return {"read_error": type(exc).__name__, "error": str(exc), "path": str(path)}


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _validate_contract(contract: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    missing = sorted(REQUIRED_CONTRACT_KEYS - set(contract))
    if missing:
        errors.append({"missing_top_level_keys": missing})
    if not isinstance(contract.get("gate_groups"), list) or not contract.get("gate_groups"):
        errors.append({"invalid": "gate_groups must be a non-empty list"})
    for idx, group in enumerate(contract.get("gate_groups", [])):
        for key in ("id", "why_it_matters", "required_evidence"):
            if key not in group:
                errors.append({"gate_index": idx, "missing": key})
        if not isinstance(group.get("required_evidence"), list) or not group.get("required_evidence"):
            errors.append({"gate_index": idx, "invalid": "required_evidence"})
    if not isinstance(contract.get("thresholds"), dict):
        errors.append({"invalid": "thresholds must be a dict"})
    return errors


def _hard_cell_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(ROOT / "runs" / "benchmarks" / "r14fcast_*" / "metrics.json"))):
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            rows = payload["results"]
        elif isinstance(payload, list):
            rows = payload
        else:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            coverage = _finite(row.get("nccopo_coverage_90_hard_cell_tail_guard"))
            if coverage is None:
                continue
            marginal_width = _finite(row.get("nccopo_interval_width_mean"))
            hard_width = _finite(row.get("nccopo_interval_width_mean_hard_cell_tail_guard"))
            width_ratio = None
            if marginal_width and hard_width is not None and marginal_width > 0:
                width_ratio = hard_width / marginal_width
            records.append(
                {
                    "run": Path(path).parent.name,
                    "model": row.get("model_name") or row.get("model"),
                    "target": row.get("target"),
                    "horizon": row.get("horizon"),
                    "ablation": row.get("ablation"),
                    "coverage": coverage,
                    "marginal_width": marginal_width,
                    "hard_cell_width": hard_width,
                    "width_ratio_vs_marginal": width_ratio,
                    "status": row.get("nccopo_hard_cell_status"),
                    "metrics_path": str(Path(path).relative_to(ROOT)),
                }
            )
    return records


def _hard_cell_summary(records: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    floor = float(thresholds.get("coverage_floor", 0.88))
    max_ratio = float(thresholds.get("hard_cell_max_width_ratio_vs_marginal", 3.0))
    matrix_min = int(thresholds.get("hard_cell_min_records_for_matrix", 8))
    coverages = [float(row["coverage"]) for row in records if row.get("coverage") is not None]
    ratios = [float(row["width_ratio_vs_marginal"]) for row in records if row.get("width_ratio_vs_marginal") is not None]
    return {
        "n_records": len(records),
        "coverage_min": min(coverages) if coverages else None,
        "coverage_mean": sum(coverages) / len(coverages) if coverages else None,
        "below_floor": sum(1 for value in coverages if value < floor),
        "width_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "width_ratio_max": max(ratios) if ratios else None,
        "single_formal_record_passed": bool(coverages) and min(coverages) >= floor,
        "matrix_sufficient": len(records) >= matrix_min and bool(coverages) and min(coverages) >= floor,
        "width_sharpness_guard_passed": bool(ratios) and max(ratios) <= max_ratio,
        "weakest_records": sorted(records, key=lambda row: row.get("coverage") or -1.0)[:10],
    }


def _evaluate_gates(contract: dict[str, Any], artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    thresholds = contract.get("thresholds", {}) if isinstance(contract.get("thresholds"), dict) else {}
    contract_errors = _validate_contract(contract)
    gate = artifacts.get("generalization_gate", {})
    gate_flags = gate.get("gates", {}) if isinstance(gate.get("gates"), dict) else {}
    read_gate = artifacts.get("read_gate", {})
    source_path = artifacts.get("source_path", {})
    mainline = artifacts.get("mainline_value", {})
    coverage = artifacts.get("coverage_weakness", {})
    literature = artifacts.get("literature", {})
    public_pack = artifacts.get("public_pack", {})
    hard_cell = artifacts.get("hard_cell_summary", {})

    cqr_summary = ((coverage.get("summaries") or {}).get("cqr_lite") or {}) if isinstance(coverage, dict) else {}
    mainline_win_rate = _finite(mainline.get("mainline_win_rate"))
    mainline_pairs = int(mainline.get("n_paired_cells") or 0)
    mainline_wins = int(mainline.get("mainline_wins") or 0)
    mainline_losses = int(mainline.get("mainline_losses") or 0)

    not_implemented = literature.get("not_implemented_ids") or []
    if not isinstance(not_implemented, list):
        not_implemented = []

    evidence = {
        "champion_contract_valid": not contract_errors,
        "source_event_state_telemetry_passed": bool(gate_flags.get("source_event_state_telemetry_passed")),
        "source_event_state_feature_table_passed": bool(gate_flags.get("source_event_state_feature_table_passed")),
        "source_runtime_read_policy_wired": bool(gate_flags.get("source_runtime_read_policy_wired")),
        "source_read_gate_counterfactual_passed": bool(read_gate) and read_gate.get("status") == "passed",
        "source_path_activation_passed": bool(source_path) and source_path.get("status") == "passed",
        "source_regime_has_point_accuracy_signal": bool(gate_flags.get("source_regime_has_point_accuracy_signal")),
        "mainline_point_champion_vs_best_non_mainline": (
            mainline_pairs >= int(thresholds.get("mainline_min_paired_cells", 30))
            and mainline_win_rate is not None
            and mainline_win_rate >= float(thresholds.get("mainline_min_win_rate_vs_best_non_mainline", 0.5))
            and mainline_wins > mainline_losses
        ),
        "hard_cell_tail_guard_formal_passed": bool(hard_cell.get("single_formal_record_passed")),
        "hard_cell_tail_guard_matrix_sufficient": bool(hard_cell.get("matrix_sufficient")),
        "hard_cell_width_sharpness_guard_passed": bool(hard_cell.get("width_sharpness_guard_passed")),
        "cqr_lite_global_weak_cells_resolved": int(cqr_summary.get("below_0_88") or 0) == 0 and bool(cqr_summary),
        "public_pack_complete_zero_error": public_pack.get("status") == "passed" and int(public_pack.get("total_errors") or 0) == 0,
        "embedding_bakeoff_passed": bool(gate_flags.get("embedding_bakeoff_passed")),
        "embedding_downstream_pair_passed": bool(gate_flags.get("embedding_downstream_pair_passed")),
        "literature_mechanism_registry_valid": literature.get("status") == "passed",
        "no_unimplemented_frontier_mechanism_blockers": len(not_implemented) == 0,
    }

    group_results = []
    for group in contract.get("gate_groups", []):
        required = group.get("required_evidence", [])
        missing = [item for item in required if not evidence.get(item)]
        group_results.append(
            {
                "id": group.get("id"),
                "passed": not missing,
                "missing_evidence": missing,
                "why_it_matters": group.get("why_it_matters"),
            }
        )

    oral_grade = bool(evidence.get("champion_contract_valid")) and all(row["passed"] for row in group_results)
    source_read_overall = read_gate.get("overall", {}) if isinstance(read_gate.get("overall"), dict) else {}
    return {
        "status": "oral_grade_method_ready" if oral_grade else "not_oral_grade_method_ready",
        "contract_errors": contract_errors,
        "evidence": evidence,
        "gate_groups": group_results,
        "diagnostics": {
            "read_gate_status": read_gate.get("status"),
            "read_gate_mean_delta_abs_error_source_minus_core": source_read_overall.get("mean_delta_abs_error_source_minus_core"),
            "read_gate_win_rate": source_read_overall.get("win_rate"),
            "source_path_status": source_path.get("status"),
            "source_pair_active_mae_wins": source_path.get("source_pair_active_mae_wins"),
            "source_pair_active_mae_losses": source_path.get("source_pair_active_mae_losses"),
            "source_pair_active_mae_delta_mean": source_path.get("source_pair_active_mae_delta_mean"),
            "mainline_n_paired_cells": mainline_pairs,
            "mainline_win_rate": mainline_win_rate,
            "mainline_wins": mainline_wins,
            "mainline_losses": mainline_losses,
            "cqr_lite_below_0_88": cqr_summary.get("below_0_88"),
            "cqr_lite_below_0_90": cqr_summary.get("below_0_90"),
            "hard_cell_summary": hard_cell,
            "not_implemented_frontier_mechanisms": not_implemented,
        },
    }


def _next_actions(evaluation: dict[str, Any]) -> list[str]:
    evidence = evaluation.get("evidence", {})
    actions = []
    if not evidence.get("source_read_gate_counterfactual_passed"):
        actions.append("Redesign source reading around confidence, recency, novelty, and tail-risk buckets; rerun strict row-key read-gate until mean source-minus-core error is negative in promotable read buckets.")
    if not evidence.get("source_path_activation_passed"):
        actions.append("Keep source-scaling demoted; if revisited, require nonzero active rows plus paired MAE benefit before promotion.")
    if not evidence.get("mainline_point_champion_vs_best_non_mainline"):
        actions.append("Run a broader paired point-value audit across all three heads and compare the auditable protocol wrapper over top forecasters instead of assuming mainline is the point champion.")
    if not evidence.get("hard_cell_tail_guard_matrix_sufficient"):
        actions.append("Expand hard-cell tail-guard formal reruns across funding h14/h30 and core_only/core_text/core_edgar/full before reliability claims.")
    if not evidence.get("hard_cell_width_sharpness_guard_passed"):
        actions.append("Add a width/sharpness gate and tune tail calibration so coverage gains do not come from excessive interval inflation.")
    if not evidence.get("cqr_lite_global_weak_cells_resolved"):
        actions.append("Treat CQR-lite as a baseline protocol, not a solved reliability layer; prioritize hard-region calibration and distributional objectives.")
    if not evidence.get("no_unimplemented_frontier_mechanism_blockers"):
        actions.append("Convert the unimplemented frontier mechanisms into bounded experiments: distributional tail objective, selective hard-region learning, and offline specialist distillation.")
    return actions


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    artifact_paths = {
        "generalization_gate": _latest("r14_generalization_gate_audit_*.json"),
        "read_gate": _latest("r14_source_read_gate_counterfactual_audit_*.json"),
        "source_path": _latest("r14_source_path_activation_audit_*.json"),
        "mainline_value": _latest("r14_mainline_value_audit_*.json"),
        "coverage_weakness": _latest("r14_coverage_weakness_audit_*.json"),
        "literature": _latest("r14_literature_mechanism_audit_*.json"),
        "public_pack": _latest("r14_public_pack_full_summary_*.json"),
    }
    hard_cell_records = _hard_cell_records()
    artifacts = {name: _read(path) for name, path in artifact_paths.items()}
    artifacts["hard_cell_summary"] = _hard_cell_summary(hard_cell_records, contract.get("thresholds", {}))
    evaluation = _evaluate_gates(contract, artifacts)
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": evaluation["status"],
        "contract": str(CONTRACT.relative_to(ROOT)),
        "artifact_paths": {
            name: None if path is None else str(path.relative_to(ROOT))
            for name, path in artifact_paths.items()
        },
        "claim_boundary": contract.get("claim_boundary", []),
        "main_method_spine": contract.get("main_method_spine", {}),
        "candidate_strategy_rank": contract.get("candidate_strategy_rank", []),
        **evaluation,
        "next_actions": _next_actions(evaluation),
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = AUDIT_DIR / f"r14_champion_method_contract_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = out_json.with_suffix(".md")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out_md.write_text(
        "# R14 Champion Method Contract Audit\n\n```json\n"
        + json.dumps(report, indent=2, ensure_ascii=False, default=str)
        + "\n```\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "gate_groups": report["gate_groups"],
                "diagnostics": report["diagnostics"],
                "next_actions": report["next_actions"],
                "out_json": str(out_json),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())