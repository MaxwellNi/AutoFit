#!/usr/bin/env python3
"""Audit whether current R14 artifacts support cross-domain generalization claims."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs" / "research" / "generalization_validation_matrix.json"


def _latest(pattern: str) -> Path | None:
    files = sorted((ROOT / "runs" / "audits").glob(pattern))
    return files[-1] if files else None


def _read(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit records read failures
        return {"read_error": type(exc).__name__, "error": str(exc), "path": str(path)}


def main() -> int:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    expected = sorted(row["family"] for row in matrix.get("public_pack_families", []))
    public_summary_path = _latest("r14_public_pack_full_summary_*.json")
    mechanism_path = _latest("r14_literature_mechanism_audit_*.json")
    source_regime_path = _latest("r14_source_regime_conformal_audit_*.json")
    rowkey_regime_path = _latest("r14_source_regime_rowkey_conformal_diagnostic_*.json")
    weakness_path = _latest("r14_coverage_weakness_audit_*.json")
    source_path_path = _latest("r14_source_path_activation_audit_*.json")
    telemetry_path = _latest("r14_source_event_state_telemetry_audit_*.json")
    source_features_path = _latest("r14_source_event_state_features_*.json")
    read_gate_path = _latest("r14_source_read_gate_counterfactual_audit_*.json")
    runtime_read_policy_path = _latest("r14_source_runtime_read_policy_smoke_*.json")
    embedding_bakeoff_path = _latest("r14_embedding_bakeoff_*.json")
    embedding_downstream_path = _latest("r14_embedding_downstream_pair_audit_*.json")

    public_summary = _read(public_summary_path)
    mechanism = _read(mechanism_path)
    source_regime = _read(source_regime_path)
    rowkey_regime = _read(rowkey_regime_path)
    weakness = _read(weakness_path)
    source_path = _read(source_path_path)
    telemetry = _read(telemetry_path)
    source_features = _read(source_features_path)
    read_gate = _read(read_gate_path)
    runtime_read_policy = _read(runtime_read_policy_path)
    embedding_bakeoff = _read(embedding_bakeoff_path)
    embedding_downstream = _read(embedding_downstream_path)

    completed = sorted(public_summary.get("completed_families", [])) if isinstance(public_summary, dict) else []
    missing = sorted(set(expected) - set(completed))
    public_pack_passed = bool(public_summary) and public_summary.get("status") == "passed" and not missing and public_summary.get("total_errors") == 0

    source_pair_summary = (source_regime or {}).get("paired_summary_vs_core_only", {}) if isinstance(source_regime, dict) else {}
    cqr_wins = int(source_pair_summary.get("delta_nccopo_coverage_90_cqr_lite_source_minus_core_wins_higher_coverage") or 0)
    cqr_losses = int(source_pair_summary.get("delta_nccopo_coverage_90_cqr_lite_source_minus_core_losses_lower_coverage") or 0)
    mae_wins = int(source_pair_summary.get("delta_mae_source_minus_core_wins_lower_mae") or 0)
    mae_losses = int(source_pair_summary.get("delta_mae_source_minus_core_losses_higher_mae") or 0)

    coverage_summary = ((weakness or {}).get("summaries", {}) if isinstance(weakness, dict) else {})
    cqr_summary = coverage_summary.get("cqr_lite", {}) if isinstance(coverage_summary, dict) else {}
    drift_summary = coverage_summary.get("drift_guard", {}) if isinstance(coverage_summary, dict) else {}

    gates = {
        "literature_mechanism_registry_valid": bool(mechanism) and mechanism.get("status") == "passed",
        "source_regime_has_calibration_signal": cqr_wins > cqr_losses and cqr_wins > 0,
        "source_regime_has_point_accuracy_signal": mae_wins > mae_losses and mae_wins > 0,
        "rowkey_regime_diagnostic_available": bool(rowkey_regime) and rowkey_regime.get("status") == "diagnostic_completed",
        "source_event_state_telemetry_passed": bool(telemetry) and telemetry.get("status") == "passed",
        "source_event_state_feature_table_passed": bool(source_features) and source_features.get("status") == "passed",
        "source_runtime_read_policy_wired": bool(runtime_read_policy) and runtime_read_policy.get("status") == "passed",
        "source_read_gate_counterfactual_passed": bool(read_gate) and read_gate.get("status") == "passed",
        "source_path_activation_passed": bool(source_path) and source_path.get("status") == "passed",
        "embedding_bakeoff_passed": bool(embedding_bakeoff) and embedding_bakeoff.get("status") == "passed",
        "embedding_downstream_pair_passed": bool(embedding_downstream) and embedding_downstream.get("status") == "passed",
        "public_pack_complete_zero_error": public_pack_passed,
        "coverage_weak_cells_resolved": bool(cqr_summary) and int(cqr_summary.get("below_0_88") or 0) == 0,
        "drift_guard_formal_passed": bool(drift_summary) and int(drift_summary.get("below_0_88") or 0) == 0 and int(drift_summary.get("n_records") or 0) > 0,
    }
    oral_ready = all(gates.values())
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": "oral_ready" if oral_ready else "not_oral_ready",
        "matrix": str(MATRIX.relative_to(ROOT)),
        "expected_public_pack_families": expected,
        "completed_public_pack_families": completed,
        "missing_public_pack_families": missing,
        "artifact_paths": {
            "public_summary": str(public_summary_path.relative_to(ROOT)) if public_summary_path else None,
            "mechanism_audit": str(mechanism_path.relative_to(ROOT)) if mechanism_path else None,
            "source_regime_audit": str(source_regime_path.relative_to(ROOT)) if source_regime_path else None,
            "rowkey_regime_diagnostic": str(rowkey_regime_path.relative_to(ROOT)) if rowkey_regime_path else None,
            "coverage_weakness_audit": str(weakness_path.relative_to(ROOT)) if weakness_path else None,
            "source_path_activation_audit": str(source_path_path.relative_to(ROOT)) if source_path_path else None,
            "source_event_state_telemetry_audit": str(telemetry_path.relative_to(ROOT)) if telemetry_path else None,
            "source_event_state_features_audit": str(source_features_path.relative_to(ROOT)) if source_features_path else None,
            "source_runtime_read_policy_smoke": str(runtime_read_policy_path.relative_to(ROOT)) if runtime_read_policy_path else None,
            "source_read_gate_counterfactual_audit": str(read_gate_path.relative_to(ROOT)) if read_gate_path else None,
            "embedding_bakeoff_audit": str(embedding_bakeoff_path.relative_to(ROOT)) if embedding_bakeoff_path else None,
            "embedding_downstream_pair_audit": str(embedding_downstream_path.relative_to(ROOT)) if embedding_downstream_path else None,
        },
        "gates": gates,
        "source_regime_summary": {
            "cqr_lite_wins": cqr_wins,
            "cqr_lite_losses": cqr_losses,
            "mae_wins": mae_wins,
            "mae_losses": mae_losses,
            "interpretation": "calibration signal only" if cqr_wins > cqr_losses and mae_wins <= mae_losses else "needs review",
        },
        "coverage_summary": {
            "cqr_lite_below_0_88": cqr_summary.get("below_0_88"),
            "cqr_lite_below_0_90": cqr_summary.get("below_0_90"),
            "drift_guard_below_0_88": drift_summary.get("below_0_88"),
            "drift_guard_below_0_90": drift_summary.get("below_0_90"),
        },
        "source_event_state_summary": {
            "telemetry_status": None if not isinstance(telemetry, dict) else telemetry.get("status"),
            "text_match_rate": None if not isinstance(telemetry, dict) else telemetry.get("text_telemetry", {}).get("match_rate_on_prediction_keys"),
            "edgar_match_rate": None if not isinstance(telemetry, dict) else telemetry.get("edgar_telemetry", {}).get("match_rate_on_prediction_cik_day_keys"),
            "source_path_status": None if not isinstance(source_path, dict) else source_path.get("status"),
            "source_path_active_rows": None if not isinstance(source_path, dict) else source_path.get("source_scale_active_rows"),
            "source_path_paired_benefit_passed": None if not isinstance(source_path, dict) else source_path.get("paired_benefit_passed"),
            "source_feature_table_status": None if not isinstance(source_features, dict) else source_features.get("status"),
            "source_feature_table_summary": None if not isinstance(source_features, dict) else source_features.get("summary"),
            "source_runtime_read_policy_status": None if not isinstance(runtime_read_policy, dict) else runtime_read_policy.get("status"),
            "source_runtime_read_policy_checks": None if not isinstance(runtime_read_policy, dict) else runtime_read_policy.get("checks"),
            "source_read_gate_counterfactual_status": None if not isinstance(read_gate, dict) else read_gate.get("status"),
            "source_read_gate_counterfactual_overall": None if not isinstance(read_gate, dict) else read_gate.get("overall"),
            "source_read_gate_counterfactual_by_bucket": None if not isinstance(read_gate, dict) else read_gate.get("by_read_bucket"),
            "embedding_bakeoff_status": None if not isinstance(embedding_bakeoff, dict) else embedding_bakeoff.get("status"),
            "embedding_downstream_status": None if not isinstance(embedding_downstream, dict) else embedding_downstream.get("status"),
            "embedding_downstream_summary": None if not isinstance(embedding_downstream, dict) else embedding_downstream.get("summary"),
        },
        "claim_boundary": matrix.get("claim_boundary", []),
        "next_required_actions": [
            "Do not promote source-read point-forecast claims until read-confidence buckets reduce mean source-vs-core absolute error.",
            "Run formal temporal reruns with the runtime read/no-read policy and compare against dense source features under strict ablations.",
            "Add a tail-aware source-read guard for funding so row-count wins cannot hide large-error regressions.",
            "Resolve coverage weak cells below 0.88 before oral-level reliability claims.",
            "Add normalized public-pack comparisons before cross-industry SOTA claims.",
        ],
    }
    out_json = ROOT / "runs" / "audits" / f"r14_generalization_gate_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out_md.write_text("# R14 Generalization Gate Audit\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n```\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "gates": gates,
        "missing_public_pack_families": missing,
        "source_regime_summary": report["source_regime_summary"],
        "coverage_summary": report["coverage_summary"],
        "out_json": str(out_json),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())