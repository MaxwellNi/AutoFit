#!/usr/bin/env python3
"""Track Round-14 execution gates from landed audit artifacts.

This is a meta-audit: it does not decide science by narrative. It reads the
latest dashboard/text/trunk audits and records which promises are already
executed, which are only partially executed, and which are still missing an
executable artifact.
"""

from __future__ import annotations

import glob
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_execution_gate_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def _latest(pattern: str) -> Path | None:
    paths = sorted(ROOT.glob(pattern))
    return paths[-1] if paths else None


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        return {"_load_error": f"{type(exc).__name__}: {exc}", "_path": str(path)}


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _status(pass_: bool, partial: bool = False) -> str:
    if pass_:
        return "passed"
    if partial:
        return "partial"
    return "not_passed"


def _script_exists(path: str) -> bool:
    return (ROOT / path).exists()


def _coverage_gate(dashboard: dict[str, Any] | None) -> dict[str, Any]:
    checks = (dashboard or {}).get("checks", {})
    coverage = checks.get("2_v2_coverage", {})
    tpub = checks.get("4_tpub_external", {})
    read_errors = checks.get("9_metric_read_errors", {})
    c90_mean = _finite(coverage.get("c90_mean"))
    mondrian_delta = _finite(coverage.get("mondrian_delta_mean"))
    studentized_delta = _finite(coverage.get("studentized_delta_mean"))
    studentized_n = int(coverage.get("studentized_n_records") or 0)
    read_error_count = int(read_errors.get("n_errors") or 0)
    external_done = any(bool(v.get("exists")) for v in tpub.values() if isinstance(v, dict))
    return {
        "marginal_coverage": {
            "status": _status(bool(c90_mean is not None and c90_mean >= 0.88)),
            "c90_mean": c90_mean,
            "target": "near 0.90; tracker pass threshold currently >=0.88",
        },
        "mondrian_variant": {
            "status": _status(bool(mondrian_delta is not None and mondrian_delta > 0.0), partial=coverage.get("mondrian_n_records") is not None),
            "n_records": coverage.get("mondrian_n_records"),
            "delta_mean": mondrian_delta,
        },
        "studentized_variant": {
            "status": _status(bool(studentized_delta is not None and studentized_delta > 0.0 and read_error_count == 0), partial=studentized_n > 0),
            "n_records_visible": studentized_n,
            "delta_mean_visible": studentized_delta,
            "metric_read_errors": read_error_count,
            "note": "If read errors are nonzero, studentized counts are readable-window lower bounds.",
        },
        "external_public_validation": {
            "status": _status(external_done),
            "datasets": tpub,
        },
    }


def _text_edgar_gate(text_audit: dict[str, Any] | None, trunk_audit: dict[str, Any] | None) -> dict[str, Any]:
    result = (text_audit or {}).get("result_delta_audit", {})
    by_ablation = result.get("summary", {}).get("by_ablation", {})
    source = result.get("source_scale_activation", {})
    artifact = (text_audit or {}).get("embedding_artifact_audit", {})
    join = (text_audit or {}).get("join_log_audit", {})
    guard = (trunk_audit or {}).get("guard_summary", {})

    def ablation_status(name: str) -> dict[str, Any]:
        item = by_ablation.get(name, {})
        wins = int(item.get("mae_wins_lower_is_better") or 0)
        losses = int(item.get("mae_losses") or 0)
        pairs = int(item.get("n_pairs") or 0)
        return {
            "status": _status(wins > losses and pairs > 0, partial=pairs > 0),
            "n_pairs": pairs,
            "mae_wins": wins,
            "mae_losses": losses,
            "mae_ties": item.get("mae_ties"),
            "c90_closeness_wins": item.get("c90_closeness_wins_lower_abs_gap_is_better"),
            "c90_closeness_losses": item.get("c90_closeness_losses"),
        }

    parquet = artifact.get("parquet", {})
    latest_edgar = join.get("latest_edgar_join") or {}
    source_positive = int(source.get("positive_rows") or 0)
    source_observed = int(source.get("observed_rows") or 0)
    return {
        "artifact_join_integrity": {
            "status": _status(bool(parquet.get("exists") and parquet.get("n_text_emb_columns") == 64 and latest_edgar)),
            "text_rows": parquet.get("num_rows"),
            "text_embedding_columns": parquet.get("n_text_emb_columns"),
            "latest_edgar_join": latest_edgar,
        },
        "core_text_effect": ablation_status("core_text"),
        "core_edgar_effect": ablation_status("core_edgar"),
        "full_effect": ablation_status("full"),
        "source_scaling_activation": {
            "status": _status(source_positive > 0, partial=source_observed > 0),
            "positive_rows": source_positive,
            "observed_rows": source_observed,
            "trunk_source_scale_positive_rows": guard.get("source_scale_positive_rows"),
            "note": "No source-scaling novelty claim is allowed while positive_rows is zero.",
        },
        "missing_text_edgar_execution_items": [
            item for item, exists in {
                "embedding_model_bakeoff": _script_exists("scripts/r14_embedding_bakeoff.py"),
                "event_semantics_probe": _script_exists("scripts/r14_event_semantics_probe.py"),
                "text_counterfactual_ablation": _script_exists("scripts/r14_text_counterfactual_audit.py"),
                "source_path_activation_fix": _script_exists("scripts/r14_source_path_activation_audit.py"),
            }.items() if not exists
        ],
    }


def _counterfactual_gate() -> dict[str, Any]:
    paths = sorted(glob.glob(str(ROOT / "runs/benchmarks/r14fcast_main_h*_co_*/predictions.parquet")))
    schema_sample: list[str] = []
    stable_key_found = False
    if paths:
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(paths[-1])
            schema_sample = list(pf.schema.names)
            key_names = {"entity_id", "crawled_date_day", "date", "row_id", "snapshot_id"}
            stable_key_found = bool(key_names.intersection(schema_sample))
        except Exception as exc:
            schema_sample = [f"schema_error:{type(exc).__name__}:{exc}"]
    return {
        "strict_identical_row_counterfactual": {
            "status": _status(False, partial=bool(paths)),
            "prediction_artifacts_found": len(paths),
            "latest_schema": schema_sample,
            "stable_row_key_found": stable_key_found,
            "note": "Existing predictions lack stable row keys; rerun a dedicated counterfactual audit with entity/date/original-row keys.",
        }
    }


def _method_next_scripts_gate() -> dict[str, Any]:
    required = {
        "sigma_residual_diagnostic": "scripts/r14_sigma_residual_diagnostic.py",
        "subgroup_conformal_pilot": "scripts/r14_subgroup_conformal_pilot.py",
        "cqr_pilot": "scripts/r14_cqr_pilot.py",
        "gpd_evt_tail_pilot": "scripts/r14_gpd_evt_tail_pilot.py",
        "fremtpl2_external_validation": "scripts/r14_fremtpl2_external_validation.py",
    }
    return {
        name: {"status": _status(_script_exists(path)), "path": path}
        for name, path in required.items()
    }


def main() -> int:
    dashboard_path = _latest("runs/audits/r14_audit_dashboard_*.json")
    text_path = _latest("runs/audits/r14_text_edgar_signal_audit_*.json")
    trunk_path = _latest("runs/audits/r14_mainline_trunk_signal_audit_*.json")
    dashboard = _load(dashboard_path)
    text_audit = _load(text_path)
    trunk_audit = _load(trunk_path)

    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "source_artifacts": {
            "dashboard": str(dashboard_path) if dashboard_path else None,
            "text_edgar_audit": str(text_path) if text_path else None,
            "trunk_guard_audit": str(trunk_path) if trunk_path else None,
        },
        "coverage_gates": _coverage_gate(dashboard),
        "text_edgar_gates": _text_edgar_gate(text_audit, trunk_audit),
        "counterfactual_gates": _counterfactual_gate(),
        "next_method_execution_gates": _method_next_scripts_gate(),
        "claim_policy": [
            "Do not claim text embeddings are effective until paired core_text deltas pass and event/counterfactual probes exist.",
            "Do not claim source-scaling novelty while source_scale_positive_rows is zero.",
            "Do not claim coverage solved until c90 is near 0.90 without metric read errors.",
            "Do not claim external validation until at least one public dataset artifact exists.",
        ],
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    lines = [f"# R14 Execution Gate Tracker — {report['timestamp_cest']}", "", "```json"]
    lines.append(json.dumps(report, indent=2, default=str))
    lines.append("```")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"OK: {OUT_JSON}")
    print(f"OK: {OUT_MD}")
    print(json.dumps({
        "coverage_gates": report["coverage_gates"],
        "text_edgar_gates": report["text_edgar_gates"],
        "next_method_execution_gates": report["next_method_execution_gates"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())