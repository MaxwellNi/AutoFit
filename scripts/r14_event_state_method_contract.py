#!/usr/bin/env python3
"""Build an auditable contract for the Event-State NC-CoPo method claim.

The report intentionally separates what can be claimed now from what remains
blocked. It is a paper-claim guard, not a benchmark runner.
"""

from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "audits"


def _latest(pattern: str) -> Path | None:
    paths = sorted(Path(p) for p in glob.glob(str(ROOT / pattern)))
    return paths[-1] if paths else None


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def _status_from_tracker(tracker: dict[str, Any], *keys: str) -> Any:
    cur: Any = tracker
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _counterfactual_summary(counter: dict[str, Any]) -> dict[str, Any]:
    strict = counter.get("strict_rowkey_counterfactual", {}) if counter else {}
    comparisons = strict.get("comparisons", []) if isinstance(strict, dict) else []
    return {
        "artifact_status": counter.get("status"),
        "strict_status": strict.get("status"),
        "max_rows_per_file": strict.get("max_rows_per_file"),
        "selected_files": (strict.get("rowkey_prediction_file_selection") or {}).get("selected_files"),
        "eligible_files": (strict.get("rowkey_prediction_file_selection") or {}).get("eligible_files"),
        "comparisons": [
            {
                "ablation": item.get("ablation"),
                "status": item.get("status"),
                "n_overlap": item.get("n_overlap"),
                "unique_strict_keys": item.get("n_overlap_unique_strict_keys"),
                "y_true_equal_rate": item.get("y_true_equal_rate"),
                "horizons": item.get("horizons"),
                "model_count": item.get("model_count"),
                "prediction_delta_mean": (item.get("prediction_delta") or {}).get("mean"),
                "abs_error_delta_mean": (item.get("abs_error_delta_alt_minus_core_only") or {}).get("mean"),
            }
            for item in comparisons
        ],
    }


def _dashboard_summary(dashboard: dict[str, Any]) -> dict[str, Any]:
    checks = dashboard.get("checks", {}) if dashboard else {}
    coverage = checks.get("2_v2_coverage", {})
    read_errors = checks.get("9_metric_read_errors", {})
    matrix = checks.get("8_coverage_matrix", {})
    return {
        "marginal_c90_mean": coverage.get("c90_mean"),
        "marginal_c90_min": coverage.get("c90_min"),
        "marginal_c90_max": coverage.get("c90_max"),
        "cqr_lite_n_records": coverage.get("cqr_lite_n_records"),
        "cqr_lite_c90_mean": coverage.get("cqr_lite_c90_mean"),
        "cqr_lite_delta_mean": coverage.get("cqr_lite_delta_mean"),
        "canonical_cqr_lite_n_records": coverage.get("canonical_cqr_lite_n_records"),
        "canonical_cqr_lite_c90_mean": coverage.get("canonical_cqr_lite_c90_mean"),
        "canonical_cqr_lite_delta_mean": coverage.get("canonical_cqr_lite_delta_mean"),
        "gpd_evt_n_records": coverage.get("gpd_evt_n_records"),
        "gpd_evt_c90_mean": coverage.get("gpd_evt_c90_mean"),
        "gpd_evt_delta_mean": coverage.get("gpd_evt_delta_mean"),
        "studentized_delta_mean": coverage.get("studentized_delta_mean"),
        "mondrian_delta_mean": coverage.get("mondrian_delta_mean"),
        "metric_read_errors": read_errors.get("n_errors"),
        "coverage_cells": matrix.get("n_cells_with_c90"),
        "cqr_lite_cells": matrix.get("n_cells_with_cqr_lite_c90"),
    }


def _embedding_summary(rep: dict[str, Any], tracker: dict[str, Any]) -> dict[str, Any]:
    text_artifacts = _status_from_tracker(tracker, "text_edgar_gates", "text_edgar_execution_artifacts") or {}
    bakeoff = text_artifacts.get("embedding_model_bakeoff", {}) if isinstance(text_artifacts, dict) else {}
    probes = rep.get("linear_event_probes", {}) if rep else {}
    return {
        "representation_artifact_status": rep.get("status"),
        "representation_sample_rows": rep.get("sample_rows"),
        "representation_sample_mode": rep.get("sample_mode"),
        "row_alignment_rate": rep.get("row_alignment_rate_first_sample"),
        "finite_value_rate": (rep.get("numeric_quality") or {}).get("finite_value_rate"),
        "event_probe_aucs": {key: val.get("auc") for key, val in probes.items()},
        "bakeoff_status": bakeoff.get("status"),
        "bakeoff_artifact_status": bakeoff.get("artifact_status"),
        "bakeoff_latest_artifact": bakeoff.get("latest_artifact"),
    }


def main() -> int:
    counter_path = _latest("runs/audits/r14_text_counterfactual_audit_*.json")
    dashboard_path = _latest("runs/audits/r14_audit_dashboard_*.json")
    tracker_path = _latest("runs/audits/r14_execution_gate_tracker_*.json")
    rep_path = _latest("runs/audits/r14_embedding_representation_audit_*.json")

    counter = _load(counter_path)
    dashboard = _load(dashboard_path)
    tracker = _load(tracker_path)
    rep = _load(rep_path)

    coverage = _dashboard_summary(dashboard)
    counter_summary = _counterfactual_summary(counter)
    embedding = _embedding_summary(rep, tracker)

    source_gate = _status_from_tracker(tracker, "text_edgar_gates", "source_scaling_activation") or {}
    external_gate = _status_from_tracker(tracker, "coverage_gates", "external_public_validation") or {}
    cqr_gate = _status_from_tracker(tracker, "coverage_gates", "cqr_lite_temporal_benchmark") or {}
    gpd_gate = _status_from_tracker(tracker, "coverage_gates", "gpd_evt_temporal_benchmark") or {}

    strict_passed = counter_summary.get("artifact_status") == "passed" and counter_summary.get("strict_status") == "passed"
    cqr_positive = (coverage.get("canonical_cqr_lite_delta_mean") or coverage.get("cqr_lite_delta_mean") or 0) > 0
    cqr_near_threshold = (coverage.get("canonical_cqr_lite_c90_mean") or coverage.get("cqr_lite_c90_mean") or 0) >= 0.88
    no_read_errors = coverage.get("metric_read_errors") == 0
    embedding_ready = embedding.get("bakeoff_artifact_status") == "passed"
    source_ready = (source_gate.get("positive_rows") or 0) > 0
    external_ready = external_gate.get("status") == "passed"
    oral_blockers = [
        "coverage not solved across broad temporal surface" if not cqr_near_threshold else None,
        "external public datasets not passed" if not external_ready else None,
        "embedding bakeoff not passed" if not embedding_ready else None,
        "source scaling mechanism not activated" if not source_ready else None,
    ]

    claim_levels = {
        "core_method_claim": {
            "status": "passed" if (cqr_near_threshold and cqr_positive and no_read_errors and external_ready) else "partial",
            "allowed": (
                "Event-state NC-CoPo with CQR-lite passes the canonical CQR protocol gate on the landed temporal surface; "
                "raw diagnostic-run coverage remains separately reported."
            ) if cqr_near_threshold and external_ready else (
                "Event-state NC-CoPo with CQR-lite is the current best audited coverage-repair route "
                "on the landed Block 3 temporal surface, but it is not yet solved coverage."
            ),
            "blocked_by": [
                "CQR-lite mean c90 remains below 0.88/0.90 target" if not cqr_near_threshold else None,
                "External public validation is not passed" if not external_ready else None,
            ],
        },
        "strict_counterfactual_claim": {
            "status": "passed" if strict_passed else "not_passed",
            "allowed": "Text/EDGAR/full same-row ablation differences are auditable with stable row keys." if strict_passed else "Do not claim same-row counterfactual evidence yet.",
            "blocked_by": [] if strict_passed else ["latest strict row-key audit not passed"],
        },
        "text_embedding_claim": {
            "status": "not_passed" if not embedding_ready else "passed",
            "allowed": "Clean baseline embedding only; no best-representation claim." if not embedding_ready else "Embedding bakeoff passed under frozen-panel protocol.",
            "blocked_by": ["matched 1.5B-vs-7B downstream pair audit not passed"] if not embedding_ready else [],
        },
        "source_scaling_novelty_claim": {
            "status": "not_passed" if not source_ready else "passed",
            "allowed": "Do not claim source-scaling novelty while positive source-scale rows are zero." if not source_ready else "Source-scaling path is activated and eligible for mechanism claims.",
            "blocked_by": ["source_scale_positive_rows == 0"] if not source_ready else [],
        },
        "oral_grade_readiness": {
            "status": "not_passed",
            "allowed": "Use as an execution roadmap, not as a current acceptance claim.",
            "blocked_by": [item for item in oral_blockers if item],
        },
    }
    for item in claim_levels.values():
        item["blocked_by"] = [x for x in item.get("blocked_by", []) if x]

    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": "partial",
        "method_name": "Auditable Event-State NC-CoPo Conformal Forecasting",
        "problem_structure": [
            "continuous time-series state",
            "discrete event shocks",
            "heavy-tailed or zero-inflated targets",
            "prediction intervals that must be trustworthy",
        ],
        "not_a_bag_of_models_contract": [
            "single frozen data/pointer protocol",
            "single row identity contract for train/inference/audit",
            "shared event-state representation with target-specific distribution heads allowed",
            "same NC-CoPo/CQR-lite calibration layer across targets",
            "same strict counterfactual and embedding bakeoff gates for source/text claims",
        ],
        "latest_artifacts": {
            "counterfactual": str(counter_path) if counter_path else None,
            "dashboard": str(dashboard_path) if dashboard_path else None,
            "tracker": str(tracker_path) if tracker_path else None,
            "embedding_representation": str(rep_path) if rep_path else None,
        },
        "coverage_summary": coverage,
        "counterfactual_summary": counter_summary,
        "embedding_summary": embedding,
        "gate_statuses": {
            "cqr_lite_temporal_benchmark": cqr_gate,
            "gpd_evt_temporal_benchmark": gpd_gate,
            "source_scaling_activation": source_gate,
            "external_public_validation": external_gate,
        },
        "claim_levels": claim_levels,
        "generalization_validation_ladder": [
            "Level 0: synthetic jump/heavy-tail sanity where the event mechanism is known.",
            "Level 1: Block 3 internal temporal surface across targets/horizons/ablations.",
            "Level 2: same-row counterfactual for external sources/text under stable row keys.",
            "Level 3: public external datasets with event proxies and heavy-tailed targets.",
            "Level 4: domain transfer report showing the same contract works without VC-specific features.",
        ],
        "interpretation_lock": [
            "Do not claim NeurIPS acceptance certainty; only claim audit-backed readiness milestones.",
            "Do not call this a universal SOTA replacement until public external validation passes.",
            "Do not headline text embeddings until bakeoff and downstream matched benchmark pass.",
            "Do not headline source-scaling novelty until source_scale_positive_rows is nonzero.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = OUT_DIR / f"r14_event_state_method_contract_{suffix}.json"
    out_md = out_json.with_suffix(".md")
    out_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
    out_md.write_text("# R14 Event-State Method Contract\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"OK: {out_json}")
    print(f"OK: {out_md}")
    print(json.dumps({
        "status": report["status"],
        "strict_counterfactual": claim_levels["strict_counterfactual_claim"]["status"],
        "cqr_lite_c90_mean_raw_all": coverage.get("cqr_lite_c90_mean"),
        "cqr_lite_delta_mean_raw_all": coverage.get("cqr_lite_delta_mean"),
        "canonical_cqr_lite_c90_mean": coverage.get("canonical_cqr_lite_c90_mean"),
        "canonical_cqr_lite_delta_mean": coverage.get("canonical_cqr_lite_delta_mean"),
        "metric_read_errors": coverage.get("metric_read_errors"),
        "embedding_claim": claim_levels["text_embedding_claim"]["status"],
        "source_scaling_claim": claim_levels["source_scaling_novelty_claim"]["status"],
        "oral_grade_readiness": claim_levels["oral_grade_readiness"]["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
