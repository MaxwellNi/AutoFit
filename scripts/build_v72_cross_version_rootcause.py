#!/usr/bin/env python3
"""Build cross-version (V7/V7.1/V7.2) root-cause closure artifacts for Block3 truth pack."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRUTH_PACK_DIR = ROOT / "docs" / "benchmarks" / "block3_truth_pack"
DEFAULT_MASTER_DOC = ROOT / "docs" / "AUTOFIT_V72_EVIDENCE_MASTER_20260217.md"
DEFAULT_STAMP = "20260223"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else ({} if default is None else default)
    except Exception:
        return {} if default is None else default


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _median(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def _render_md_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---" for _ in columns]) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


def _replace_or_append_auto_section(doc_text: str, section_name: str, heading: str, body: str) -> str:
    begin = f"<!-- BEGIN AUTO:{section_name} -->"
    end = f"<!-- END AUTO:{section_name} -->"
    body = body.rstrip()
    if begin in doc_text and end in doc_text:
        start = doc_text.find(begin)
        finish = doc_text.find(end)
        if start >= 0 and finish >= start:
            before = doc_text[: start + len(begin)]
            after = doc_text[finish:]
            return before + "\n" + body + "\n" + after
    if not doc_text.endswith("\n"):
        doc_text += "\n"
    return (
        doc_text
        + "\n"
        + heading
        + "\n\n"
        + begin
        + "\n"
        + body
        + "\n"
        + end
        + "\n"
    )


def _build_snapshot(tp_dir: Path) -> Dict[str, Any]:
    condition_rows = _read_csv(tp_dir / "condition_leaderboard.csv")
    failure_rows = _read_csv(tp_dir / "failure_taxonomy.csv")
    overlap_rows = _read_csv(tp_dir / "v71_vs_v7_overlap.csv")
    lineage_rows = _read_csv(tp_dir / "autofit_lineage.csv")
    pilot = _read_json(tp_dir / "v72_pilot_gate_report.json")
    fair = _read_json(tp_dir / "fairness_certification_latest.json")
    missing_summary = _read_json(tp_dir / "missing_key_manifest_summary.json")

    strict_total = int(len(condition_rows))
    v72_done = int(missing_summary.get("v72_strict_keys", 0))
    v72_missing = int(missing_summary.get("missing_keys", max(0, strict_total - v72_done)))

    champion_counts = Counter()
    for row in condition_rows:
        cat = str(row.get("best_category", "") or "")
        if cat:
            champion_counts[cat] += 1

    v71_v7_count_gain = _median(
        _to_float(r.get("relative_gain_pct"))
        for r in overlap_rows
        if str(r.get("target")) == "investors_count"
    )
    v71_v7_binary_gain = _median(
        _to_float(r.get("relative_gain_pct"))
        for r in overlap_rows
        if str(r.get("target")) == "is_funded"
    )

    autofit_count_gap = _median(
        _to_float(r.get("autofit_gap_pct"))
        for r in condition_rows
        if str(r.get("target")) == "investors_count"
    )
    autofit_binary_gap = _median(
        _to_float(r.get("autofit_gap_pct"))
        for r in condition_rows
        if str(r.get("target")) == "is_funded"
    )

    issue_counts = Counter(str(r.get("issue_type", "")) for r in failure_rows)

    return {
        "generated_at_utc": _utc_now(),
        "strict_comparable_conditions": {
            "completed": strict_total,
            "expected": strict_total,
        },
        "v72_coverage": {
            "completed": v72_done,
            "expected": strict_total,
            "missing": v72_missing,
            "ratio": float(v72_done) / float(strict_total) if strict_total else 0.0,
        },
        "gate_status": {
            "overall_pass": bool(pilot.get("overall_pass", False)),
            "fairness_pass_100": bool((pilot.get("checks") or {}).get("fairness_pass_100", False)),
            "global_normalized_mae_improvement_pct": _to_float((pilot.get("metrics") or {}).get("global_normalized_mae_improvement_pct")),
            "investors_count_gap_reduction_pct": _to_float((pilot.get("metrics") or {}).get("investors_count_gap_reduction_pct")),
        },
        "fairness_certification": {
            "overall_certified": bool(fair.get("overall_certified", False)),
            "label": str(fair.get("label", "UNKNOWN")),
        },
        "champion_distribution": {
            "deep_classical": int(champion_counts.get("deep_classical", 0)),
            "transformer_sota": int(champion_counts.get("transformer_sota", 0)),
            "foundation": int(champion_counts.get("foundation", 0)),
            "autofit": int(champion_counts.get("autofit", 0)),
        },
        "cross_version_signals": {
            "v71_vs_v7_investors_count_median_gain_pct": v71_v7_count_gain,
            "v71_vs_v7_is_funded_median_gain_pct": v71_v7_binary_gain,
            "strict_autofit_investors_count_median_gap_pct": autofit_count_gap,
            "strict_autofit_is_funded_median_gap_pct": autofit_binary_gap,
            "failure_issue_counts": dict(issue_counts),
        },
        "evidence_paths": {
            "condition_leaderboard": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
            "failure_taxonomy": "docs/benchmarks/block3_truth_pack/failure_taxonomy.csv",
            "v71_vs_v7_overlap": "docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv",
            "autofit_lineage": "docs/benchmarks/block3_truth_pack/autofit_lineage.csv",
            "v72_pilot_gate_report": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
            "fairness_certification": "docs/benchmarks/block3_truth_pack/fairness_certification_latest.json",
            "missing_manifest_summary": "docs/benchmarks/block3_truth_pack/missing_key_manifest_summary.json",
        },
    }


def _build_rootcause_matrix(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    v72_cov = snapshot.get("v72_coverage", {})
    gate = snapshot.get("gate_status", {})
    signals = snapshot.get("cross_version_signals", {})
    issue_counts = signals.get("failure_issue_counts", {}) if isinstance(signals.get("failure_issue_counts"), dict) else {}

    rows: List[Dict[str, Any]] = [
        {
            "problem_id": "count_lane_extreme_error_lineage",
            "introduced_or_observed_in": "V7.1",
            "still_unresolved_in_v72": True,
            "evidence_path": "docs/benchmarks/block3_truth_pack/failure_taxonomy.csv",
            "impact_targets": "investors_count",
            "impact_scale": f"critical_spikes={issue_counts.get('v71_count_explosion', 0)}",
            "root_mechanism": "Historical count-lane catastrophic MAE spikes indicate fragile inverse-transform/postprocess chain under sparse EDGAR slices.",
            "fix_component": "Two-part count head + spike sentinel + hard OOF reject + anchored count specialists.",
            "gate_link": "Gate-S-closure: catastrophic_spikes == 0",
        },
        {
            "problem_id": "count_lane_median_gap_persistent",
            "introduced_or_observed_in": "V7",
            "still_unresolved_in_v72": True,
            "evidence_path": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
            "impact_targets": "investors_count",
            "impact_scale": f"gap_reduction_pct={gate.get('investors_count_gap_reduction_pct')}",
            "root_mechanism": "Lane-level objective and ensemble routing have not achieved robust count generalization against deep_classical/transformer count champions.",
            "fix_component": "Count-specialized routing with NBEATS/NHITS anchors, distribution-family tuning, and tail-aware gates.",
            "gate_link": "Gate-P-closure: investors_count_gap_reduction_pct >= 50",
        },
        {
            "problem_id": "binary_lane_calibration_and_routing_gap",
            "introduced_or_observed_in": "V7",
            "still_unresolved_in_v72": True,
            "evidence_path": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
            "impact_targets": "is_funded",
            "impact_scale": f"median_gap_pct={signals.get('strict_autofit_is_funded_median_gap_pct')}",
            "root_mechanism": "Binary lane remains under-calibrated and under-routed relative to PatchTST/NHITS temporal inductive bias.",
            "fix_component": "Hazard + calibration auto-selection with binary diagnostics (Brier/ECE/LogLoss/PR-AUC) and dual-anchor policy.",
            "gate_link": "Gate-P-closure: is_funded median gap reduction >= 50%",
        },
        {
            "problem_id": "coverage_deficit_blocks_full_claim",
            "introduced_or_observed_in": "V7.2",
            "still_unresolved_in_v72": True,
            "evidence_path": "docs/benchmarks/block3_truth_pack/missing_key_manifest_summary.json",
            "impact_targets": "all_targets",
            "impact_scale": f"v72={v72_cov.get('completed')}/{v72_cov.get('expected')} missing={v72_cov.get('missing')}",
            "root_mechanism": "V7.2 strict overlap is incomplete, so full-matrix ranking and closure claims remain statistically underpowered.",
            "fix_component": "V72-first missing-key completion controller and shard prioritization by uncovered subtasks.",
            "gate_link": "Gate-F-closure: V7.2 coverage == 104/104",
        },
        {
            "problem_id": "audit_not_certified_due_spike_lineage",
            "introduced_or_observed_in": "V7.1",
            "still_unresolved_in_v72": True,
            "evidence_path": "docs/benchmarks/block3_truth_pack/fairness_certification_latest.json",
            "impact_targets": "certification",
            "impact_scale": f"label={snapshot.get('fairness_certification', {}).get('label')}",
            "root_mechanism": "Certification remains open because count spike lineage is unresolved, despite distribution diagnostics being available.",
            "fix_component": "Failure-pool rerun and spike-zero enforcement before cert refresh.",
            "gate_link": "Gate-S-closure: certification == CERTIFIED",
        },
    ]
    return rows


def _build_frontier_fix_map() -> List[Dict[str, Any]]:
    return [
        {
            "problem": "count_lane_extreme_error_lineage",
            "source": "TIDE (Intermittent Count Forecasting)",
            "mechanism": "Two-part modeling for zero inflation and positive-count magnitude with distribution-aware likelihood.",
            "integration_point": "AutoFitV72 count lane: count_two_part_head + count_distribution_family.",
            "risk": "Over-parameterization on sparse subtasks may increase variance.",
            "verification_test": "Spike sentinel zero, investors_count median gap reduction >= 50%, strict fairness unchanged.",
            "primary_link": "https://arxiv.org/abs/2502.19086",
            "status": "verified_primary",
        },
        {
            "problem": "count_lane_median_gap_persistent",
            "source": "SPADE-S",
            "mechanism": "Structured sparsity for robust forecasting under sparse observations.",
            "integration_point": "Sparse-MoE route regularization and count lane sparsity-aware weighting.",
            "risk": "Aggressive sparsity can suppress useful experts.",
            "verification_test": "OOF variance reduction without degradation on heavy-tail/binary lanes.",
            "primary_link": "https://arxiv.org/abs/2507.21155",
            "status": "verified_primary",
        },
        {
            "problem": "coverage_deficit_blocks_full_claim",
            "source": "RAFT",
            "mechanism": "Retrieval-augmented forecasting with train-only context retrieval.",
            "integration_point": "Regime retrieval standardization for lane routing features.",
            "risk": "Improper retrieval index can leak future information.",
            "verification_test": "Leakage audit pass and retrieval ablation uplift on strict keys.",
            "primary_link": "https://arxiv.org/abs/2505.04163",
            "status": "verified_primary",
        },
        {
            "problem": "binary_lane_calibration_and_routing_gap",
            "source": "TS-RAG",
            "mechanism": "Retrieval-guided temporal representation for classification and forecasting under non-stationarity.",
            "integration_point": "Binary lane routing features and hazard calibrator context features.",
            "risk": "Context mismatch across horizons can degrade calibration.",
            "verification_test": "Brier/ECE/PR-AUC deltas by horizon stay stable across ablations.",
            "primary_link": "https://arxiv.org/abs/2503.07649",
            "status": "verified_primary",
        },
        {
            "problem": "binary_lane_calibration_and_routing_gap",
            "source": "TraCeR",
            "mechanism": "Trajectory-consistent event-time modeling for temporally evolving outcomes.",
            "integration_point": "Discrete-time hazard head and monotonicity violation diagnostics.",
            "risk": "Event-time assumptions may mismatch panel irregularities.",
            "verification_test": "hazard_monotonic_violation_rate decreases and no PR-AUC regression.",
            "primary_link": "https://arxiv.org/abs/2512.18129",
            "status": "verified_primary",
        },
        {
            "problem": "binary_lane_calibration_and_routing_gap",
            "source": "NeuralSurv",
            "mechanism": "Neural survival-style calibration for time-to-event probability estimation.",
            "integration_point": "is_funded hazard calibration candidate in auto mode.",
            "risk": "Calibration can improve while MAE ranking worsens.",
            "verification_test": "Joint gate on MAE + Brier + ECE.",
            "primary_link": "https://arxiv.org/abs/2505.11054",
            "status": "verified_primary",
        },
        {
            "problem": "audit_not_certified_due_spike_lineage",
            "source": "Beyond Accuracy",
            "mechanism": "Calibration-aware time-series evaluation metrics beyond point accuracy.",
            "integration_point": "Binary diagnostics pack and certification checklist expansion.",
            "risk": "Metric inflation without practical gain if not tied to gates.",
            "verification_test": "Gate reports include binary calibration metrics with improvement thresholds.",
            "primary_link": "https://arxiv.org/abs/2510.16060",
            "status": "verified_primary",
        },
        {
            "problem": "audit_not_certified_due_spike_lineage",
            "source": "ETCE Nowcasting",
            "mechanism": "Expected temporal calibration error diagnostics for sequential predictions.",
            "integration_point": "time_consistency and hazard monotonicity certification rows.",
            "risk": "Over-fitting calibration bins on small subtasks.",
            "verification_test": "Subtask-level calibration stability with minimum support constraints.",
            "primary_link": "https://arxiv.org/abs/2510.00594",
            "status": "verified_primary",
        },
        {
            "problem": "count_lane_median_gap_persistent",
            "source": "Time-MoE",
            "mechanism": "Sparse expert activation with scalable temporal routing.",
            "integration_point": "AutoFit sparse-MoE route + offline policy action telemetry.",
            "risk": "Route collapse if confidence signal is noisy.",
            "verification_test": "Expert diversity and win-rate uplift under fixed compute budget.",
            "primary_link": "https://arxiv.org/abs/2409.16040",
            "status": "verified_primary",
        },
        {
            "problem": "count_lane_median_gap_persistent",
            "source": "Moirai-MoE",
            "mechanism": "Mixture-of-experts for regime-specialized temporal modeling.",
            "integration_point": "Champion-template anchors + sparse route prior bonuses.",
            "risk": "Complexity may hurt stability under small overlap slices.",
            "verification_test": "Reduced variance across repeated runs with stable fairness metrics.",
            "primary_link": "https://arxiv.org/abs/2410.10469",
            "status": "verified_primary",
        },
    ]


def _snapshot_md(snapshot: Dict[str, Any]) -> str:
    rows = [
        {
            "metric": "generated_at_utc",
            "value": snapshot.get("generated_at_utc"),
            "evidence_path": "docs/benchmarks/block3_truth_pack/v72_cross_version_snapshot_latest.json",
        },
        {
            "metric": "strict_comparable_completion",
            "value": "%s/%s"
            % (
                snapshot.get("strict_comparable_conditions", {}).get("completed"),
                snapshot.get("strict_comparable_conditions", {}).get("expected"),
            ),
            "evidence_path": snapshot.get("evidence_paths", {}).get("condition_leaderboard"),
        },
        {
            "metric": "v72_coverage",
            "value": "%s/%s (missing=%s)"
            % (
                snapshot.get("v72_coverage", {}).get("completed"),
                snapshot.get("v72_coverage", {}).get("expected"),
                snapshot.get("v72_coverage", {}).get("missing"),
            ),
            "evidence_path": snapshot.get("evidence_paths", {}).get("missing_manifest_summary"),
        },
        {
            "metric": "gate_p_overall_pass",
            "value": snapshot.get("gate_status", {}).get("overall_pass"),
            "evidence_path": snapshot.get("evidence_paths", {}).get("v72_pilot_gate_report"),
        },
        {
            "metric": "global_normalized_mae_improvement_pct",
            "value": snapshot.get("gate_status", {}).get("global_normalized_mae_improvement_pct"),
            "evidence_path": snapshot.get("evidence_paths", {}).get("v72_pilot_gate_report"),
        },
        {
            "metric": "investors_count_gap_reduction_pct",
            "value": snapshot.get("gate_status", {}).get("investors_count_gap_reduction_pct"),
            "evidence_path": snapshot.get("evidence_paths", {}).get("v72_pilot_gate_report"),
        },
        {
            "metric": "certification_label",
            "value": snapshot.get("fairness_certification", {}).get("label"),
            "evidence_path": snapshot.get("evidence_paths", {}).get("fairness_certification"),
        },
    ]
    return _render_md_table(rows, ["metric", "value", "evidence_path"])


def _frontier_md(rows: List[Dict[str, Any]]) -> str:
    cols = [
        "problem",
        "source",
        "mechanism",
        "integration_point",
        "risk",
        "verification_test",
        "primary_link",
        "status",
    ]
    return _render_md_table(rows, cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V7/V7.1/V7.2 cross-version root-cause closure artifacts.")
    parser.add_argument("--truth-pack-dir", type=Path, default=DEFAULT_TRUTH_PACK_DIR)
    parser.add_argument("--master-doc", type=Path, default=DEFAULT_MASTER_DOC)
    parser.add_argument("--stamp", type=str, default=DEFAULT_STAMP)
    parser.add_argument("--update-master-doc", action="store_true", default=False)
    args = parser.parse_args()

    tp_dir = args.truth_pack_dir.resolve()
    tp_dir.mkdir(parents=True, exist_ok=True)

    snapshot = _build_snapshot(tp_dir)
    rootcause_rows = _build_rootcause_matrix(snapshot)
    frontier_rows = _build_frontier_fix_map()

    snapshot_json_path = tp_dir / f"v72_cross_version_snapshot_{args.stamp}.json"
    snapshot_md_path = tp_dir / f"v72_cross_version_snapshot_{args.stamp}.md"
    snapshot_json_latest = tp_dir / "v72_cross_version_snapshot_latest.json"
    snapshot_md_latest = tp_dir / "v72_cross_version_snapshot_latest.md"

    rootcause_csv = tp_dir / "v72_cross_version_rootcause_matrix.csv"
    frontier_csv = tp_dir / "v72_frontier_fix_map_20260223.csv"

    snapshot_json = json.dumps(snapshot, indent=2, ensure_ascii=True)
    snapshot_md = _snapshot_md(snapshot)

    snapshot_json_path.write_text(snapshot_json, encoding="utf-8")
    snapshot_md_path.write_text(snapshot_md + "\n", encoding="utf-8")
    snapshot_json_latest.write_text(snapshot_json, encoding="utf-8")
    snapshot_md_latest.write_text(snapshot_md + "\n", encoding="utf-8")

    rootcause_columns = [
        "problem_id",
        "introduced_or_observed_in",
        "still_unresolved_in_v72",
        "evidence_path",
        "impact_targets",
        "impact_scale",
        "root_mechanism",
        "fix_component",
        "gate_link",
    ]
    _write_csv(rootcause_csv, rootcause_rows, rootcause_columns)

    frontier_columns = [
        "problem",
        "source",
        "mechanism",
        "integration_point",
        "risk",
        "verification_test",
        "primary_link",
        "status",
    ]
    _write_csv(frontier_csv, frontier_rows, frontier_columns)

    if args.update_master_doc:
        master_doc = args.master_doc.resolve()
        text = master_doc.read_text(encoding="utf-8") if master_doc.exists() else ""
        text = _replace_or_append_auto_section(
            text,
            "CROSS_VERSION_SNAPSHOT",
            "## Cross-Version Snapshot (V7/V7.1/V7.2)",
            snapshot_md,
        )
        text = _replace_or_append_auto_section(
            text,
            "CROSS_VERSION_ROOTCAUSE_MATRIX",
            "## Cross-Version Root-Cause Matrix",
            _render_md_table(rootcause_rows, rootcause_columns),
        )
        text = _replace_or_append_auto_section(
            text,
            "FRONTIER_FIX_MAP",
            "## Frontier-to-Fix Mapping (Primary Sources, 2026-02-23)",
            _frontier_md(frontier_rows),
        )
        master_doc.write_text(text, encoding="utf-8")

    out = {
        "generated_at_utc": _utc_now(),
        "snapshot": str(snapshot_json_latest.relative_to(ROOT)),
        "rootcause_matrix": str(rootcause_csv.relative_to(ROOT)),
        "frontier_fix_map": str(frontier_csv.relative_to(ROOT)),
        "master_doc_updated": bool(args.update_master_doc),
    }
    print(json.dumps(out, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
