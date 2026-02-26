#!/usr/bin/env python3
"""Build V7.3 handoff pack from Block3 truth-pack artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRUTH_PACK_DIR = ROOT / "docs" / "benchmarks" / "block3_truth_pack"
DEFAULT_MASTER_DOC = ROOT / "docs" / "AUTOFIT_V72_EVIDENCE_MASTER_20260217.md"
DEFAULT_STATUS_DOC = ROOT / "docs" / "BLOCK3_MODEL_STATUS.md"
DEFAULT_RESULTS_DOC = ROOT / "docs" / "BLOCK3_RESULTS.md"
DEFAULT_V73_SPEC_DOC = ROOT / "docs" / "BLOCK3_V73_RESEARCH_EXECUTION_SPEC_20260225.md"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _to_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(float(v))
    except Exception:
        return None


def _target_family(target: str) -> str:
    if target == "investors_count":
        return "count"
    if target == "is_funded":
        return "binary"
    if target == "funding_raised_usd":
        return "heavy_tail"
    return "unknown"


def _horizon_band(h: int) -> str:
    if h <= 7:
        return "short"
    if h <= 14:
        return "mid"
    return "long"


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
    return doc_text + "\n" + heading + "\n\n" + begin + "\n" + body + "\n" + end + "\n"


def _render_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---" for _ in cols]) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def _write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in cols})


def _build_v73_reuse_manifest(tp_dir: Path) -> List[Dict[str, Any]]:
    inv_rows = _read_csv(tp_dir / "condition_inventory_full.csv")
    cond_rows = _read_csv(tp_dir / "condition_leaderboard.csv")
    miss_rows = _read_csv(tp_dir / "missing_key_manifest.csv")

    key_to_bench: Dict[Tuple[str, str, str, int], str] = {}
    for r in cond_rows:
        h = _to_int(r.get("horizon"))
        if h is None:
            continue
        key = (str(r.get("task")), str(r.get("ablation")), str(r.get("target")), h)
        bench_dirs = str(r.get("bench_dirs") or "").split(";")
        first = next((x for x in bench_dirs if x), "")
        key_to_bench[key] = first

    miss_set = set()
    for r in miss_rows:
        h = _to_int(r.get("horizon"))
        if h is None:
            continue
        miss_set.add((str(r.get("task")), str(r.get("ablation")), str(r.get("target")), h))

    out: List[Dict[str, Any]] = []
    for r in inv_rows:
        h = _to_int(r.get("horizon"))
        if h is None:
            continue
        key = (str(r.get("task")), str(r.get("ablation")), str(r.get("target")), h)
        strict_completed = _to_bool(r.get("strict_completed"))
        needs_rerun = (key in miss_set) or (not strict_completed)
        out.append(
            {
                "task": key[0],
                "ablation": key[1],
                "target": key[2],
                "horizon": key[3],
                "needs_rerun": str(needs_rerun).lower(),
                "reuse_from_run": "" if needs_rerun else key_to_bench.get(key, ""),
                "reuse_reason": (
                    "missing_from_v72_coverage_manifest"
                    if key in miss_set
                    else ("strict_not_materialized" if not strict_completed else "strict_materialized_reuse_enabled")
                ),
            }
        )
    out.sort(key=lambda x: (x["task"], x["ablation"], x["target"], int(x["horizon"])))
    return out


def _build_v73_champion_component_map(tp_dir: Path) -> List[Dict[str, Any]]:
    cond_rows = _read_csv(tp_dir / "condition_leaderboard.csv")
    grouped: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    for r in cond_rows:
        if not _to_bool(r.get("condition_completed")):
            continue
        target = str(r.get("target"))
        h = _to_int(r.get("horizon"))
        ablation = str(r.get("ablation"))
        model = str(r.get("best_model"))
        if not target or h is None or not ablation or not model:
            continue
        grouped[(_target_family(target), _horizon_band(h), ablation)][model] += 1

    components = {
        "count": "Two-part count head, non-negative inverse-safe postprocess, spike sentinel, NBEATS/NHITS anchors.",
        "binary": "Discrete-time hazard head, OOF-only calibration auto-mode, PatchTST/NHITS dual anchors.",
        "heavy_tail": "Huber+Quantile dual objective, horizon-aware anchor switch, tail metrics gate (q90/q95).",
        "unknown": "Fallback robust ensemble with strict guard telemetry.",
    }
    risks = {
        "count": "Over-clipping can suppress true extremes; monitor tail metrics and spike sentinel hit rate.",
        "binary": "Calibration gain may not translate to ranking; enforce joint MAE+calibration gate.",
        "heavy_tail": "Tail-focused objectives can regress central tendency on short horizons.",
        "unknown": "Template mismatch on unseen distribution slices.",
    }
    tests = {
        "count": "investors_count median gap reduction vs V7 and catastrophic_spikes == 0.",
        "binary": "Brier/ECE/LogLoss/PR-AUC improvement with monotonic violation control.",
        "heavy_tail": "Tail pinball and MAE stability across horizon bands.",
        "unknown": "Strict fairness and coverage pass without regression.",
    }
    priority = {
        "count": "critical",
        "binary": "high",
        "heavy_tail": "high",
        "unknown": "medium",
    }

    out: List[Dict[str, Any]] = []
    for (family, hb, ablation), counter in sorted(grouped.items()):
        champs = [m for m, _ in counter.most_common(3)]
        out.append(
            {
                "target_family": family,
                "horizon_band": hb,
                "ablation": ablation,
                "champion_models": ",".join(champs),
                "key_components": components.get(family, components["unknown"]),
                "transfer_priority": priority.get(family, priority["unknown"]),
                "risk": risks.get(family, risks["unknown"]),
                "verification_test": tests.get(family, tests["unknown"]),
            }
        )
    return out


def _build_v73_rl_policy_spec(tp_dir: Path) -> Dict[str, Any]:
    summary = _read_json(tp_dir / "truth_pack_summary.json")
    pilot = _read_json(tp_dir / "v72_pilot_gate_report.json")
    return {
        "generated_at_utc": _utc_now(),
        "policy_name": "v73_offline_policy_v1",
        "policy_type": "contextual_bandit_with_safe_fallback",
        "state_schema": [
            "lane_family",
            "horizon_band",
            "ablation",
            "missingness_bucket",
            "nonstationarity_score",
            "periodicity_score",
            "heavy_tail_score",
            "exog_strength",
            "text_strength",
            "edgar_strength",
        ],
        "action_schema": [
            "template_id",
            "candidate_subset_id",
            "count_distribution_family",
            "binary_calibration_mode",
            "top_k",
        ],
        "reward": {
            "formula": "oof_improvement - compute_penalty - guard_penalty",
            "terms": {
                "oof_improvement": "delta of OOF MAE/logloss versus lane baseline",
                "compute_penalty": "normalized train_time and inference_time cost",
                "guard_penalty": "large penalty for fairness/coverage/guard violations",
            },
        },
        "constraints": {
            "selection_data_scope": "train_val_oof_only",
            "test_feedback_allowed": False,
            "fairness_guard_required": True,
            "coverage_threshold": 0.98,
            "spike_sentinel_required_for_count": True,
        },
        "bootstrap_context": {
            "strict_completed_conditions": summary.get("strict_completed_conditions"),
            "expected_conditions": summary.get("expected_conditions"),
            "v72_missing_keys": summary.get("v72_missing_keys"),
            "v72_coverage_ratio": summary.get("v72_coverage_ratio"),
            "v72_pilot_overall_pass": pilot.get("overall_pass"),
            "v72_overlap_keys": (pilot.get("counts") or {}).get("overlap_keys_v7_v72_non_autofit"),
        },
        "evidence_paths": {
            "truth_pack_summary": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json",
            "v72_pilot_gate_report": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json",
            "condition_leaderboard": "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
        },
    }


def _ensure_v73_spec_template(path: Path) -> None:
    if path.exists():
        return
    text = """# Block3 V7.3 Research and Execution Spec (2026-02-25)

This document defines the handoff baseline for V7.3 research and execution on the finalized freeze.

## Current Benchmark Facts

<!-- BEGIN AUTO:V73_CURRENT_FACTS -->
<!-- END AUTO:V73_CURRENT_FACTS -->

## 104-key Task Universe (keys, not scheduler jobs)

<!-- BEGIN AUTO:V73_TASK_UNIVERSE -->
<!-- END AUTO:V73_TASK_UNIVERSE -->

## Reuse Policy

<!-- BEGIN AUTO:V73_REUSE_POLICY -->
<!-- END AUTO:V73_REUSE_POLICY -->

## Champion Component Transfer Matrix

<!-- BEGIN AUTO:V73_CHAMPION_TRANSFER -->
<!-- END AUTO:V73_CHAMPION_TRANSFER -->

## V7.3 Architecture (count / binary / heavy-tail)

1. Count lane uses two-part head and strict spike-safe guards.
2. Binary lane uses hazard head with OOF-only calibration selection.
3. Heavy-tail lane uses dual-objective robust losses and tail diagnostics.
4. Routing uses lane family + horizon band + ablation + missingness bucket.

## Offline RL Policy for Routing/HPO

<!-- BEGIN AUTO:V73_OFFLINE_RL_POLICY -->
<!-- END AUTO:V73_OFFLINE_RL_POLICY -->

## Smoke / Pilot / Full Gates

1. Stage S: contract, freeze, fairness/coverage guards, lane telemetry checks.
2. Stage P: representative keys for all target families with OOF-only selection.
3. Stage F: full 104-key closure with strict-comparable reporting.

## Failure Handling (OOM / GPU reset / retry policy)

1. Never auto-cancel running jobs.
2. Retry failed keys with upgraded memory profile only when under-provisioned.
3. Persist retry provenance in queue action ledger.

## ETA and Queue Strategy

1. V72-first completion remains active until V7.2 closes strict coverage.
2. V7.3 runs use missing-key first and reuse-first submission policy.

## Reproducibility Checklist

1. Insider-only runtime (`python >= 3.11`).
2. Contract assertion before any submission.
3. Freeze pointer-only data access.
4. Train/val/OOF-only model selection and policy updates.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _update_docs(
    tp_dir: Path,
    master_doc: Path,
    status_doc: Path,
    results_doc: Path,
    spec_doc: Path,
    reuse_rows: List[Dict[str, Any]],
    champion_rows: List[Dict[str, Any]],
    rl_policy: Dict[str, Any],
) -> None:
    summary = _read_json(tp_dir / "truth_pack_summary.json")
    pilot = _read_json(tp_dir / "v72_pilot_gate_report.json")
    execution = _read_json(tp_dir / "execution_status_latest.json")
    fairness = _read_json(tp_dir / "fairness_certification_latest.json")

    baseline_rows = [
        {"metric": "generated_at_utc", "value": _utc_now(), "evidence_path": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json"},
        {"metric": "strict_conditions", "value": f"{summary.get('strict_completed_conditions')}/{summary.get('expected_conditions')}", "evidence_path": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json"},
        {"metric": "v72_coverage", "value": f"{summary.get('expected_conditions', 0) - summary.get('v72_missing_keys', 0)}/{summary.get('expected_conditions')} ({summary.get('v72_coverage_ratio')})", "evidence_path": "docs/benchmarks/block3_truth_pack/truth_pack_summary.json"},
        {"metric": "v72_pilot_overall_pass", "value": pilot.get("overall_pass"), "evidence_path": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json"},
        {"metric": "v72_global_improvement_pct", "value": (pilot.get("metrics") or {}).get("global_normalized_mae_improvement_pct"), "evidence_path": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json"},
        {"metric": "v72_investors_gap_reduction_pct", "value": (pilot.get("metrics") or {}).get("investors_count_gap_reduction_pct"), "evidence_path": "docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json"},
        {"metric": "execution_v72_progress", "value": (execution.get("v72_progress") or {}).get("bar"), "evidence_path": "docs/benchmarks/block3_truth_pack/execution_status_latest.json"},
        {"metric": "fairness_label", "value": fairness.get("label"), "evidence_path": "docs/benchmarks/block3_truth_pack/fairness_certification_latest.json"},
    ]
    baseline_md = _render_table(baseline_rows, ["metric", "value", "evidence_path"])
    baseline_md += "\n\nCondition keys are evaluation subtasks (`task × ablation × target × horizon`), not scheduler jobs."

    master_text = master_doc.read_text(encoding="utf-8") if master_doc.exists() else ""
    master_text = _replace_or_append_auto_section(
        master_text,
        "V73_DEVELOPMENT_BASELINE",
        "## V7.3 Development Baseline (2026-02-25 Snapshot)",
        baseline_md,
    )
    master_doc.write_text(master_text, encoding="utf-8")

    _ensure_v73_spec_template(spec_doc)
    spec_text = spec_doc.read_text(encoding="utf-8")

    task_rows = [
        {"task": "task1_outcome", "keys": 48, "targets": "funding_raised_usd, investors_count, is_funded"},
        {"task": "task2_forecast", "keys": 32, "targets": "funding_raised_usd, investors_count"},
        {"task": "task3_risk_adjust", "keys": 24, "targets": "funding_raised_usd, investors_count"},
    ]
    task_md = _render_table(task_rows, ["task", "keys", "targets"])
    task_md += "\n\nTotal condition keys: **104**."

    reuse_summary = Counter(r["needs_rerun"] for r in reuse_rows)
    reuse_head = [
        {
            "metric": "needs_rerun_true",
            "value": reuse_summary.get("true", 0),
            "evidence_path": "docs/benchmarks/block3_truth_pack/v73_reuse_manifest.csv",
        },
        {
            "metric": "needs_rerun_false",
            "value": reuse_summary.get("false", 0),
            "evidence_path": "docs/benchmarks/block3_truth_pack/v73_reuse_manifest.csv",
        },
    ]
    reuse_md = _render_table(reuse_head, ["metric", "value", "evidence_path"]) + "\n\n"
    reuse_md += _render_table(reuse_rows[:20], ["task", "ablation", "target", "horizon", "needs_rerun", "reuse_from_run", "reuse_reason"])

    champion_md = _render_table(
        champion_rows,
        ["target_family", "horizon_band", "ablation", "champion_models", "key_components", "transfer_priority", "risk", "verification_test"],
    )
    rl_md = "```json\n" + json.dumps(rl_policy, indent=2, ensure_ascii=True) + "\n```"

    spec_text = _replace_or_append_auto_section(spec_text, "V73_CURRENT_FACTS", "## Current Benchmark Facts", baseline_md)
    spec_text = _replace_or_append_auto_section(spec_text, "V73_TASK_UNIVERSE", "## 104-key Task Universe (keys, not scheduler jobs)", task_md)
    spec_text = _replace_or_append_auto_section(spec_text, "V73_REUSE_POLICY", "## Reuse Policy", reuse_md)
    spec_text = _replace_or_append_auto_section(spec_text, "V73_CHAMPION_TRANSFER", "## Champion Component Transfer Matrix", champion_md)
    spec_text = _replace_or_append_auto_section(spec_text, "V73_OFFLINE_RL_POLICY", "## Offline RL Policy for Routing/HPO", rl_md)
    spec_doc.write_text(spec_text, encoding="utf-8")

    now = _utc_now()
    status_text = "\n".join(
        [
            "# Block 3 Model Benchmark Status",
            "",
            f"> Last Updated: {now}",
            "> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`",
            "> V7.3 execution spec: `docs/BLOCK3_V73_RESEARCH_EXECUTION_SPEC_20260225.md`",
            "",
            "## Snapshot",
            "",
            "| Metric | Value | Evidence |",
            "|---|---:|---|",
            f"| strict_condition_completion | {summary.get('strict_completed_conditions')}/{summary.get('expected_conditions')} | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |",
            f"| v72_missing_keys | {summary.get('v72_missing_keys')} | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |",
            f"| v72_progress_bar | {(execution.get('v72_progress') or {}).get('bar')} | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |",
            f"| running_total | {execution.get('running_total')} | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |",
            f"| pending_total | {execution.get('pending_total')} | `docs/benchmarks/block3_truth_pack/execution_status_latest.json` |",
            "",
            "## Notes",
            "",
            "1. V7.3 handoff artifacts are generated under `docs/benchmarks/block3_truth_pack/`.",
            "2. Non-GitHub synchronization uses pull-from-iris rsync workflow on the target GPU host.",
            "",
        ]
    )
    status_doc.write_text(status_text, encoding="utf-8")

    results_text = "\n".join(
        [
            "# Block 3 Benchmark Results",
            "",
            f"> Last Updated: {now}",
            "> Single source of truth: `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`",
            "> V7.3 execution spec: `docs/BLOCK3_V73_RESEARCH_EXECUTION_SPEC_20260225.md`",
            "",
            "## Strict Snapshot",
            "",
            "| Metric | Value | Evidence |",
            "|---|---:|---|",
            f"| strict_records | {summary.get('strict_records')} | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |",
            f"| strict_condition_completion | {summary.get('strict_condition_completion')} | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |",
            f"| v72_pilot_overall_pass | {pilot.get('overall_pass')} | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |",
            f"| v72_global_improvement_pct | {(pilot.get('metrics') or {}).get('global_normalized_mae_improvement_pct')} | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |",
            f"| v72_investors_gap_reduction_pct | {(pilot.get('metrics') or {}).get('investors_count_gap_reduction_pct')} | `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json` |",
            f"| v73_reuse_manifest_rows | {len(reuse_rows)} | `docs/benchmarks/block3_truth_pack/v73_reuse_manifest.csv` |",
            "",
            "## Notes",
            "",
            "1. V7.3 aims to close remaining key coverage and lane-specific performance gaps without altering fairness rules.",
            "2. Reuse-first policy avoids redundant reruns of already materialized strict comparable keys.",
            "",
        ]
    )
    results_doc.write_text(results_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V7.3 handoff pack artifacts and update docs.")
    parser.add_argument("--truth-pack-dir", type=Path, default=DEFAULT_TRUTH_PACK_DIR)
    parser.add_argument("--master-doc", type=Path, default=DEFAULT_MASTER_DOC)
    parser.add_argument("--status-doc", type=Path, default=DEFAULT_STATUS_DOC)
    parser.add_argument("--results-doc", type=Path, default=DEFAULT_RESULTS_DOC)
    parser.add_argument("--spec-doc", type=Path, default=DEFAULT_V73_SPEC_DOC)
    parser.add_argument("--update-docs", action="store_true", default=False)
    args = parser.parse_args()

    tp_dir = args.truth_pack_dir.resolve()
    tp_dir.mkdir(parents=True, exist_ok=True)

    reuse_rows = _build_v73_reuse_manifest(tp_dir)
    champion_rows = _build_v73_champion_component_map(tp_dir)
    rl_policy = _build_v73_rl_policy_spec(tp_dir)

    reuse_csv = tp_dir / "v73_reuse_manifest.csv"
    champion_csv = tp_dir / "v73_champion_component_map.csv"
    rl_json = tp_dir / "v73_rl_policy_spec.json"

    _write_csv(
        reuse_csv,
        reuse_rows,
        ["task", "ablation", "target", "horizon", "needs_rerun", "reuse_from_run", "reuse_reason"],
    )
    _write_csv(
        champion_csv,
        champion_rows,
        [
            "target_family",
            "horizon_band",
            "ablation",
            "champion_models",
            "key_components",
            "transfer_priority",
            "risk",
            "verification_test",
        ],
    )
    rl_json.write_text(json.dumps(rl_policy, indent=2, ensure_ascii=True), encoding="utf-8")

    if args.update_docs:
        _update_docs(
            tp_dir=tp_dir,
            master_doc=args.master_doc.resolve(),
            status_doc=args.status_doc.resolve(),
            results_doc=args.results_doc.resolve(),
            spec_doc=args.spec_doc.resolve(),
            reuse_rows=reuse_rows,
            champion_rows=champion_rows,
            rl_policy=rl_policy,
        )

    print(
        json.dumps(
            {
                "generated_at_utc": _utc_now(),
                "v73_reuse_manifest": str(reuse_csv.relative_to(ROOT)),
                "v73_champion_component_map": str(champion_csv.relative_to(ROOT)),
                "v73_rl_policy_spec": str(rl_json.relative_to(ROOT)),
                "docs_updated": bool(args.update_docs),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
