#!/usr/bin/env python3
"""Audit source/text path activation in landed mainline metrics."""

from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_source_path_activation_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")


def _latest(pattern: str):
    paths = sorted(glob.glob(str(ROOT / pattern)))
    return paths[-1] if paths else None


def main() -> int:
    text_path = _latest("runs/audits/r14_text_edgar_signal_audit_*.json")
    trunk_path = _latest("runs/audits/r14_mainline_trunk_signal_audit_*.json")
    text = json.load(open(text_path)) if text_path else {}
    trunk = json.load(open(trunk_path)) if trunk_path else {}
    source = text.get("result_delta_audit", {}).get("source_scale_activation", {})
    guard = trunk.get("guard_summary", {})
    positive = max(int(source.get("positive_rows") or 0), int(guard.get("source_scale_positive_rows") or 0))
    observed = max(int(source.get("observed_rows") or 0), int(guard.get("source_scale_observed_rows") or 0))
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": "passed" if positive > 0 else "not_passed",
        "source_scale_positive_rows": positive,
        "source_scale_observed_rows": observed,
        "source_scaling_enabled_counts": guard.get("lane_source_scaling_enabled"),
        "source_scale_silently_dead_counts": guard.get("lane_source_scale_silently_dead"),
        "source_scale_fallback_active_counts": guard.get("lane_ss_fallback_active"),
        "source_scale_fallback_no_op_counts": guard.get("lane_ss_fallback_env_requested_no_op"),
        "residual_blend_observed_rows": guard.get("residual_blend_observed_rows"),
        "residual_blend_positive_rows": guard.get("residual_blend_positive_rows"),
        "residual_blend_mean": guard.get("residual_blend_mean"),
        "text_audit": text_path,
        "trunk_audit": trunk_path,
        "guard_summary": guard,
        "pass_rule": "passed only when lane_source_scale_strength has at least one positive landed metrics row; enabled/fallback/blend counts are diagnostic evidence, not sufficient by themselves.",
        "required_next_fix": "Instrument and rerun source path so text/EDGAR scale is nonzero, then prove paired/counterfactual benefit.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
    OUT_MD.write_text("# R14 Source Path Activation Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"OK: {OUT_JSON}")
    print(f"OK: {OUT_MD}")
    print(json.dumps({"status": report["status"], "source_scale_positive_rows": positive, "source_scale_observed_rows": observed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())