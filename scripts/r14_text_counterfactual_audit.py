#!/usr/bin/env python3
"""Audit whether strict text counterfactual evidence exists."""

from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_text_counterfactual_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")


def _latest_text_audit():
    paths = sorted(glob.glob(str(ROOT / "runs/audits/r14_text_edgar_signal_audit_*.json")))
    if not paths:
        return None
    return json.load(open(paths[-1]))


def main() -> int:
    text_audit = _latest_text_audit() or {}
    paired = text_audit.get("result_delta_audit", {}).get("summary", {}).get("by_ablation", {})
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": "not_passed",
        "paired_ablation_proxy_available": bool(paired),
        "paired_ablation_proxy": paired,
        "strict_counterfactual_artifacts": sorted(glob.glob(str(ROOT / "runs/audits/*text*counterfactual*.json"))),
        "why_not_passed": [
            "Current evidence compares core_text/core_edgar/full vs core_only runs.",
            "It does not hold the same row and model state fixed while replacing/removing event text.",
            "A strict pass requires a rerun retaining entity/date/row keys and controlled text perturbations.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
    OUT_MD.write_text("# R14 Text Counterfactual Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"OK: {OUT_JSON}")
    print(f"OK: {OUT_MD}")
    print(json.dumps({"status": report["status"], "paired_ablation_proxy_available": report["paired_ablation_proxy_available"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())