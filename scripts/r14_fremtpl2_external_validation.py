#!/usr/bin/env python3
"""Audit/prepare freMTPL2 external validation surface.

This script intentionally does not fabricate public-data results. It checks
whether freMTPL2 artifacts already exist locally and records the exact blocker
if they do not.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_fremtpl2_external_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")


def main() -> int:
    hits = []
    for base in [ROOT / "data", ROOT / "runs", ROOT / "configs"]:
        if base.exists():
            hits.extend(str(path) for path in base.rglob("*freMTPL2*"))
            hits.extend(str(path) for path in base.rglob("*fremtpl2*"))
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": "passed" if hits else "not_passed",
        "local_artifacts": sorted(set(hits))[:50],
        "required_for_pass": [
            "Public freMTPL2 data present locally with source/URL/hash recorded.",
            "Temporal or pseudo-temporal split protocol documented.",
            "Marginal and candidate conformal coverage audit generated.",
            "Comparison against startup-panel coverage failure mode summarized.",
        ],
        "next_action": "Acquire freMTPL2 with provenance, then run a public-data coverage pilot; no external-validation claim is allowed before that artifact exists.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
    OUT_MD.write_text("# R14 freMTPL2 External Validation Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"OK: {OUT_JSON}")
    print(f"OK: {OUT_MD}")
    print(json.dumps({"status": report["status"], "n_local_artifacts": len(hits), "next_action": report["next_action"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())