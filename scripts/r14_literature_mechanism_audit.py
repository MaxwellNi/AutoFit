#!/usr/bin/env python3
"""Audit the frontier-mechanism registry for source/regime replacement paths."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "research" / "frontier_mechanism_audit.json"


REQUIRED = {
    "id",
    "status",
    "frontier_signals",
    "source_verification",
    "implementation_surface",
    "hypothesis",
    "required_gates",
    "current_evidence",
}


def _load() -> dict[str, Any]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def main() -> int:
    data = _load()
    mechanisms = data.get("mechanisms", [])
    errors = []
    for idx, row in enumerate(mechanisms):
        missing = sorted(REQUIRED - set(row))
        if missing:
            errors.append({"index": idx, "id": row.get("id"), "missing": missing})
        for list_key in ("frontier_signals", "implementation_surface", "required_gates", "current_evidence"):
            if not isinstance(row.get(list_key), list) or not row.get(list_key):
                errors.append({"index": idx, "id": row.get("id"), "invalid_list": list_key})

    status_counts = Counter(str(row.get("status")) for row in mechanisms)
    verification_counts = Counter(str(row.get("source_verification")) for row in mechanisms)
    active = [row for row in mechanisms if row.get("status") == "active_candidate"]
    not_implemented = [row for row in mechanisms if row.get("status") == "candidate_not_implemented"]
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "registry": str(REGISTRY.relative_to(ROOT)),
        "registry_name": data.get("registry_name"),
        "registry_version": data.get("registry_version"),
        "status": "passed" if not errors else "not_passed",
        "n_mechanisms": len(mechanisms),
        "status_counts": dict(status_counts),
        "source_verification_counts": dict(verification_counts),
        "n_active_candidates": len(active),
        "n_candidate_not_implemented": len(not_implemented),
        "errors": errors,
        "claim_boundary": data.get("claim_boundary", []),
        "active_candidate_ids": [row.get("id") for row in active],
        "not_implemented_ids": [row.get("id") for row in not_implemented],
        "highest_priority_next_steps": [
            "Run source-regime conformal audit over landed metrics.",
            "Complete formal drift-guard temporal rerun for h30/core_text weak coverage cell.",
            "Run official-source refresh for QDF/JAPAN/Selective Learning/retrieval lines before paper claims.",
            "Promote only mechanisms with linked artifacts, not literature plausibility alone.",
        ],
        "mechanisms": mechanisms,
    }
    out_json = ROOT / "runs" / "audits" / f"r14_literature_mechanism_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out_md.write_text("# R14 Literature Mechanism Audit\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n```\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "n_mechanisms", "status_counts", "source_verification_counts", "active_candidate_ids", "not_implemented_ids", "errors")}, indent=2, ensure_ascii=False, default=str))
    print(out_json)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())