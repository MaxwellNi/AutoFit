#!/usr/bin/env python3
"""Audit whether a real embedding-model bake-off has been executed."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_embedding_bakeoff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")


def _latest(pattern: str) -> Path | None:
    paths = sorted(ROOT.glob(pattern))
    return paths[-1] if paths else None


def _load(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        return json.load(open(path))
    except Exception as exc:
        return {"_load_error": f"{type(exc).__name__}:{exc}", "_path": str(path)}


def main() -> int:
    candidates = []
    for meta in sorted(ROOT.glob("runs/text_embeddings*/embedding_metadata.json")):
        try:
            payload = json.load(open(meta))
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}:{exc}"}
        candidates.append({"metadata_path": str(meta), "metadata": payload})
    pair_path = _latest("runs/audits/r14_embedding_downstream_pair_audit_*.json")
    pair_payload = _load(pair_path)
    pair_status = None if pair_payload is None else pair_payload.get("status")
    status = "not_passed"
    if len(candidates) >= 2:
        status = "passed" if pair_status == "passed" else "partial"
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": status,
        "n_embedding_artifact_candidates": len(candidates),
        "candidates": candidates,
        "downstream_pair_audit": {
            "latest_artifact": str(pair_path) if pair_path else None,
            "status": pair_status,
            "summary": None if pair_payload is None else pair_payload.get("summary"),
        },
        "required_for_pass": [
            "At least two independently generated embedding artifacts on the same frozen panel.",
            "Same downstream paired benchmark protocol for each artifact.",
            "Event-probe and counterfactual audits comparing the candidates.",
        ],
        "sota_shortlist_to_run": [
            "current: Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            "general strong embedding candidate from current MTEB-style shortlist",
            "finance/legal long-context embedding candidate",
            "domain-adapted startup/offering text candidate if available",
        ],
        "claim_lock": "Do not claim embedding SOTA or optimality from a single embedding artifact.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
    OUT_MD.write_text("# R14 Embedding Bakeoff Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"OK: {OUT_JSON}")
    print(f"OK: {OUT_MD}")
    print(json.dumps({"status": report["status"], "n_embedding_artifact_candidates": len(candidates), "downstream_pair_status": pair_status, "claim_lock": report["claim_lock"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())