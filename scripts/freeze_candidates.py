#!/usr/bin/env python
"""
Scan runs for freeze candidates; recommend SELECTED based on MANIFEST
(rows_scanned ~112M for offers, built_at, delta_version).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

repo_root = Path(__file__).resolve().parent.parent
TARGET_OFFERS_ROWS = 112_000_000


def _manifest_if_exists(dir_path: Path) -> Dict[str, Any] | None:
    manifest_path = dir_path / "MANIFEST.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_candidates(runs: Path, prefix: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if not runs.exists():
        return candidates
    for d in sorted(runs.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        m = _manifest_if_exists(d)
        entry: Dict[str, Any] = {
            "path": str(d.relative_to(repo_root)),
            "name": d.name,
            "manifest_exists": m is not None,
            "built_at": m.get("built_at") if m else None,
            "rows_scanned": m.get("rows_scanned") or m.get("raw_rows_scanned") if m else None,
            "rows_emitted": m.get("rows_emitted") or m.get("output_rows") if m else None,
            "delta_version": m.get("delta_version") or m.get("raw_edgar_delta_version") or m.get("raw_offers_delta_version") if m else None,
            "active_files": m.get("active_files") or m.get("raw_offers_active_files") or m.get("raw_edgar_active_files") if m else None,
            "columns": m.get("output_columns") or m.get("columns", [])[:30] if m else None,
        }
        candidates.append(entry)
    return sorted(candidates, key=lambda x: (x.get("built_at") or ""), reverse=True)


def _recommend_selected(candidates: List[Dict[str, Any]], artifact_type: str) -> str | None:
    if not candidates:
        return None
    with_manifest = [c for c in candidates if c.get("manifest_exists")]
    if not with_manifest:
        return candidates[0].get("path")
    if artifact_type in ("offers_core_full_daily", "offers_core_full_snapshot"):
        best = None
        best_diff = float("inf")
        for c in with_manifest:
            rs = c.get("rows_scanned")
            if rs is None:
                continue
            diff = abs(rs - TARGET_OFFERS_ROWS)
            if diff < best_diff:
                best_diff = diff
                best = c.get("path")
        return best or with_manifest[0].get("path")
    return with_manifest[0].get("path")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze candidates with SELECTED recommendation.")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    runs = repo_root / "runs"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cats = {
        "offers_core_full_daily": "offers_core_full_daily",
        "offers_core_full_snapshot": "offers_core_full_snapshot",
        "edgar_feature_store_full_daily": "edgar_feature_store_full_daily",
        "multiscale_full": "multiscale_full",
    }
    result: Dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "target_offers_rows": TARGET_OFFERS_ROWS,
        "candidates": {},
        "selected": {},
        "is_full_scale": {},
    }

    for key, prefix in cats.items():
        cands = _list_candidates(runs, prefix)
        result["candidates"][key] = cands
        sel = _recommend_selected(cands, key)
        result["selected"][key] = sel
        if sel:
            sel_entry = next((c for c in cands if c.get("path") == sel), None)
            if sel_entry and key.startswith("offers_core"):
                rs = sel_entry.get("rows_scanned")
                result["is_full_scale"][key] = rs is not None and rs >= TARGET_OFFERS_ROWS * 0.99
            else:
                result["is_full_scale"][key] = sel_entry.get("manifest_exists", False)

    json_path = args.output_dir / "freeze_candidates.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)

    md_lines = [
        "# Freeze Candidates",
        "",
        f"- Checked: {result['checked_at']}",
        f"- Target offers rows: {result['target_offers_rows']:,}",
        "",
        "## SELECTED (recommended)",
        "",
    ]
    for k, v in result["selected"].items():
        md_lines.append(f"- **{k}**: {v}")
    md_lines.append("")
    md_lines.append("## Candidates by artifact")
    for k, cands in result["candidates"].items():
        md_lines.append(f"\n### {k}")
        for c in cands[:8]:
            sel = " [SELECTED]" if c.get("path") == result["selected"].get(k) else ""
            rs = c.get("rows_scanned")
            rs_s = f"{rs:,}" if rs is not None else "-"
            md_lines.append(f"- {c['path']}{sel}: rows_scanned={rs_s}, built_at={c.get('built_at')}")

    md_path = args.output_dir / "freeze_candidates.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
