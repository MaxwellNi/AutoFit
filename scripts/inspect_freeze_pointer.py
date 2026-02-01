#!/usr/bin/env python
"""
Inspect freeze pointer: verify each pointed directory exists, has MANIFEST.json,
delta versions consistent, built_at, rows. List candidate runs for comparison.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

repo_root = Path(__file__).resolve().parent.parent


def _load_pointer(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _manifest_if_exists(dir_path: Path) -> Dict[str, Any] | None:
    manifest_path = dir_path / "MANIFEST.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _dir_size_mb(path: Path) -> float:
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024 * 1024)
    except Exception:
        return 0.0


def _list_candidate_runs(pattern: str) -> List[Dict[str, Any]]:
    """List runs matching pattern (e.g. offers_core_full_daily_*)."""
    candidates: List[Dict[str, Any]] = []
    runs = repo_root / "runs"
    if not runs.exists():
        return candidates
    for d in sorted(runs.iterdir()):
        if not d.is_dir() or pattern.split("*")[0] not in d.name:
            continue
        m = _manifest_if_exists(d)
        entry: Dict[str, Any] = {
            "path": str(d.relative_to(repo_root)),
            "exists": True,
            "manifest_exists": m is not None,
            "built_at": m.get("built_at") if m else None,
            "delta_version": (m.get("delta_version") or m.get("raw_edgar_delta_version") or m.get("raw_offers_delta_version")) if m else None,
            "rows_emitted": (m.get("rows_emitted") or m.get("output_rows")) if m else None,
            "rows_scanned": (m.get("rows_scanned") or m.get("raw_rows_scanned")) if m else None,
        }
        candidates.append(entry)
    return sorted(candidates, key=lambda x: (x.get("built_at") or ""), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect freeze pointer status.")
    parser.add_argument("--pointer", type=Path, default=repo_root / "docs/audits/FULL_SCALE_POINTER.yaml")
    parser.add_argument("--output_dir", type=Path, default=repo_root / "runs/orchestrator/20260129_073037/analysis")
    args = parser.parse_args()

    pointer = _load_pointer(args.pointer)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    status: Dict[str, Any] = {
        "pointer_path": str(args.pointer),
        "stamp": pointer.get("stamp", ""),
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "entries": {},
        "all_ok": True,
        "candidates": {},
    }

    def check_entry(key: str, dir_key: str, path_val: str | None) -> None:
        if not path_val:
            return
        p = repo_root / path_val.replace("runs/", "runs/").lstrip("/")
        if p.suffix in (".parquet", ".json"):
            p = p.parent
        exists = p.exists()
        manifest = _manifest_if_exists(p) if p.is_dir() else _manifest_if_exists(p.parent) if p.parent.name == "edgar_features" else None
        if p.name == "edgar_features":
            manifest = _manifest_if_exists(p.parent)
        entry: Dict[str, Any] = {
            "path": path_val,
            "exists": exists,
            "manifest_exists": manifest is not None,
            "size_mb": round(_dir_size_mb(p), 2) if p.is_dir() else None,
            "built_at": manifest.get("built_at") if manifest else None,
            "rows_emitted": manifest.get("rows_emitted") or manifest.get("output_rows") if manifest else None,
            "rows_scanned": manifest.get("rows_scanned") or manifest.get("raw_rows_scanned") if manifest else None,
            "delta_version": manifest.get("delta_version") or manifest.get("raw_edgar_delta_version") or manifest.get("raw_offers_delta_version") if manifest else None,
        }
        status["entries"][key] = entry
        if not exists or not manifest:
            status["all_ok"] = False

    if "offers_text" in pointer:
        pt = pointer["offers_text"]
        check_entry("offers_text", "dir", pt.get("dir"))
    if "offers_core_daily" in pointer:
        pt = pointer["offers_core_daily"]
        check_entry("offers_core_daily", "dir", pt.get("dir"))
    if "offers_core_snapshot" in pointer:
        pt = pointer["offers_core_snapshot"]
        check_entry("offers_core_snapshot", "dir", pt.get("dir"))
    if "snapshots_index" in pointer:
        pt = pointer["snapshots_index"]
        offer_path = pt.get("offer_day") or pt.get("offer_day_path")
        if offer_path:
            snap_dir = str(Path(offer_path).parent)
            p = repo_root / snap_dir
            status["entries"]["snapshots_index"] = {
                "path": snap_dir,
                "exists": p.exists(),
                "manifest_exists": (p / "MANIFEST.json").exists(),
                "offer_day_exists": (repo_root / offer_path).exists() if offer_path else False,
            }
            if not p.exists():
                status["all_ok"] = False
    if "edgar_store_full_daily" in pointer:
        pt = pointer["edgar_store_full_daily"]
        check_entry("edgar_store_full_daily", "dir", pt.get("dir"))
    if "multiscale_full" in pointer:
        pt = pointer["multiscale_full"]
        check_entry("multiscale_full", "dir", pt.get("dir"))
    if "analysis" in pointer:
        pt = pointer["analysis"]
        check_entry("analysis", "dir", pt.get("dir"))

    status["candidates"]["offers_core_full_daily"] = _list_candidate_runs("offers_core_full_daily*")
    status["candidates"]["offers_core_full_snapshot"] = _list_candidate_runs("offers_core_full_snapshot*")
    status["candidates"]["edgar_feature_store"] = _list_candidate_runs("edgar_feature_store*")
    status["candidates"]["multiscale_full"] = _list_candidate_runs("multiscale_full*")

    json_path = args.output_dir / "freeze_pointer_status.json"
    json_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)

    md_lines = [
        "# Freeze Pointer Status",
        "",
        f"- **Pointer:** {args.pointer}",
        f"- **Stamp:** {status['stamp']}",
        f"- **Checked:** {status['checked_at']}",
        f"- **All OK:** {status['all_ok']}",
        "",
        "## Entries",
        "",
    ]
    for k, v in status["entries"].items():
        md_lines.append(f"### {k}")
        md_lines.append(f"- exists: {v.get('exists')}")
        md_lines.append(f"- manifest_exists: {v.get('manifest_exists')}")
        md_lines.append(f"- built_at: {v.get('built_at')}")
        md_lines.append(f"- rows_emitted: {v.get('rows_emitted')}")
        md_lines.append("")

    md_lines.append("## Candidates (by built_at desc)")
    for cat, cands in status["candidates"].items():
        md_lines.append(f"\n### {cat}")
        for c in cands[:10]:
            md_lines.append(f"- {c['path']}: manifest={c.get('manifest_exists')}, built_at={c.get('built_at')}")

    md_path = args.output_dir / "freeze_pointer_status.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {md_path}", flush=True)

    print("\n=== Final Freeze Pointer Table (Block 3 MUST read FULL_SCALE_POINTER.yaml only) ===", flush=True)
    print("| Artifact | Path | SELECTED | manifest | built_at | rows |", flush=True)
    print("|----------|------|----------|----------|----------|------|", flush=True)
    selected_dirs = set()
    for k, v in pointer.items():
        if isinstance(v, dict) and "dir" in v:
            selected_dirs.add(v["dir"])
    for cat, cands in status["candidates"].items():
        for c in cands[:5]:
            path = c.get("path", "")
            sel = "SELECTED" if path in selected_dirs else ""
            m = "yes" if c.get("manifest_exists") else "no"
            bt = str(c.get("built_at", ""))[:19]
            rows = c.get("rows_emitted") or c.get("rows_scanned") or "-"
            print(f"| {cat} | {path} | {sel} | {m} | {bt} | {rows} |", flush=True)
    print("", flush=True)


if __name__ == "__main__":
    main()
