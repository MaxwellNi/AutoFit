#!/usr/bin/env python
"""
Fix/create top-level MANIFEST.json for multiscale_full directory.
Aggregates info from nested subdirectories and source manifests.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:12]
    except Exception:
        return "unknown"


def _parquet_info(path: Path) -> Dict[str, Any]:
    """Get basic info from parquet file."""
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        return {
            "path": str(path),
            "num_rows": pf.metadata.num_rows,
            "num_columns": len(schema.names),
            "columns": schema.names[:50],  # First 50 columns
        }
    except Exception as e:
        return {"path": str(path), "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix multiscale MANIFEST.json")
    parser.add_argument("--stamp", type=str, required=True)
    parser.add_argument("--multiscale_dir", type=Path, required=True)
    parser.add_argument("--offers_daily_manifest", type=Path, required=True)
    parser.add_argument("--edgar_manifest", type=Path, required=True)
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    # Find all parquet files in the multiscale directory
    parquets = list(args.multiscale_dir.rglob("*.parquet"))
    nested_manifest = list(args.multiscale_dir.rglob("MANIFEST.json"))

    # Load source manifests for provenance
    offers_manifest = json.loads(args.offers_daily_manifest.read_text()) if args.offers_daily_manifest.exists() else {}
    edgar_manifest = json.loads(args.edgar_manifest.read_text()) if args.edgar_manifest.exists() else {}

    # Build table info
    output_tables: Dict[str, Any] = {}
    for pq_path in parquets:
        rel_path = pq_path.relative_to(args.multiscale_dir)
        output_tables[str(rel_path)] = _parquet_info(pq_path)

    # Load nested manifest if exists
    nested_info = {}
    if nested_manifest:
        for nm in nested_manifest:
            try:
                nested_info[str(nm.relative_to(args.multiscale_dir))] = json.loads(nm.read_text())
            except Exception:
                pass

    manifest = {
        "artifact": "multiscale_full",
        "stamp": args.stamp,
        "variant": "TRAIN_WIDE_FINAL",
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_head": _git_head(),
        "source_offers_daily": {
            "delta_version": offers_manifest.get("delta_version"),
            "active_files": offers_manifest.get("active_files"),
            "rows_scanned": offers_manifest.get("rows_scanned"),
            "rows_emitted": offers_manifest.get("rows_emitted"),
        },
        "source_edgar": {
            "raw_edgar_delta_version": edgar_manifest.get("raw_edgar_delta_version"),
            "raw_edgar_active_files": edgar_manifest.get("raw_edgar_active_files"),
            "output_rows": edgar_manifest.get("output_rows"),
        },
        "output_tables": output_tables,
        "nested_manifests": nested_info,
        "parquet_count": len(parquets),
    }

    # Write top-level MANIFEST.json
    manifest_path = args.multiscale_dir / "MANIFEST.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"MANIFEST exists at {manifest_path}, use --overwrite 1 to replace")
        return

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
