#!/usr/bin/env python
"""Write MANIFEST.json for an existing edgar_feature_store_full_daily dir (retrofit)."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parent.parent


def _delta_version_and_files(path: Path) -> tuple[int | None, int]:
    try:
        from deltalake import DeltaTable
    except ImportError:
        return None, 0
    if not path.exists() or not (path / "_delta_log").exists():
        return None, 0
    try:
        dt = DeltaTable(str(path))
        ver = dt.version()
        fl = dt.file_uris() if hasattr(dt, "file_uris") else (dt.files() if hasattr(dt, "files") else [])
        return ver, len(fl)
    except Exception:
        return None, 0


def _snapshots_stats(path: Path | None) -> dict:
    out = {"rows": 0, "unique_cik": 0, "unique_keys": ""}
    if not path or not path.exists():
        return out
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        out["rows"] = len(df)
        if "cik" in df.columns:
            out["unique_cik"] = int(df["cik"].nunique())
        key_cols = [c for c in ["cik", "crawled_date_day", "snapshot_ts", "platform_name", "offer_id"] if c in df.columns]
        out["unique_keys"] = "|".join(key_cols[:3])
    except Exception:
        pass
    return out


def _output_stats(output_path: Path) -> dict:
    out: dict = {"rows": 0, "columns": []}
    try:
        import pyarrow.dataset as ds
        dset = ds.dataset(str(output_path), format="parquet", partitioning="hive", exclude_invalid_files=True, ignore_prefixes=["_"])
        out["columns"] = dset.schema.names if hasattr(dset.schema, "names") else [f.name for f in dset.schema]
        out["rows"] = sum(b.num_rows for b in dset.scanner(batch_size=200_000).to_batches())
    except Exception:
        try:
            import pandas as pd
            df = pd.read_parquet(output_path)
            out["rows"] = len(df)
            out["columns"] = list(df.columns)
        except Exception:
            pass
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Write MANIFEST.json for existing edgar_store_full_daily dir.")
    p.add_argument("--edgar_store_dir", type=Path, required=True)
    p.add_argument("--raw_edgar_delta", type=Path, default=Path("data/raw/edgar/accessions"))
    p.add_argument("--snapshots_index_parquet", type=Path, default=None)
    args = p.parse_args()

    edgar_features = args.edgar_store_dir / "edgar_features"
    if not edgar_features.exists():
        edgar_features = args.edgar_store_dir
    edgar_ver, edgar_files = _delta_version_and_files(args.raw_edgar_delta)
    snap_stats = _snapshots_stats(args.snapshots_index_parquet)
    out_stats = _output_stats(edgar_features)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root).stdout.strip() or "unknown"

    manifest = {
        "raw_edgar_delta": str(args.raw_edgar_delta.resolve()),
        "raw_edgar_delta_version": edgar_ver,
        "raw_edgar_active_files": edgar_files,
        "snapshots_index_path": str(args.snapshots_index_parquet.resolve()) if args.snapshots_index_parquet else None,
        "snapshots_index_rows": snap_stats.get("rows", 0),
        "snapshots_unique_cik": snap_stats.get("unique_cik", 0),
        "snapshots_unique_keys": snap_stats.get("unique_keys", ""),
        "output_rows": out_stats.get("rows", 0),
        "output_columns": out_stats.get("columns", [])[:50],
        "partition_strategy": "snapshot_year",
        "git_head": git_head,
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "note": "Retrofitted by write_edgar_store_manifest.py",
    }
    manifest_path = args.edgar_store_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
