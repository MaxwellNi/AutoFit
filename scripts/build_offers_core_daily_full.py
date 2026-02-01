#!/usr/bin/env python
"""
Build full-scale offers_core_daily from raw offers Delta.
Streaming mode: uses DeltaTable scanner, processes in chunks, dedup via SQLite.
Output: offers_core_daily.parquet, offers_static.parquet, MANIFEST.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

SNAPSHOT_COL = "crawled_date_day"
CORE_COLS = [
    "platform_name", "offer_id", "crawled_date_day",
    "funding_goal_usd", "funding_raised_usd", "investors_count", "is_funded",
    "cik", "link", "hash_id",
    "datetime_open_offering", "datetime_close_offering",
]
DEDUP_COL = "processed_datetime"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full offers_core_daily (streaming, no OOM).")
    parser.add_argument("--raw_offers_delta", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--chunk_rows", type=int, default=None)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--output_base", type=str, default="offers_core_daily", help="Base name: offers_core_snapshot or offers_core_daily")
    args = parser.parse_args()

    out_core = args.output_dir / f"{args.output_base}.parquet"
    out_static = args.output_dir / "offers_static.parquet"
    if out_core.exists() and args.overwrite != 1:
        print(f"ERROR: {out_core} exists and --overwrite 0.", file=sys.stderr)
        sys.exit(1)

    raw_path = args.raw_offers_delta.resolve()
    if not raw_path.exists() or not (raw_path / "_delta_log").exists():
        print(f"ERROR: not a Delta table: {raw_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from deltalake import DeltaTable
    except ImportError:
        raise RuntimeError("deltalake required")

    dt = DeltaTable(str(raw_path))
    delta_version = dt.version()
    files_list = dt.file_uris() if hasattr(dt, "file_uris") else (dt.files() if hasattr(dt, "files") else [])
    active_files = len(files_list)

    schema_names = [f.name for f in dt.schema().fields]
    read_cols = [c for c in CORE_COLS + [DEDUP_COL] if c in schema_names]
    if DEDUP_COL not in read_cols and "processed_datetime" in schema_names:
        read_cols.append("processed_datetime")
    if "crawled_date_day" not in read_cols:
        read_cols.append("crawled_date_day")

    chunk_rows = args.chunk_rows
    if chunk_rows is None:
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024**3)
        except Exception:
            avail_gb = 16.0
        chunk_rows = min(25_000_000, max(2_000_000, int(0.4 * max(4, avail_gb - 8) * 1024**3 / 600)))

    print(f"build_offers_core_daily: delta={raw_path} version={delta_version} active_files={active_files}", flush=True)
    print(f"build_offers_core_daily: snapshot_col={SNAPSHOT_COL} chunk_rows={chunk_rows:,} output={args.output_dir}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dset = dt.to_pyarrow_dataset()
    limit_rows = args.limit_rows

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    try:
        conn = sqlite3.connect(db_path)
        raw_rows_scanned = 0
        rows_written = 0
        chunk_buf: List[pd.DataFrame] = []
        buf_rows = 0
        schema_created = False

        for i, batch in enumerate(dset.scanner(columns=read_cols, batch_size=200_000).to_batches()):
            df = batch.to_pandas()
            raw_rows_scanned += len(df)
            df["entity_id"] = df["platform_name"].astype(str) + "|" + df["offer_id"].astype(str)
            df["snapshot_ts"] = pd.to_datetime(df[SNAPSHOT_COL], errors="coerce", utc=True)
            df = df.dropna(subset=["entity_id", "snapshot_ts"])
            chunk_buf.append(df)
            buf_rows += len(df)

            if buf_rows >= chunk_rows or (limit_rows and raw_rows_scanned >= limit_rows):
                merged = pd.concat(chunk_buf, ignore_index=True)
                if limit_rows and len(merged) > limit_rows:
                    merged = merged.head(limit_rows)
                chunk_buf, buf_rows = [], 0

                dedup_col = DEDUP_COL if DEDUP_COL in merged.columns else None
                if dedup_col:
                    merged = merged.sort_values(["entity_id", "snapshot_ts", dedup_col], kind="mergesort")
                merged = merged.drop_duplicates(subset=["entity_id", "snapshot_ts"], keep="last")

                if not merged.empty:
                    if not schema_created:
                        merged.head(0).to_sql("core", conn, if_exists="replace", index=False)
                        schema_created = True
                    for j in range(0, len(merged), 1_000_000):
                        sub = merged.iloc[j:j+1_000_000]
                        sub.to_sql("core", conn, if_exists="append", index=False, method="multi", chunksize=30)
                    rows_written += len(merged)
                del merged

            if (i + 1) % 250 == 0 or raw_rows_scanned >= 5_000_000:
                print(f"build_offers_core_daily: scanned {raw_rows_scanned:,} rows, written {rows_written:,}", flush=True)
            if limit_rows and raw_rows_scanned >= limit_rows:
                break

        if chunk_buf:
            merged = pd.concat(chunk_buf, ignore_index=True)
            dedup_col = DEDUP_COL if DEDUP_COL in merged.columns else None
            if dedup_col:
                merged = merged.sort_values(["entity_id", "snapshot_ts", dedup_col], kind="mergesort")
            merged = merged.drop_duplicates(subset=["entity_id", "snapshot_ts"], keep="last")
            if not merged.empty:
                if not schema_created:
                    merged.head(0).to_sql("core", conn, if_exists="replace", index=False)
                    schema_created = True
                for j in range(0, len(merged), 1_000_000):
                    sub = merged.iloc[j:j+1_000_000]
                    sub.to_sql("core", conn, if_exists="append", index=False, method="multi", chunksize=30)
                rows_written += len(merged)
            del merged
        chunk_buf = []

        if not schema_created or rows_written == 0:
            rows_emitted = 0
            n_entities = 0
            date_min = date_max = None
        else:
            result: pd.DataFrame | None = None
            for chunk in pd.read_sql_query("SELECT * FROM core", conn, chunksize=500_000):
                if result is None:
                    result = chunk.copy()
                else:
                    result = pd.concat([result, chunk], ignore_index=True).drop_duplicates(
                        subset=["entity_id", "snapshot_ts"], keep="last"
                    ).reset_index(drop=True)
                del chunk

            if result is not None:
                rows_emitted = len(result)
                n_entities = int(result["entity_id"].nunique())
                snap = pd.to_datetime(result["snapshot_ts"], errors="coerce")
                date_min = str(snap.min().date()) if not snap.empty else None
                date_max = str(snap.max().date()) if not snap.empty else None

                out_cols = ["entity_id", "platform_name", "offer_id", "snapshot_ts", SNAPSHOT_COL]
                out_cols += [c for c in CORE_COLS if c in result.columns and c not in out_cols]
                result = result[[c for c in out_cols if c in result.columns]]

                import pyarrow as pa
                import pyarrow.parquet as pq
                writer = None
                for j in range(0, len(result), 1_000_000):
                    sub = result.iloc[j:j+1_000_000]
                    tbl = pa.Table.from_pandas(sub, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(str(out_core), tbl.schema)
                    writer.write_table(tbl)
                if writer:
                    writer.close()

                static = result.drop_duplicates(subset=["entity_id"], keep="last").copy()
                static["static_snapshot_ts"] = pd.to_datetime(static["snapshot_ts"], errors="coerce", utc=True)
                static_cols = ["entity_id", "platform_name", "offer_id", "static_snapshot_ts"]
                static_cols += [c for c in ["cik", "link", "hash_id", "datetime_open_offering", "datetime_close_offering"] if c in static.columns]
                static = static[[c for c in static_cols if c in static.columns]]
                static.to_parquet(out_static, index=False)

                del result, static
            else:
                rows_emitted = 0
                n_entities = 0
                date_min = date_max = None
        conn.close()
    finally:
        Path(db_path).unlink(missing_ok=True)

    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root).stdout.strip() or "unknown"
    manifest = {
        "raw_offers_delta": str(raw_path),
        "delta_version": delta_version,
        "active_files": active_files,
        "rows_scanned": raw_rows_scanned,
        "rows_emitted": rows_emitted,
        "n_unique_entities": n_entities,
        "date_min": date_min,
        "date_max": date_max,
        "snapshot_col": SNAPSHOT_COL,
        "output_base": args.output_base,
        "git_head": git_head,
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_core} ({rows_emitted:,} rows), {out_static}", flush=True)


if __name__ == "__main__":
    main()
