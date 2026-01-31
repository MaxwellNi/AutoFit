#!/usr/bin/env python
"""
Build offers_text parquet from raw offers delta (narrative columns for NBI/NCI).
Joins by entity_id + snapshot_ts. Converts array fields to joined strings.
Output: offers_text.parquet + MANIFEST.json (local, untracked).

STREAMING mode: processes in chunks to avoid OOM. Uses SQLite for disk-based
deduplication. Chunk size is auto-tuned from available RAM (40%% of avail minus
8GB headroom, ~800B/row) unless --chunk_rows overrides. Cap 5M--50M rows/chunk.

If offers_text.parquet exists and --overwrite 0, exits with error to prevent
accidental overwrite of full build by a debug run with --limit_rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent


def _join_list(val: Any) -> str:
    """Convert list/array to newline-joined string."""
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    if isinstance(val, (list, tuple)):
        return "\n".join(str(x) for x in val if x is not None and str(x).strip())
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        try:
            return "\n".join(str(x) for x in val if x is not None and str(x).strip())
        except (TypeError, ValueError):
            return str(val).strip() if val is not None else ""
    s = str(val).strip()
    return s if s else ""


def _process_chunk(
    df: pd.DataFrame,
    array_cols: List[str],
    dedup_col: str | None,
) -> pd.DataFrame:
    """Process one chunk: add entity_id, snapshot_ts, text cols; dedupe."""
    if df.empty:
        return df
    df = df.copy()
    df["entity_id"] = df["platform_name"].astype(str) + "|" + df["offer_id"].astype(str)
    df["snapshot_ts"] = pd.to_datetime(df["crawled_date_day"], errors="coerce", utc=True)
    for c in array_cols:
        if c in df.columns:
            df[f"{c}_text"] = df[c].apply(_join_list)
        else:
            df[f"{c}_text"] = ""
    text_cols = [
        "headline", "title", "description_text", "company_description", "product_description",
        "financial_condition", "financial_forecasts", "financial_risks", "offering_purpose",
        "term_sheet", "front_video_transcript"
    ] + [f"{c}_text" for c in array_cols]
    out_cols = ["entity_id", "platform_name", "offer_id", "snapshot_ts"] + [
        c for c in text_cols if c in df.columns
    ]
    df = df[[c for c in out_cols if c in df.columns]]
    df = df.dropna(subset=["entity_id", "snapshot_ts"])
    if dedup_col and dedup_col in df.columns:
        df = df.sort_values(["entity_id", "snapshot_ts", dedup_col], kind="mergesort")
    df = df.drop_duplicates(subset=["entity_id", "snapshot_ts"], keep="last")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offers_text from raw offers delta (streaming, no OOM).")
    parser.add_argument("--raw_offers_delta", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--limit_rows", type=int, default=None, help="Limit rows read (for testing); omit for full build")
    parser.add_argument("--overwrite", type=int, default=0, help="1=allow overwrite if output exists; 0=fail if exists")
    parser.add_argument("--chunk_rows", type=int, default=None, help="Rows per chunk (default: auto from 40%% of available RAM, ~800B/row)")
    args = parser.parse_args()

    chunk_rows = args.chunk_rows
    if chunk_rows is None:
        avail_gb = 16.0
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024**3)
        except Exception:
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            avail_gb = int(line.split()[1]) / 1024**2  # kB -> GB
                            break
            except Exception:
                pass
        headroom_gb = 8.0
        usable_gb = max(4.0, avail_gb - headroom_gb)
        bytes_per_row = 800
        chunk_rows = int(0.4 * usable_gb * 1024**3 / bytes_per_row)
        chunk_rows = max(5_000_000, min(50_000_000, chunk_rows))
        print(f"build_offers_text: auto chunk_rows={chunk_rows:,} (avail_ram~{avail_gb:.0f}GB)", flush=True)

    print("build_offers_text: starting (streaming mode)", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "offers_text.parquet"
    if out_path.exists() and args.overwrite != 1:
        print(f"ERROR: {out_path} exists and --overwrite 0. Refusing.", file=sys.stderr)
        sys.exit(1)

    raw_path = args.raw_offers_delta.resolve()
    if not raw_path.exists():
        print(f"ERROR: raw_offers_delta does not exist: {raw_path}", file=sys.stderr)
        sys.exit(1)
    if not (raw_path / "_delta_log").exists():
        print(f"ERROR: not a Delta table (no _delta_log): {raw_path}", file=sys.stderr)
        sys.exit(1)

    try:
        from deltalake import DeltaTable
    except ImportError:
        raise RuntimeError("deltalake required; pip install deltalake")

    print(f"build_offers_text: opening Delta {raw_path}", flush=True)
    dt = DeltaTable(str(raw_path))
    delta_version = dt.version()
    files_list = dt.file_uris() if hasattr(dt, "file_uris") else (dt.files() if hasattr(dt, "files") else [])
    raw_active_files = len(files_list)

    scalar_cols = [
        "platform_name", "offer_id", "crawled_date_day", "headline", "title", "description_text",
        "company_description", "product_description", "financial_condition", "financial_forecasts",
        "financial_risks", "offering_purpose", "term_sheet", "front_video_transcript",
    ]
    array_cols = ["use_of_funds", "reasons_to_invest", "updates", "questions"]
    schema_names = [f.name for f in dt.schema().fields]
    read_scalar = [c for c in scalar_cols if c in schema_names]
    read_arrays = [c for c in array_cols if c in schema_names]
    dedup_col = "processed_datetime" if "processed_datetime" in schema_names else None
    if dedup_col:
        read_scalar.append(dedup_col)
    read_cols = read_scalar + read_arrays

    dset = dt.to_pyarrow_dataset()
    limit_rows = args.limit_rows

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    try:
        conn = sqlite3.connect(db_path)
        raw_rows_scanned = 0
        rows_all_text_null_dropped = 0
        rows_written_to_db = 0
        chunk_buf: List[pd.DataFrame] = []
        buf_rows = 0
        schema_created = False

        print(f"build_offers_text: scanning (chunk_rows={chunk_rows:,}, limit_rows={limit_rows})", flush=True)
        for i, batch in enumerate(dset.scanner(columns=read_cols, batch_size=200_000).to_batches()):
            df = batch.to_pandas()
            raw_rows_scanned += len(df)
            chunk_buf.append(df)
            buf_rows += len(df)

            if buf_rows >= chunk_rows or (limit_rows and raw_rows_scanned >= limit_rows):
                merged = pd.concat(chunk_buf, ignore_index=True)
                if limit_rows and len(merged) > limit_rows:
                    merged = merged.head(limit_rows)
                before = len(merged)
                processed = _process_chunk(merged, array_cols, dedup_col)
                rows_all_text_null_dropped += before - len(processed) if not processed.empty else before
                chunk_buf = []
                buf_rows = 0

                if not processed.empty:
                    if not schema_created:
                        processed.head(0).to_sql("offers_text", conn, if_exists="replace", index=False)
                        schema_created = True
                    processed.to_sql("offers_text", conn, if_exists="append", index=False, method="multi", chunksize=50_000)
                    rows_written_to_db += len(processed)
                del merged, processed

            if (i + 1) % 250 == 0 or raw_rows_scanned >= 5_000_000:
                print(f"build_offers_text: scanned {raw_rows_scanned:,} rows, written {rows_written_to_db:,}", flush=True)
            if limit_rows and raw_rows_scanned >= limit_rows:
                print(f"build_offers_text: hit limit {limit_rows:,}", flush=True)
                break

        if chunk_buf:
            merged = pd.concat(chunk_buf, ignore_index=True)
            before = len(merged)
            processed = _process_chunk(merged, array_cols, dedup_col)
            rows_all_text_null_dropped += before - len(processed) if not processed.empty else before
            if not processed.empty:
                if not schema_created:
                    processed.head(0).to_sql("offers_text", conn, if_exists="replace", index=False)
                    schema_created = True
                processed.to_sql("offers_text", conn, if_exists="append", index=False, method="multi", chunksize=50_000)
                rows_written_to_db += len(processed)
            del merged, processed
        chunk_buf = []

        if not schema_created or rows_written_to_db == 0:
            rows_emitted = 0
            n_unique_entity_id = 0
            n_unique_entity_day = 0
        else:
            print(f"build_offers_text: deduping in SQLite ({rows_written_to_db:,} rows)", flush=True)
            conn.execute(
                "CREATE TABLE offers_text_deduped AS SELECT * FROM offers_text WHERE rowid IN "
                "(SELECT MAX(rowid) FROM offers_text GROUP BY entity_id, snapshot_ts)"
            )
            conn.execute("DROP TABLE offers_text")
            conn.execute("ALTER TABLE offers_text_deduped RENAME TO offers_text")
            conn.commit()
            print(f"build_offers_text: exporting to parquet (streaming)", flush=True)
            import pyarrow as pa
            import pyarrow.parquet as pq
            writer = None
            for chunk in pd.read_sql_query("SELECT * FROM offers_text", conn, chunksize=500_000):
                tbl = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_path), tbl.schema)
                writer.write_table(tbl)
            if writer:
                writer.close()
            rows_emitted = int(pd.read_sql_query("SELECT COUNT(*) as n FROM offers_text", conn).iloc[0]["n"])
            n_unique_entity_id = int(pd.read_sql_query("SELECT COUNT(DISTINCT entity_id) as n FROM offers_text", conn).iloc[0]["n"])
            n_unique_entity_day = int(pd.read_sql_query("SELECT COUNT(DISTINCT date(snapshot_ts)) as n FROM offers_text", conn).iloc[0]["n"])
        conn.close()
    finally:
        Path(db_path).unlink(missing_ok=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows_emitted == 0:
        scalar_cols_out = ["headline", "title", "description_text", "company_description", "product_description",
                          "financial_condition", "financial_forecasts", "financial_risks", "offering_purpose",
                          "term_sheet", "front_video_transcript"] + [f"{c}_text" for c in array_cols]
        pd.DataFrame(columns=["entity_id", "platform_name", "offer_id", "snapshot_ts"] + scalar_cols_out).to_parquet(out_path, index=False)

    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "limit_rows": limit_rows,
        "raw_offers_version": delta_version,
        "raw_active_files": raw_active_files,
        "raw_rows_scanned": raw_rows_scanned,
        "rows_emitted": rows_emitted,
        "rows_all_text_null_dropped": rows_all_text_null_dropped,
        "n_unique_entity_id": n_unique_entity_id,
        "n_unique_entity_day": n_unique_entity_day,
        "script_sha256": script_sha,
        "output_path": str(out_path),
        "row_count": rows_emitted,
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({rows_emitted:,} rows)", flush=True)


if __name__ == "__main__":
    main()
