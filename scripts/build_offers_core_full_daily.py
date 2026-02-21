#!/usr/bin/env python
"""
Build offers_core_full_daily from offers_core_snapshot (wide).
Adds derived columns: delta_funding_raised, pct_change, count_snapshots_per_day.
For same granularity (entity-day), daily = snapshot + derived, using last_non_null.
If offers_extras_snapshot exists, build offers_extras_daily similarly.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offers_core_daily from snapshot (add derived cols).")
    parser.add_argument("--snapshot_dir", type=Path, required=True, help="Dir with offers_core_snapshot.parquet")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--extras_snapshot", type=Path, default=None, help="Optional offers_extras_snapshot.parquet")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "pandas", "duckdb"], help="Execution backend for daily build")
    parser.add_argument("--duckdb_tmp_dir", type=Path, default=None, help="Optional duckdb temp dir")
    args = parser.parse_args()

    snap_path = args.snapshot_dir / "offers_core_snapshot.parquet"
    static_path = args.snapshot_dir / "offers_static.parquet"
    out_daily = args.output_dir / "offers_core_daily.parquet"
    out_static = args.output_dir / "offers_static.parquet"
    extras_snap = args.extras_snapshot or (args.snapshot_dir / "offers_extras_snapshot.parquet")
    out_extras = args.output_dir / "offers_extras_daily.parquet"

    if not snap_path.exists():
        print(f"ERROR: {snap_path} not found. Run build_offers_core_full_snapshot first.", file=sys.stderr)
        sys.exit(1)
    if out_daily.exists() and args.overwrite != 1:
        print(f"ERROR: {out_daily} exists and --overwrite 0.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    snap_size_gb = snap_path.stat().st_size / (1024**3)
    use_duckdb = args.backend == "duckdb" or (args.backend == "auto" and snap_size_gb >= 2.0)

    if use_duckdb:
        import duckdb  # type: ignore
        import pyarrow.parquet as pq

        tmp_dir = args.duckdb_tmp_dir or (args.output_dir / "_duckdb_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.execute("SET temp_directory=?", [str(tmp_dir)])

        mem_limit_gb = os.getenv("DUCKDB_MEMORY_LIMIT_GB", "").strip()
        if mem_limit_gb:
            if mem_limit_gb.isdigit():
                con.execute("SET memory_limit=?", [f"{mem_limit_gb}GB"])
            else:
                con.execute("SET memory_limit=?", [mem_limit_gb])

        threads_env = os.getenv("DUCKDB_THREADS", "").strip()
        if threads_env:
            try:
                threads = int(threads_env)
            except ValueError:
                threads = 0
            if threads > 0:
                con.execute("SET threads=?", [threads])

        snap_schema = pq.ParquetFile(snap_path).schema.names
        has_snapshot_ts = "snapshot_ts" in snap_schema
        has_crawled_day = "crawled_date_day" in snap_schema
        has_crawled_date = "crawled_date" in snap_schema
        has_processed = "processed_datetime" in snap_schema

        if has_snapshot_ts:
            snapshot_ts_expr = "snapshot_ts"
        elif has_crawled_day:
            snapshot_ts_expr = "CAST(crawled_date_day AS TIMESTAMP)"
        elif has_crawled_date:
            snapshot_ts_expr = "crawled_date"
        else:
            snapshot_ts_expr = "NULL"

        if has_crawled_day:
            crawled_day_expr = "CAST(crawled_date_day AS DATE)"
        elif has_snapshot_ts:
            crawled_day_expr = "CAST(snapshot_ts AS DATE)"
        elif has_crawled_date:
            crawled_day_expr = "CAST(crawled_date AS DATE)"
        else:
            crawled_day_expr = "NULL"

        order_processed = "processed_datetime DESC NULLS LAST" if has_processed else "NULL"

        # Core daily
        core_sql = f"""
        WITH base AS (
            SELECT *,
                   {snapshot_ts_expr} AS snapshot_ts2,
                   {crawled_day_expr} AS crawled_date_day2
            FROM read_parquet('{snap_path.as_posix()}')
        ), ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY entity_id, crawled_date_day2
                       ORDER BY snapshot_ts2 DESC, {order_processed}
                   ) AS rn,
                   COUNT(*) OVER (PARTITION BY entity_id, crawled_date_day2) AS count_snapshots_per_day
            FROM base
        )
        SELECT
            *,
            (TRY_CAST(funding_raised_usd AS DOUBLE) - LAG(TRY_CAST(funding_raised_usd AS DOUBLE)) OVER (PARTITION BY entity_id ORDER BY crawled_date_day2))
                AS delta_funding_raised,
            (TRY_CAST(funding_raised_usd AS DOUBLE) - LAG(TRY_CAST(funding_raised_usd AS DOUBLE)) OVER (PARTITION BY entity_id ORDER BY crawled_date_day2))
                / NULLIF(LAG(TRY_CAST(funding_raised_usd AS DOUBLE)) OVER (PARTITION BY entity_id ORDER BY crawled_date_day2), 0)
                AS pct_change
        FROM ranked
        WHERE rn = 1
        """
        con.execute(
            f"COPY ({core_sql}) TO '{out_daily.as_posix()}' (FORMAT PARQUET)"
        )

        daily_schema = pq.ParquetFile(out_daily).schema.names
        static_cols = ["entity_id", "platform_name", "offer_id"]
        for c in ["cik", "datetime_open_offering", "datetime_close_offering"]:
            if c in daily_schema:
                static_cols.append(c)
        static_select = ", ".join(static_cols)

        # Static
        static_sql = f"""
        WITH base AS (
            SELECT *,
                   COALESCE(TRY_CAST(snapshot_ts AS TIMESTAMP),
                            TRY_CAST(crawled_date_day AS TIMESTAMP),
                            TRY_CAST(crawled_date AS TIMESTAMP)) AS snapshot_ts2
            FROM read_parquet('{out_daily.as_posix()}')
        ), ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY snapshot_ts2 DESC) AS rn
            FROM base
        )
        SELECT
            {static_select},
            snapshot_ts2 AS static_snapshot_ts
        FROM ranked
        WHERE rn = 1
        """
        con.execute(
            f"COPY ({static_sql}) TO '{out_static.as_posix()}' (FORMAT PARQUET)"
        )

        extras_output_columns = []
        if extras_snap.exists():
            ex_schema = pq.ParquetFile(extras_snap).schema.names
            ex_has_snapshot_ts = "snapshot_ts" in ex_schema
            ex_has_crawled_day = "crawled_date_day" in ex_schema
            ex_has_crawled_date = "crawled_date" in ex_schema
            ex_has_processed = "processed_datetime" in ex_schema

            if ex_has_snapshot_ts:
                ex_snapshot_ts_expr = "snapshot_ts"
            elif ex_has_crawled_day:
                ex_snapshot_ts_expr = "CAST(crawled_date_day AS TIMESTAMP)"
            elif ex_has_crawled_date:
                ex_snapshot_ts_expr = "crawled_date"
            else:
                ex_snapshot_ts_expr = "NULL"

            if ex_has_crawled_day:
                ex_crawled_day_expr = "CAST(crawled_date_day AS DATE)"
            elif ex_has_snapshot_ts:
                ex_crawled_day_expr = "CAST(snapshot_ts AS DATE)"
            elif ex_has_crawled_date:
                ex_crawled_day_expr = "CAST(crawled_date AS DATE)"
            else:
                ex_crawled_day_expr = "NULL"

            ex_order_processed = "processed_datetime DESC NULLS LAST" if ex_has_processed else "NULL"

            extras_sql = f"""
            WITH base AS (
                SELECT *,
                       {ex_snapshot_ts_expr} AS snapshot_ts2,
                       {ex_crawled_day_expr} AS crawled_date_day2
                FROM read_parquet('{extras_snap.as_posix()}')
            ), ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY entity_id, crawled_date_day2
                           ORDER BY snapshot_ts2 DESC, {ex_order_processed}
                       ) AS rn
                FROM base
            )
            SELECT * FROM ranked WHERE rn = 1
            """
            con.execute(
                f"COPY ({extras_sql}) TO '{out_extras.as_posix()}' (FORMAT PARQUET)"
            )
            extras_output_columns = pq.ParquetFile(out_extras).schema.names

        output_columns = pq.ParquetFile(out_daily).schema.names
        rows_emitted = pq.ParquetFile(out_daily).metadata.num_rows

        manifest_path = args.snapshot_dir / "MANIFEST.json"
        manifest: dict = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["output_type"] = "daily"
        manifest["snapshot_dir"] = str(args.snapshot_dir)
        manifest["rows_emitted"] = rows_emitted
        manifest["output_columns"] = output_columns
        if extras_output_columns:
            manifest["extras_output_columns"] = extras_output_columns
            manifest["extras_snapshot_path"] = str(extras_snap)
            manifest["extras_daily_path"] = str(out_extras)
        manifest["grain"] = "entity-day"
        manifest["partition_strategy"] = "single_file"
        manifest["cmd_args"] = sys.argv[1:]
        manifest["built_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        manifest["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root).stdout.strip() or "unknown"
        (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"Wrote {out_daily} ({rows_emitted:,} rows), {out_static}", flush=True)
        if extras_output_columns:
            print(f"Wrote {out_extras}", flush=True)
        con.close()
        return

    df = pd.read_parquet(snap_path)
    if "entity_id" not in df.columns:
        if "platform_name" in df.columns and "offer_id" in df.columns:
            df["entity_id"] = df["platform_name"].astype(str) + "|" + df["offer_id"].astype(str)
        else:
            print("ERROR: snapshot missing entity_id and platform_name/offer_id", file=sys.stderr)
            sys.exit(1)
    if "snapshot_ts" not in df.columns:
        if "crawled_date_day" in df.columns:
            df["snapshot_ts"] = pd.to_datetime(df["crawled_date_day"], errors="coerce", utc=True)
        elif "crawled_date" in df.columns:
            df["snapshot_ts"] = pd.to_datetime(df["crawled_date"], errors="coerce", utc=True)
        else:
            print("ERROR: snapshot missing snapshot_ts/crawled_date_day/crawled_date", file=sys.stderr)
            sys.exit(1)
    df = df.sort_values(["entity_id", "snapshot_ts"], kind="mergesort")
    if "crawled_date_day" not in df.columns:
        df["crawled_date_day"] = pd.to_datetime(df["snapshot_ts"], errors="coerce").dt.date.astype(str)
    key_cols = ["entity_id", "crawled_date_day"]
    counts = df.groupby(key_cols, sort=False).size().reset_index(name="count_snapshots_per_day")

    # Ensure last_non_null within entity-day while preserving key columns
    non_key_cols = [c for c in df.columns if c not in key_cols]
    if non_key_cols:
        df[non_key_cols] = df.groupby(key_cols, sort=False)[non_key_cols].ffill()
    df = df.groupby(key_cols, sort=False).tail(1)
    # Defragment before adding derived columns
    df = df.copy()

    # Derived: delta_funding_raised, pct_change (vs prev day)
    if "funding_raised_usd" in df.columns:
        base = pd.to_numeric(df["funding_raised_usd"], errors="coerce")
        prev = base.groupby(df["entity_id"]).shift(1)
        delta = base - prev
        pct = delta / prev.replace(0, float("nan"))
        df = df.assign(
            delta_funding_raised=delta,
            pct_change=pct,
        )
    df = df.merge(counts, on=key_cols, how="left")

    df.to_parquet(out_daily, index=False)
    output_columns = list(df.columns)
    if static_path.exists():
        import shutil
        shutil.copy(static_path, out_static)
    else:
        static = df.drop_duplicates(subset=["entity_id"], keep="last")[["entity_id", "platform_name", "offer_id"] + [c for c in ["cik", "datetime_open_offering", "datetime_close_offering"] if c in df.columns]]
        static["static_snapshot_ts"] = static["snapshot_ts"]
        static.to_parquet(out_static, index=False)

    extras_output_columns = []
    if extras_snap.exists():
        ex = pd.read_parquet(extras_snap)
        if "entity_id" not in ex.columns:
            if "platform_name" in ex.columns and "offer_id" in ex.columns:
                ex["entity_id"] = ex["platform_name"].astype(str) + "|" + ex["offer_id"].astype(str)
            else:
                print("ERROR: extras snapshot missing entity_id and platform_name/offer_id", file=sys.stderr)
                sys.exit(1)
        if "snapshot_ts" not in ex.columns:
            if "crawled_date_day" in ex.columns:
                ex["snapshot_ts"] = pd.to_datetime(ex["crawled_date_day"], errors="coerce", utc=True)
            elif "crawled_date" in ex.columns:
                ex["snapshot_ts"] = pd.to_datetime(ex["crawled_date"], errors="coerce", utc=True)
            else:
                print("ERROR: extras snapshot missing snapshot_ts/crawled_date_day/crawled_date", file=sys.stderr)
                sys.exit(1)
        ex = ex.sort_values(["entity_id", "snapshot_ts"], kind="mergesort")
        if "crawled_date_day" not in ex.columns:
            ex["crawled_date_day"] = pd.to_datetime(ex["snapshot_ts"], errors="coerce").dt.date.astype(str)
        ex_key_cols = ["entity_id", "crawled_date_day"]
        ex_non_key = [c for c in ex.columns if c not in ex_key_cols]
        if ex_non_key:
            ex[ex_non_key] = ex.groupby(ex_key_cols, sort=False)[ex_non_key].ffill()
        ex = ex.groupby(ex_key_cols, sort=False).tail(1)
        ex.to_parquet(out_extras, index=False)
        extras_output_columns = list(ex.columns)

    manifest_path = args.snapshot_dir / "MANIFEST.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_type"] = "daily"
    manifest["snapshot_dir"] = str(args.snapshot_dir)
    manifest["rows_emitted"] = len(df)
    manifest["output_columns"] = output_columns
    if extras_output_columns:
        manifest["extras_output_columns"] = extras_output_columns
        manifest["extras_snapshot_path"] = str(extras_snap)
        manifest["extras_daily_path"] = str(out_extras)
    manifest["grain"] = "entity-day"
    manifest["partition_strategy"] = "single_file"
    manifest["cmd_args"] = sys.argv[1:]
    manifest["built_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_root).stdout.strip() or "unknown"
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {out_daily} ({len(df):,} rows), {out_static}", flush=True)
    if extras_output_columns:
        print(f"Wrote {out_extras} ({len(ex):,} rows)", flush=True)


if __name__ == "__main__":
    main()
