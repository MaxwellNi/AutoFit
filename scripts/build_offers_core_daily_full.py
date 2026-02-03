#!/usr/bin/env python
"""
Build full-scale offers_core_snapshot from raw offers Delta (streaming, no OOM).
Wide contract:
- Keep almost all scalar columns (string/number/bool/timestamp) in core snapshot.
- Nested columns (list/map/struct/array) are not dropped; they are routed to
  derived columns: <col>__json, <col>__len, <col>__hash in offers_extras_snapshot.
Manifest records output_columns, dropped_columns, grain, partition strategy, cmd args.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

SNAPSHOT_COL = "crawled_date_day"
SNAPSHOT_FALLBACK_COLS = ["crawled_date", "snapshot_ts"]
CORE_COLS = [
    "platform_name", "offer_id", "crawled_date_day",
    "funding_goal_usd", "funding_raised_usd", "investors_count", "is_funded",
    "cik", "link", "hash_id",
    "datetime_open_offering", "datetime_close_offering",
]
DEDUP_COL = "processed_datetime"


def _load_contract(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _is_nested_type(type_str: str) -> bool:
    tl = str(type_str).lower()
    return "list" in tl or "struct" in tl or "map" in tl or "array" in tl


def _nested_cols_from_schema(schema_types: Dict[str, str]) -> List[str]:
    return [c for c, t in schema_types.items() if _is_nested_type(t)]


def _derived_nested_cols(nested_cols: List[str]) -> List[str]:
    out = []
    for c in nested_cols:
        out.extend([f"{c}__json", f"{c}__len", f"{c}__hash"])
    return out


def _stable_json(val: Any) -> Optional[str]:
    if val is None:
        return None
    try:
        return json.dumps(val, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return json.dumps(str(val), ensure_ascii=True)


def _value_len(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (list, tuple, dict)):
        return len(val)
    return 1


def _value_hash(js: Optional[str]) -> Optional[str]:
    if js is None:
        return None
    return hashlib.sha1(js.encode("utf-8")).hexdigest()


def _align_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def _arrow_type_for(type_str: str, override: Optional[str] = None):
    import pyarrow as pa
    if override == "string":
        return pa.string()
    if override == "float":
        return pa.float64()
    tl = str(type_str).lower()
    if "int" in tl or "float" in tl or "double" in tl or "decimal" in tl:
        return pa.float64()
    if "bool" in tl:
        return pa.float64()
    if "timestamp" in tl or "date" in tl:
        return pa.string()
    return pa.string()


def _arrow_schema_for(cols: List[str], schema_types: Dict[str, str], overrides: Dict[str, str]):
    import pyarrow as pa
    fields = [pa.field(c, _arrow_type_for(schema_types.get(c, ""), overrides.get(c))) for c in cols]
    return pa.schema(fields)


def _coerce_chunk_types(df: pd.DataFrame, schema_types: Dict[str, str], overrides: Dict[str, str]) -> pd.DataFrame:
    for col in df.columns:
        ov = overrides.get(col)
        if ov == "string":
            df[col] = df[col].astype("string")
            continue
        if ov == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            continue
        tl = str(schema_types.get(col, "")).lower()
        if "int" in tl or "float" in tl or "double" in tl or "decimal" in tl:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif "bool" in tl:
            df[col] = df[col].map({True: 1, False: 0})
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif "timestamp" in tl or "date" in tl:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        else:
            df[col] = df[col].astype("string")
    return df


def _nested_presence_mask(batch, nested_cols: List[str], schema_types: Dict[str, str]) -> Optional[List[bool]]:
    if not nested_cols:
        return None
    import pyarrow as pa
    import pyarrow.compute as pc
    mask = None
    for col in nested_cols:
        try:
            idx = batch.schema.get_field_index(col)
            if idx < 0:
                continue
            arr = batch.column(idx)
        except Exception:
            continue
        if isinstance(arr, pa.ChunkedArray):
            try:
                arr = arr.combine_chunks()
            except Exception:
                pass
        t = schema_types.get(col, "")
        if "list" in t.lower() or "array" in t.lower():
            try:
                lengths = pc.list_value_length(arr)
                present = pc.greater(lengths, 0)
            except Exception:
                present = pc.invert(pc.is_null(arr))
        else:
            present = pc.invert(pc.is_null(arr))
        present = pc.fill_null(present, False)
        if isinstance(present, pa.ChunkedArray):
            try:
                present = present.combine_chunks()
            except Exception:
                pass
        mask = present if mask is None else pc.or_(mask, present)
    if mask is None:
        return None
    return mask.to_numpy(zero_copy_only=False).tolist()


def _build_nested_df(
    batch,
    nested_cols: List[str],
    schema_types: Dict[str, str],
    row_idx: List[int],
) -> pd.DataFrame:
    if not nested_cols or not row_idx:
        return pd.DataFrame()
    import pyarrow as pa
    data: Dict[str, List[Any]] = {}
    idx_list = list(row_idx)
    for col in nested_cols:
        try:
            idx = batch.schema.get_field_index(col)
            if idx < 0:
                continue
            arr = batch.column(idx)
            if isinstance(arr, pa.ChunkedArray):
                try:
                    arr = arr.combine_chunks()
                except Exception:
                    pass
            arr = arr.to_pylist()
        except Exception:
            continue
        json_vals: List[Optional[str]] = []
        len_vals: List[Optional[int]] = []
        hash_vals: List[Optional[str]] = []
        for i in idx_list:
            val = arr[i] if i < len(arr) else None
            if val is None:
                json_vals.append(None)
                len_vals.append(None)
                hash_vals.append(None)
                continue
            js = _stable_json(val)
            json_vals.append(js)
            len_vals.append(_value_len(val))
            hash_vals.append(_value_hash(js))
        data[f"{col}__json"] = json_vals
        data[f"{col}__len"] = len_vals
        data[f"{col}__hash"] = hash_vals
    return pd.DataFrame(data)


def _write_deduped_parquet(
    conn: sqlite3.Connection,
    table: str,
    out_path: Path,
    out_cols: List[str],
    dedup_col: str,
    static_cols: List[str],
    schema_types: Dict[str, str],
    overrides: Dict[str, str],
    output_schema=None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    query = (
        "SELECT " + ", ".join([f'"{c}"' for c in out_cols]) + " FROM ("
        "SELECT *, ROW_NUMBER() OVER (PARTITION BY entity_id, snapshot_ts ORDER BY "
        f'"{dedup_col}" DESC' + ") AS rn FROM " + table + ") WHERE rn=1"
    )
    writer = None
    rows_emitted = 0
    n_entities = 0
    date_min = None
    date_max = None
    entities_seen: set = set()
    static_latest: Dict[str, Dict[str, Any]] = {}

    for chunk in pd.read_sql_query(query, conn, chunksize=500_000):
        if chunk.empty:
            continue
        rows_emitted += len(chunk)
        if "entity_id" in chunk.columns:
            entities_seen.update(chunk["entity_id"].astype(str).tolist())
        if "snapshot_ts" in chunk.columns:
            snap = pd.to_datetime(chunk["snapshot_ts"], errors="coerce")
            if not snap.empty:
                dmin = str(snap.min().date()) if snap.notna().any() else None
                dmax = str(snap.max().date()) if snap.notna().any() else None
                date_min = dmin if date_min is None or (dmin and dmin < date_min) else date_min
                date_max = dmax if date_max is None or (dmax and dmax > date_max) else date_max
        if static_cols and "entity_id" in chunk.columns:
            sub = chunk[[c for c in (static_cols + ["snapshot_ts"]) if c in chunk.columns]].copy()
            if "snapshot_ts" in sub.columns:
                sub["_ts"] = pd.to_datetime(sub["snapshot_ts"], errors="coerce")
                sub = sub.sort_values("_ts")
                latest = sub.groupby("entity_id", as_index=False).tail(1)
                for row in latest.itertuples(index=False):
                    ent = str(getattr(row, "entity_id"))
                    ts = getattr(row, "_ts", None)
                    prev = static_latest.get(ent)
                    if prev is None or (ts is not None and prev.get("_ts") is not None and ts > prev["_ts"]) or (prev.get("_ts") is None and ts is not None):
                        static_latest[ent] = row._asdict()
        chunk = _coerce_chunk_types(chunk, schema_types, overrides)
        if output_schema is not None:
            tbl = pa.Table.from_pandas(chunk[out_cols], preserve_index=False, schema=output_schema, safe=False)
        else:
            tbl = pa.Table.from_pandas(chunk[out_cols], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), output_schema or tbl.schema)
        if output_schema is not None and tbl.schema != output_schema:
            try:
                tbl = tbl.cast(output_schema, safe=False)
            except Exception:
                pass
        writer.write_table(tbl)
    if writer:
        writer.close()

    n_entities = len(entities_seen)
    static_rows = []
    if static_latest:
        for ent, row in static_latest.items():
            row.pop("_ts", None)
            static_rows.append(row)
    return {
        "rows_emitted": rows_emitted,
        "n_unique_entities": n_entities,
        "date_min": date_min,
        "date_max": date_max,
    }, static_rows


def _write_part_parquet(
    df: pd.DataFrame,
    out_dir: Path,
    prefix: str,
    part_idx: int,
    schema,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_part_{part_idx:06d}.parquet"
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, out_path)
    return part_idx + 1


def _duckdb_dedup_to_parquet(
    core_glob: str,
    out_core: Path,
    core_cols: List[str],
    dedup_col: str,
    temp_dir: Optional[Path],
    out_static: Path,
    static_cols: List[str],
    extras_glob: Optional[str] = None,
    out_extras: Optional[Path] = None,
    extras_cols: Optional[List[str]] = None,
    extras_dedup_col: Optional[str] = None,
) -> Dict[str, Any]:
    import duckdb  # type: ignore

    con = duckdb.connect()
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute("SET temp_directory=?", [str(temp_dir)])

    core_select = ", ".join([f'"{c}"' for c in core_cols])
    dedup_col = dedup_col or "snapshot_ts"
    con.execute(
        f"""
        COPY (
            SELECT {core_select} FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY entity_id, snapshot_ts ORDER BY \"{dedup_col}\" DESC
                ) AS rn
                FROM read_parquet('{core_glob}')
            ) WHERE rn=1
        ) TO '{out_core.as_posix()}' (FORMAT PARQUET)
        """
    )

    stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS rows_emitted,
            COUNT(DISTINCT entity_id) AS n_unique_entities,
            MIN(snapshot_ts) AS date_min,
            MAX(snapshot_ts) AS date_max
        FROM read_parquet('{out_core.as_posix()}')
        """
    ).fetchone()
    rows_emitted, n_entities, date_min, date_max = stats

    static_select_cols = [c for c in static_cols if c in core_cols]
    static_select = ", ".join([f'"{c}"' for c in static_select_cols])
    if static_select:
        static_select = static_select + ", "
    con.execute(
        f"""
        COPY (
            SELECT {static_select}snapshot_ts AS static_snapshot_ts FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY entity_id ORDER BY snapshot_ts DESC
                ) AS rn
                FROM read_parquet('{out_core.as_posix()}')
            ) WHERE rn=1
        ) TO '{out_static.as_posix()}' (FORMAT PARQUET)
        """
    )

    extras_output_columns: List[str] = []
    if extras_glob and out_extras and extras_cols:
        extras_select = ", ".join([f'"{c}"' for c in extras_cols])
        dedup_ex = extras_dedup_col or "snapshot_ts"
        con.execute(
            f"""
            COPY (
                SELECT {extras_select} FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY entity_id, snapshot_ts ORDER BY \"{dedup_ex}\" DESC
                    ) AS rn
                    FROM read_parquet('{extras_glob}')
                ) WHERE rn=1
            ) TO '{out_extras.as_posix()}' (FORMAT PARQUET)
            """
        )
        extras_output_columns = extras_cols

    con.close()
    return {
        "rows_emitted": rows_emitted,
        "n_unique_entities": n_entities,
        "date_min": date_min,
        "date_max": date_max,
        "extras_output_columns": extras_output_columns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full offers_core_snapshot (wide, streaming).")
    parser.add_argument("--raw_offers_delta", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=None, help="column_contract_wide.yaml")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", type=int, default=0)
    parser.add_argument("--chunk_rows", type=int, default=None)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--output_base", type=str, default="offers_core_daily", help="Base name: offers_core_snapshot or offers_core_daily")
    parser.add_argument("--extras_output_base", type=str, default=None, help="Base name for nested-derived table")
    parser.add_argument("--sqlite_dir", type=Path, default=None, help="Directory to place sqlite temp DB (avoid /tmp full)")
    parser.add_argument("--sqlite_path", type=Path, default=None, help="Explicit sqlite DB path (overrides sqlite_dir)")
    parser.add_argument("--min_free_gb", type=float, default=20.0, help="Min free GB required (fallback for sqlite/output dirs)")
    parser.add_argument("--min_free_gb_sqlite", type=float, default=None, help="Min free GB required for sqlite_dir (overrides --min_free_gb)")
    parser.add_argument("--min_free_gb_output", type=float, default=None, help="Min free GB required for output_dir (overrides --min_free_gb)")
    parser.add_argument("--oom_guard_pct", type=float, default=0.95, help="Flush buffers if RSS exceeds this fraction of total RAM (0 to disable)")
    parser.add_argument("--oom_guard_free_gb", type=float, default=4.0, help="Flush buffers if available RAM falls below this GB (0 to disable)")
    parser.add_argument("--oom_guard_check_every", type=int, default=25, help="Check OOM guard every N batches")
    parser.add_argument("--dedup_backend", type=str, default="sqlite", choices=["sqlite", "duckdb"], help="Backend for dedup/write (sqlite or duckdb)")
    args = parser.parse_args()

    contract = _load_contract(args.contract)
    snap_cfg = contract.get("offers_core_snapshot", {}) if contract else {}
    must = snap_cfg.get("must_keep", [])
    should = snap_cfg.get("should_add", [])
    can_drop = set(snap_cfg.get("can_drop", []))
    derived_nested = snap_cfg.get("derived_nested", [])
    nested_cols_contract = snap_cfg.get("nested_columns", [])
    core_cols = list(dict.fromkeys(CORE_COLS + [c for c in must + should if c not in can_drop]))

    out_core = args.output_dir / f"{args.output_base}.parquet"
    out_static = args.output_dir / "offers_static.parquet"
    extras_base = args.extras_output_base or ("offers_extras_snapshot" if "snapshot" in args.output_base else "offers_extras_daily")
    out_extras = args.output_dir / f"{extras_base}.parquet"

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
    schema_types = {f.name: str(f.type) for f in dt.schema().fields}

    nested_cols = nested_cols_contract or _nested_cols_from_schema(schema_types)
    derived_nested_cols = derived_nested or _derived_nested_cols(nested_cols)
    scalar_cols = [c for c in core_cols if c in schema_names and c not in nested_cols]

    missing_in_raw = [c for c in core_cols if c not in schema_names]
    dropped_columns = [{"column": c, "reason": "missing_in_raw"} for c in missing_in_raw]
    if nested_cols:
        dropped_columns.extend([{"column": c, "reason": "nested_routed_to_extras"} for c in nested_cols])

    read_cols = list(dict.fromkeys(
        [c for c in scalar_cols if c in schema_names] +
        [c for c in nested_cols if c in schema_names] +
        [DEDUP_COL] +
        ["platform_name", "offer_id", SNAPSHOT_COL] +
        [c for c in SNAPSHOT_FALLBACK_COLS if c in schema_names]
    ))
    if DEDUP_COL not in read_cols and DEDUP_COL in schema_names:
        read_cols.append(DEDUP_COL)
    if SNAPSHOT_COL not in read_cols:
        read_cols.append(SNAPSHOT_COL)

    core_out_cols = ["entity_id", "platform_name", "offer_id", "snapshot_ts", SNAPSHOT_COL]
    core_out_cols += [c for c in scalar_cols if c in schema_names and c not in core_out_cols]
    core_out_cols = list(dict.fromkeys(core_out_cols))
    extras_out_cols = ["entity_id", "platform_name", "offer_id", "snapshot_ts", SNAPSHOT_COL] + derived_nested_cols
    extras_out_cols = list(dict.fromkeys([c for c in extras_out_cols if c != DEDUP_COL]))
    core_stage_cols = list(dict.fromkeys(core_out_cols + ([DEDUP_COL] if DEDUP_COL in schema_names else [])))
    extras_stage_cols = list(dict.fromkeys(extras_out_cols + ([DEDUP_COL] if DEDUP_COL in schema_names else [])))
    core_sql_cols = list(dict.fromkeys(core_stage_cols))
    extras_sql_cols = list(dict.fromkeys(extras_stage_cols))
    overrides: Dict[str, str] = {}
    for c in core_out_cols + extras_out_cols:
        if c in ("entity_id", "platform_name", "offer_id", "snapshot_ts", SNAPSHOT_COL):
            overrides[c] = "string"
        elif c.endswith("__json") or c.endswith("__hash"):
            overrides[c] = "string"
        elif c.endswith("__len"):
            overrides[c] = "float"
        elif "date" in c or "datetime" in c or "timestamp" in c:
            overrides[c] = "string"
    if DEDUP_COL in schema_names:
        overrides[DEDUP_COL] = "string"
    core_schema = _arrow_schema_for(core_out_cols, schema_types, overrides)
    extras_schema = _arrow_schema_for(extras_out_cols, schema_types, overrides)
    core_stage_schema = _arrow_schema_for(core_stage_cols, schema_types, overrides)
    extras_stage_schema = _arrow_schema_for(extras_stage_cols, schema_types, overrides)

    chunk_rows = args.chunk_rows
    if chunk_rows is None:
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024**3)
        except Exception:
            avail_gb = 16.0
        # Conservative estimate for wide rows; keep buffer small to avoid OOM
        row_bytes_est = max(1500, 48 * len(core_out_cols))
        target_gb = max(2.0, min(32.0, (avail_gb - 6) * 0.25))
        chunk_rows = int(target_gb * (1024**3) / row_bytes_est)
        chunk_rows = min(2_000_000, max(200_000, chunk_rows))
    print(
        f"build_offers_core_snapshot: estimated_row_bytes={int(row_bytes_est) if 'row_bytes_est' in locals() else 'n/a'} "
        f"target_buffer_gb={target_gb if 'target_gb' in locals() else 'n/a'}",
        flush=True,
    )

    print(f"build_offers_core_snapshot: delta={raw_path} version={delta_version} active_files={active_files}", flush=True)
    print(f"build_offers_core_snapshot: snapshot_col={SNAPSHOT_COL} chunk_rows={chunk_rows:,} output={args.output_dir}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    min_free_output = args.min_free_gb_output if args.min_free_gb_output is not None else args.min_free_gb
    if min_free_output and min_free_output > 0:
        try:
            free_gb_out = shutil.disk_usage(args.output_dir).free / (1024**3)
            if free_gb_out < min_free_output:
                print(
                    f"ERROR: output_dir free space {free_gb_out:.1f} GB < min_free_gb_output {min_free_output}. "
                    f"Set --min_free_gb_output lower or choose a larger disk.",
                    file=sys.stderr,
                )
                sys.exit(2)
        except Exception:
            pass
    import pyarrow.parquet as pq
    limit_rows = args.limit_rows

    dedup_backend = args.dedup_backend.lower()
    use_duckdb = dedup_backend == "duckdb"
    core_stage_dir = args.output_dir / "_staging_core" if use_duckdb else None
    extras_stage_dir = args.output_dir / "_staging_extras" if use_duckdb else None
    core_part_idx = 0
    extras_part_idx = 0

    # Choose sqlite/duckdb temp location (avoid /tmp filling up)
    min_free_sqlite = args.min_free_gb_sqlite if args.min_free_gb_sqlite is not None else args.min_free_gb
    if use_duckdb:
        sqlite_dir = args.sqlite_dir or (args.output_dir / "_duckdb_tmp")
        if args.overwrite == 1:
            shutil.rmtree(core_stage_dir, ignore_errors=True)
            shutil.rmtree(extras_stage_dir, ignore_errors=True)
        if min_free_sqlite and min_free_sqlite > 0:
            try:
                free_gb = shutil.disk_usage(sqlite_dir).free / (1024**3)
                if free_gb < min_free_sqlite:
                    print(
                        f"ERROR: temp dir free space {free_gb:.1f} GB < min_free_gb_sqlite {min_free_sqlite}. "
                        f"Use --sqlite_dir to a larger disk.",
                        file=sys.stderr,
                    )
                    sys.exit(2)
            except Exception:
                pass
        db_path = None
    else:
        if args.sqlite_path is not None:
            db_path = str(args.sqlite_path)
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            sqlite_dir = args.sqlite_dir or (args.output_dir / "_sqlite")
            sqlite_dir.mkdir(parents=True, exist_ok=True)
            if min_free_sqlite and min_free_sqlite > 0:
                try:
                    free_gb = shutil.disk_usage(sqlite_dir).free / (1024**3)
                    if free_gb < min_free_sqlite:
                        print(
                            f"ERROR: sqlite_dir free space {free_gb:.1f} GB < min_free_gb_sqlite {min_free_sqlite}. "
                            f"Use --sqlite_dir to a larger disk.",
                            file=sys.stderr,
                        )
                        sys.exit(2)
                except Exception:
                    pass
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir=str(sqlite_dir)) as tf:
                db_path = tf.name
    try:
        proc = None
        try:
            import psutil  # type: ignore
            if (args.oom_guard_pct and args.oom_guard_pct > 0) or (args.oom_guard_free_gb and args.oom_guard_free_gb > 0):
                proc = psutil.Process()
        except Exception:
            proc = None
        conn = None
        if not use_duckdb:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=OFF;")
            conn.execute("PRAGMA synchronous=OFF;")
            conn.execute("PRAGMA temp_store=MEMORY;")
        raw_rows_scanned = 0
        rows_written = 0
        rows_written_extras = 0
        static_rows: List[Dict[str, Any]] = []
        chunk_buf: List[pd.DataFrame] = []
        chunk_buf_extras: List[pd.DataFrame] = []
        buf_rows = 0
        buf_rows_extras = 0
        core_schema_created = False
        extras_schema_created = False

        batch_idx = 0
        for f in files_list:
            fpath = str(f)
            if fpath.startswith("file://"):
                fpath = fpath.replace("file://", "", 1)
            try:
                pf = pq.ParquetFile(fpath)
            except Exception:
                continue
            file_cols = [c for c in read_cols if c in pf.schema.names]
            if not file_cols:
                continue
            for rg in range(pf.num_row_groups):
                if limit_rows and raw_rows_scanned >= limit_rows:
                    break
                try:
                    table = pf.read_row_group(rg, columns=file_cols)
                except Exception:
                    continue
                if table.num_rows == 0:
                    continue
                raw_rows_scanned += table.num_rows
                batch_idx += 1
                scalar_cols_present = [c for c in table.column_names if c not in nested_cols]
                if not scalar_cols_present:
                    continue
                df = table.select(scalar_cols_present).to_pandas()
                if df.empty:
                    continue
                df["_row_idx"] = range(len(df))
                if "platform_name" not in df.columns or "offer_id" not in df.columns:
                    print("ERROR: platform_name/offer_id missing in raw schema", file=sys.stderr)
                    sys.exit(2)
                df["entity_id"] = df["platform_name"].astype(str) + "|" + df["offer_id"].astype(str)
                # Robust snapshot time handling (schema evolution)
                if "snapshot_ts" in df.columns:
                    snap_src = df["snapshot_ts"]
                elif SNAPSHOT_COL in df.columns:
                    snap_src = df[SNAPSHOT_COL]
                elif "crawled_date" in df.columns:
                    snap_src = df["crawled_date"]
                else:
                    # No usable time column in this batch
                    continue
                df["snapshot_ts"] = pd.to_datetime(snap_src, errors="coerce", utc=True)
                if SNAPSHOT_COL not in df.columns:
                    df[SNAPSHOT_COL] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True).dt.date.astype(str)
                df = df.dropna(subset=["entity_id", "snapshot_ts"])
                if df.empty:
                    continue

                # Core buffer (scalar columns only)
                core_keep = ["entity_id", "platform_name", "offer_id", "snapshot_ts", SNAPSHOT_COL]
                core_keep += [c for c in scalar_cols if c in df.columns and c not in core_keep]
                if DEDUP_COL in df.columns and DEDUP_COL not in core_keep:
                    core_keep.append(DEDUP_COL)
                core_df = df[core_keep].copy()
                core_df = _align_columns(core_df, core_sql_cols)
                chunk_buf.append(core_df)
                buf_rows += len(core_df)

                # Extras buffer (nested derived)
                if nested_cols:
                    mask = _nested_presence_mask(table, nested_cols, schema_types)
                    if mask is not None:
                        valid_idx = df["_row_idx"].astype(int).tolist()
                        include_idx = [idx for idx in valid_idx if idx < len(mask) and mask[idx]]
                        if include_idx:
                            nested_df = _build_nested_df(table, nested_cols, schema_types, include_idx)
                            if not nested_df.empty:
                                keys_df = df.set_index("_row_idx").loc[include_idx, ["entity_id", "platform_name", "offer_id", "snapshot_ts", SNAPSHOT_COL]].reset_index(drop=True)
                                extras_df = pd.concat([keys_df.reset_index(drop=True), nested_df.reset_index(drop=True)], axis=1)
                                if DEDUP_COL in df.columns and DEDUP_COL not in extras_df.columns:
                                    extras_df[DEDUP_COL] = df.set_index("_row_idx").loc[include_idx, DEDUP_COL].values
                                extras_df = _align_columns(extras_df, extras_sql_cols)
                                chunk_buf_extras.append(extras_df)
                                buf_rows_extras += len(extras_df)

                force_flush = False
                if proc and (batch_idx % max(1, args.oom_guard_check_every) == 0):
                    try:
                        mem = psutil.virtual_memory()  # type: ignore
                        rss_gb = proc.memory_info().rss / (1024**3)
                        avail_gb = mem.available / (1024**3)
                        if args.oom_guard_pct and (rss_gb / max(1e-9, mem.total / (1024**3))) >= args.oom_guard_pct:
                            force_flush = True
                        if args.oom_guard_free_gb and avail_gb <= args.oom_guard_free_gb:
                            force_flush = True
                        if force_flush:
                            print(
                                f"OOM guard: rss_gb={rss_gb:.2f} avail_gb={avail_gb:.2f} -> forcing flush",
                                flush=True,
                            )
                    except Exception:
                        pass

                if buf_rows >= chunk_rows or (limit_rows and raw_rows_scanned >= limit_rows) or force_flush:
                    merged = pd.concat(chunk_buf, ignore_index=True)
                    chunk_buf, buf_rows = [], 0
                    merged = _align_columns(merged, core_sql_cols)
                    if limit_rows and len(merged) > limit_rows:
                        merged = merged.head(limit_rows)
                    dedup_col = DEDUP_COL if DEDUP_COL in merged.columns else "snapshot_ts"
                    merged = merged.sort_values(["entity_id", "snapshot_ts", dedup_col], kind="mergesort")
                    merged = merged.drop_duplicates(subset=["entity_id", "snapshot_ts"], keep="last")
                    if not merged.empty:
                        if use_duckdb:
                            core_part_idx = _write_part_parquet(merged, core_stage_dir, "core", core_part_idx, core_stage_schema)
                            core_schema_created = True
                        else:
                            if not core_schema_created:
                                pd.DataFrame(columns=core_sql_cols).to_sql("core", conn, if_exists="replace", index=False)
                                core_schema_created = True
                            for j in range(0, len(merged), 1_000_000):
                                merged.iloc[j:j+1_000_000].to_sql("core", conn, if_exists="append", index=False, method="multi", chunksize=30)
                        rows_written += len(merged)
                    del merged

                    if chunk_buf_extras:
                        merged_ex = pd.concat(chunk_buf_extras, ignore_index=True)
                        chunk_buf_extras, buf_rows_extras = [], 0
                        merged_ex = _align_columns(merged_ex, extras_sql_cols)
                        if not merged_ex.empty:
                            dedup_col = DEDUP_COL if DEDUP_COL in merged_ex.columns else "snapshot_ts"
                            merged_ex = merged_ex.sort_values(["entity_id", "snapshot_ts", dedup_col], kind="mergesort")
                            merged_ex = merged_ex.drop_duplicates(subset=["entity_id", "snapshot_ts"], keep="last")
                            if use_duckdb:
                                extras_part_idx = _write_part_parquet(merged_ex, extras_stage_dir, "extras", extras_part_idx, extras_stage_schema)
                                extras_schema_created = True
                            else:
                                if not extras_schema_created:
                                    pd.DataFrame(columns=extras_sql_cols).to_sql("extras", conn, if_exists="replace", index=False)
                                    extras_schema_created = True
                                for j in range(0, len(merged_ex), 1_000_000):
                                    merged_ex.iloc[j:j+1_000_000].to_sql("extras", conn, if_exists="append", index=False, method="multi", chunksize=30)
                            rows_written_extras += len(merged_ex)
                        del merged_ex

            if (batch_idx) % 250 == 0 or raw_rows_scanned >= 5_000_000:
                print(f"build_offers_core_snapshot: scanned {raw_rows_scanned:,} rows, written {rows_written:,}", flush=True)
            if limit_rows and raw_rows_scanned >= limit_rows:
                break

        if chunk_buf:
            merged = pd.concat(chunk_buf, ignore_index=True)
            chunk_buf = []
            merged = _align_columns(merged, core_sql_cols)
            dedup_col = DEDUP_COL if DEDUP_COL in merged.columns else "snapshot_ts"
            merged = merged.sort_values(["entity_id", "snapshot_ts", dedup_col], kind="mergesort")
            merged = merged.drop_duplicates(subset=["entity_id", "snapshot_ts"], keep="last")
            if not merged.empty:
                if use_duckdb:
                    core_part_idx = _write_part_parquet(merged, core_stage_dir, "core", core_part_idx, core_stage_schema)
                    core_schema_created = True
                else:
                    if not core_schema_created:
                        pd.DataFrame(columns=core_sql_cols).to_sql("core", conn, if_exists="replace", index=False)
                        core_schema_created = True
                    for j in range(0, len(merged), 1_000_000):
                        merged.iloc[j:j+1_000_000].to_sql("core", conn, if_exists="append", index=False, method="multi", chunksize=30)
                rows_written += len(merged)
            del merged
        if chunk_buf_extras:
            merged_ex = pd.concat(chunk_buf_extras, ignore_index=True)
            chunk_buf_extras = []
            merged_ex = _align_columns(merged_ex, extras_sql_cols)
            dedup_col = DEDUP_COL if DEDUP_COL in merged_ex.columns else "snapshot_ts"
            merged_ex = merged_ex.sort_values(["entity_id", "snapshot_ts", dedup_col], kind="mergesort")
            merged_ex = merged_ex.drop_duplicates(subset=["entity_id", "snapshot_ts"], keep="last")
            if not merged_ex.empty:
                if use_duckdb:
                    extras_part_idx = _write_part_parquet(merged_ex, extras_stage_dir, "extras", extras_part_idx, extras_stage_schema)
                    extras_schema_created = True
                else:
                    if not extras_schema_created:
                        pd.DataFrame(columns=extras_sql_cols).to_sql("extras", conn, if_exists="replace", index=False)
                        extras_schema_created = True
                    for j in range(0, len(merged_ex), 1_000_000):
                        merged_ex.iloc[j:j+1_000_000].to_sql("extras", conn, if_exists="append", index=False, method="multi", chunksize=30)
                rows_written_extras += len(merged_ex)
            del merged_ex

        if not core_schema_created or rows_written == 0:
            rows_emitted = 0
            n_entities = 0
            date_min = date_max = None
            output_columns = []
            extras_output_columns = []
            static_rows = []
        else:
            if use_duckdb:
                core_glob = (core_stage_dir / "core_part_*.parquet").as_posix()
                extras_glob = (extras_stage_dir / "extras_part_*.parquet").as_posix() if extras_schema_created else None
                dedup_col = DEDUP_COL if DEDUP_COL in core_stage_cols else "snapshot_ts"
                extras_dedup = DEDUP_COL if DEDUP_COL in extras_stage_cols else "snapshot_ts"
                stats = _duckdb_dedup_to_parquet(
                    core_glob=core_glob,
                    out_core=out_core,
                    core_cols=core_out_cols,
                    dedup_col=dedup_col,
                    temp_dir=sqlite_dir if use_duckdb else None,
                    out_static=out_static,
                    static_cols=["entity_id", "platform_name", "offer_id", "cik", "link", "hash_id", "datetime_open_offering", "datetime_close_offering"],
                    extras_glob=extras_glob,
                    out_extras=out_extras if extras_schema_created and rows_written_extras > 0 else None,
                    extras_cols=extras_out_cols if extras_schema_created and rows_written_extras > 0 else None,
                    extras_dedup_col=extras_dedup,
                )
                rows_emitted = stats["rows_emitted"]
                n_entities = stats["n_unique_entities"]
                date_min = stats["date_min"]
                date_max = stats["date_max"]
                output_columns = core_out_cols
                extras_output_columns = stats.get("extras_output_columns", [])
                if not extras_output_columns and nested_cols:
                    empty_extras = pd.DataFrame(columns=extras_out_cols)
                    empty_extras.to_parquet(out_extras, index=False)
                    extras_output_columns = extras_out_cols
                shutil.rmtree(core_stage_dir, ignore_errors=True)
                shutil.rmtree(extras_stage_dir, ignore_errors=True)
            else:
                core_table_cols = [r[1] for r in conn.execute("PRAGMA table_info(core)").fetchall()]
                dedup_col = DEDUP_COL if DEDUP_COL in core_table_cols else "snapshot_ts"
                stats, static_rows = _write_deduped_parquet(
                    conn,
                    "core",
                    out_core,
                    core_out_cols,
                    dedup_col,
                    ["entity_id", "platform_name", "offer_id", "cik", "link", "hash_id", "datetime_open_offering", "datetime_close_offering"],
                    schema_types,
                    overrides,
                    output_schema=core_schema,
                )
                rows_emitted = stats["rows_emitted"]
                n_entities = stats["n_unique_entities"]
                date_min = stats["date_min"]
                date_max = stats["date_max"]
                output_columns = core_out_cols
                extras_output_columns = []
                if extras_schema_created and rows_written_extras > 0:
                    extras_table_cols = [r[1] for r in conn.execute("PRAGMA table_info(extras)").fetchall()]
                    extras_cols = [c for c in extras_out_cols if c in extras_table_cols]
                    dedup_col_ex = DEDUP_COL if DEDUP_COL in extras_table_cols else "snapshot_ts"
                    _write_deduped_parquet(
                        conn,
                        "extras",
                        out_extras,
                        extras_cols,
                        dedup_col_ex,
                        [],
                        schema_types,
                        overrides,
                        output_schema=extras_schema,
                    )
                    extras_output_columns = extras_cols
                elif nested_cols:
                    # Ensure extras parquet exists (schema-only) even when no nested rows were present
                    empty_extras = pd.DataFrame(columns=extras_out_cols)
                    empty_extras.to_parquet(out_extras, index=False)
                    extras_output_columns = extras_out_cols

        if static_rows and not use_duckdb:
            static_df = pd.DataFrame(static_rows)
            static_df["static_snapshot_ts"] = pd.to_datetime(static_df["snapshot_ts"], errors="coerce", utc=True)
            keep = ["entity_id", "platform_name", "offer_id", "static_snapshot_ts"]
            keep += [c for c in ["cik", "link", "hash_id", "datetime_open_offering", "datetime_close_offering"] if c in static_df.columns]
            static_df = static_df[[c for c in keep if c in static_df.columns]]
            static_df.to_parquet(out_static, index=False)
        if conn is not None:
            conn.close()
    finally:
        if db_path:
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
        "output_columns": output_columns,
        "extras_output_columns": extras_output_columns,
        "nested_columns": nested_cols,
        "derived_nested_columns": derived_nested_cols,
        "dropped_columns": dropped_columns,
        "grain": "entity-day",
        "partition_strategy": "single_file",
        "cmd_args": sys.argv[1:],
        "git_head": git_head,
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_core} ({rows_emitted:,} rows), {out_static}", flush=True)
    if extras_output_columns:
        print(f"Wrote {out_extras} ({rows_written_extras:,} rows)", flush=True)


if __name__ == "__main__":
    main()
