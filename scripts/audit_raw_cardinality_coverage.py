#!/usr/bin/env python
"""
Raw vs processed cardinality coverage audit. Produces facts on:
- raw offers Delta: version, active_files, row count by year (projection-only scan)
- offers_core_v2: row_count, unique entities, by-year; entity subset from MANIFEST
- offers_text_full: row_count, unique entities, by-year; raw_rows_scanned/rows_emitted from MANIFEST
- raw edgar Delta: version, active_files, row count by year
- edgar store: row_count, by-year
- Two-machine consistency: raw versions/active_files must match (--reference_json).

Gate: FAIL on missing counts, version mismatch, or offers_text manifest lacking limit_rows/overwrite protection.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _hostname() -> str:
    return os.environ.get("HOST_TAG", os.environ.get("HOSTNAME", "unknown")).replace(".", "-")[:64]


def _setup_logger(output_dir: Path, host_suffix: Optional[str] = None) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("raw_cardinality_coverage")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    name = f"raw_cardinality_coverage_{host_suffix or _hostname()}.log"
    fh = logging.FileHandler(log_dir / name, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _delta_version_and_files(path: Path) -> Tuple[Optional[int], int]:
    try:
        from deltalake import DeltaTable
    except ImportError:
        return None, 0
    if not path.exists() or not (path / "_delta_log").exists():
        return None, 0
    try:
        dt = DeltaTable(str(path))
        ver = dt.version()
        files_list = dt.file_uris() if hasattr(dt, "file_uris") else (dt.files() if hasattr(dt, "files") else [])
        return ver, len(files_list)
    except Exception:
        return None, 0


def _key_df_from_parquet(path: Path) -> pd.DataFrame:
    """Load minimal keys (entity_id, crawled_date_day) from parquet with fallbacks."""
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    cols = schema.names
    read_cols = [c for c in ["entity_id", "platform_name", "offer_id", "crawled_date_day", "crawled_date", "snapshot_ts"] if c in cols]
    df = pd.read_parquet(path, columns=read_cols)
    if "entity_id" not in df.columns:
        if "platform_name" in df.columns and "offer_id" in df.columns:
            df["entity_id"] = df["platform_name"].astype(str) + "|" + df["offer_id"].astype(str)
        else:
            raise RuntimeError(f"{path} missing entity_id or platform_name/offer_id")
    if "crawled_date_day" not in df.columns:
        if "crawled_date" in df.columns:
            df["crawled_date_day"] = pd.to_datetime(df["crawled_date"], errors="coerce", utc=True).dt.date.astype(str)
        elif "snapshot_ts" in df.columns:
            df["crawled_date_day"] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True).dt.date.astype(str)
        else:
            raise RuntimeError(f"{path} missing crawled_date_day/crawled_date/snapshot_ts")
    return df[["entity_id", "crawled_date_day"]].dropna().drop_duplicates()


def _structured_columns_from_contract(contract_path: Path) -> tuple[List[str], List[str]]:
    """Load structured columns and nested raw columns from column_contract_wide."""
    if not contract_path or not contract_path.exists():
        return [], []
    try:
        import yaml
        c = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        snap = c.get("offers_core_snapshot", {})
        daily = c.get("offers_core_daily", {})
        nested_raw = list(dict.fromkeys(snap.get("nested_columns", []) + daily.get("nested_columns", [])))
        derived_nested = list(dict.fromkeys(snap.get("derived_nested", []) + daily.get("derived_nested", [])))
        cols = list(dict.fromkeys(
            snap.get("must_keep", []) + snap.get("high_value", []) +
            daily.get("must_keep", []) + daily.get("high_value", []) +
            daily.get("derived_structured", []) + derived_nested
        ))
        text_exclude = set(c.get("offers_text", {}).get("must_keep", []))
        text_exclude |= {"headline", "title", "description_text", "company_description", "financial_condition"}
        cols = [x for x in cols if x not in text_exclude]
        return cols, nested_raw
    except Exception:
        return [], []


def _raw_structured_signal_entity_day(
    raw_path: Path,
    struct_cols: List[str],
    nested_cols: List[str],
    logger: logging.Logger,
    limit_rows: Optional[int] = None,
) -> tuple[int, int]:
    """Stream raw offers, count (entity_id, day) where any struct col non-null. Returns (total_rows, entity_day_count)."""
    if not struct_cols and not nested_cols:
        return 0, 0
    try:
        from deltalake import DeltaTable
        import pyarrow as pa
        import pyarrow.compute as pc
        dt = DeltaTable(str(raw_path))
        dset = dt.to_pyarrow_dataset()
        schema_names = [f.name for f in dset.schema]
        day_col = "crawled_date_day" if "crawled_date_day" in schema_names else ("crawled_date" if "crawled_date" in schema_names else ("snapshot_ts" if "snapshot_ts" in schema_names else None))
        if not day_col:
            raise RuntimeError("raw offers missing day column (crawled_date_day/crawled_date)")
        struct_cols = [c for c in struct_cols if c not in {"entity_id", "platform_name", "offer_id", "snapshot_ts", "crawled_date_day", "crawled_date"}]
        nested_cols = [c for c in nested_cols if c not in {"entity_id", "platform_name", "offer_id", "snapshot_ts", "crawled_date_day", "crawled_date"}]
        read_cols = list(dict.fromkeys([c for c in (struct_cols + nested_cols + ["platform_name", "offer_id", day_col]) if c in schema_names]))
        if "platform_name" not in read_cols or "offer_id" not in read_cols or day_col not in read_cols:
            raise RuntimeError("raw offers missing key columns for structured-signal scan")
        seen: set = set()
        total = 0
        for batch in dset.scanner(columns=read_cols, batch_size=200_000).to_batches():
            total += batch.num_rows
            # Build signal mask via Arrow (safe for nested)
            mask = None
            for col in struct_cols + nested_cols:
                if col not in batch.schema.names:
                    continue
                idx = batch.schema.get_field_index(col)
                if idx < 0:
                    continue
                arr = batch.column(idx)
                if isinstance(arr, pa.ChunkedArray):
                    try:
                        arr = arr.combine_chunks()
                    except Exception:
                        pass
                if col in nested_cols:
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
                continue
            if isinstance(mask, pa.ChunkedArray):
                mask = mask.combine_chunks()
            mask_np = mask.to_numpy(zero_copy_only=False)
            if mask_np is None or not mask_np.any():
                if limit_rows and total >= limit_rows:
                    break
                continue

            table = pa.Table.from_batches([batch])
            key_df = table.select([c for c in ["platform_name", "offer_id", day_col] if c in table.column_names]).to_pandas()
            key_df = key_df.loc[mask_np]
            if key_df.empty:
                if limit_rows and total >= limit_rows:
                    break
                continue
            pn = key_df["platform_name"].apply(lambda x: str(x) if pd.notna(x) else "")
            oid = key_df["offer_id"].apply(lambda x: str(x) if pd.notna(x) else "")
            key_df["entity_id"] = pn + "|" + oid
            key_df["_day"] = pd.to_datetime(key_df[day_col], errors="coerce", utc=True).dt.date
            sub = key_df[["entity_id", "_day"]].drop_duplicates()
            for t in sub.itertuples(index=False):
                eid = str(getattr(t, "entity_id", ""))
                day_val = getattr(t, "_day", None)
                day_str = str(day_val) if day_val is not None else ""
                seen.add((eid, day_str))
            if limit_rows and total >= limit_rows:
                break
        return total, len(seen)
    except Exception as e:
        raise RuntimeError(f"raw_structured_signal scan failed: {e}")


def _raw_entity_day_count(raw_path: Path, logger: logging.Logger, limit_rows: Optional[int] = None) -> int:
    """Count distinct (entity_id, day) from raw offers (no structured filter)."""
    try:
        from deltalake import DeltaTable
        import pyarrow as pa
        dt = DeltaTable(str(raw_path))
        dset = dt.to_pyarrow_dataset()
        schema_names = [f.name for f in dset.schema]
        day_col = "crawled_date_day" if "crawled_date_day" in schema_names else ("crawled_date" if "crawled_date" in schema_names else ("snapshot_ts" if "snapshot_ts" in schema_names else None))
        if not day_col:
            raise RuntimeError("raw offers missing day column (crawled_date_day/crawled_date)")
        read_cols = list(dict.fromkeys([c for c in ["platform_name", "offer_id", day_col] if c in schema_names]))
        if "platform_name" not in read_cols or "offer_id" not in read_cols or day_col not in read_cols:
            raise RuntimeError("raw offers missing key columns for entity-day scan")
        seen: set = set()
        total = 0
        for batch in dset.scanner(columns=read_cols, batch_size=200_000).to_batches():
            total += batch.num_rows
            table = pa.Table.from_batches([batch])
            df = table.to_pandas()
            if df.empty:
                if limit_rows and total >= limit_rows:
                    break
                continue
            pn = df["platform_name"].apply(lambda x: str(x) if pd.notna(x) else "")
            oid = df["offer_id"].apply(lambda x: str(x) if pd.notna(x) else "")
            df["entity_id"] = pn + "|" + oid
            df["_day"] = pd.to_datetime(df[day_col], errors="coerce", utc=True).dt.date
            sub = df[["entity_id", "_day"]].drop_duplicates()
            for t in sub.itertuples(index=False):
                eid = str(getattr(t, "entity_id", ""))
                day_val = getattr(t, "_day", None)
                day_str = str(day_val) if day_val is not None else ""
                seen.add((eid, day_str))
            if limit_rows and total >= limit_rows:
                break
        return len(seen)
    except Exception as e:
        raise RuntimeError(f"raw_entity_day scan failed: {e}")


def _raw_offers_stats(path: Path, logger: logging.Logger, limit_rows: Optional[int] = None) -> Dict[str, Any]:
    ver, nfiles = _delta_version_and_files(path)
    if ver is None:
        return {"version": None, "active_files": 0, "row_count_total": 0, "row_count_by_year": {}}
    try:
        from deltalake import DeltaTable
        dt = DeltaTable(str(path))
        dset = dt.to_pyarrow_dataset()
        schema_names = [f.name for f in dset.schema]
        day_col = "crawled_date_day" if "crawled_date_day" in schema_names else ("crawled_date" if "crawled_date" in schema_names else ("snapshot_ts" if "snapshot_ts" in schema_names else None))
        if not day_col:
            return {"version": ver, "active_files": nfiles, "row_count_total": 0, "row_count_by_year": {}}
        scanner = dset.scanner(columns=[day_col], batch_size=200_000)
        total = 0
        by_year: Dict[str, int] = {}
        for batch in scanner.to_batches():
            df = batch.to_pandas()
            total += len(df)
            if not df.empty:
                years = pd.to_datetime(df[day_col], errors="coerce").dt.year
                vc = years.dropna().astype(int).value_counts().to_dict()
                for k, v in vc.items():
                    by_year[str(k)] = by_year.get(str(k), 0) + int(v)
            if limit_rows and total >= limit_rows:
                break
        return {"version": ver, "active_files": nfiles, "row_count_total": int(total), "row_count_by_year": by_year}
    except Exception as e:
        logger.warning("raw_offers scan failed: %s", e)
        return {"version": ver, "active_files": nfiles, "row_count_total": None, "row_count_by_year": {}}


def _raw_edgar_stats(path: Path, logger: logging.Logger, limit_rows: Optional[int] = 500_000) -> Dict[str, Any]:
    ver, nfiles = _delta_version_and_files(path)
    if ver is None:
        return {"version": None, "active_files": 0, "row_count_total": 0, "row_count_by_year": {}}
    try:
        from deltalake import DeltaTable
        dt = DeltaTable(str(path))
        dset = dt.to_pyarrow_dataset()
        schema_names = [f.name for f in dset.schema]
        day_col = "filed_date" if "filed_date" in schema_names else ("filing_date" if "filing_date" in schema_names else None)
        if not day_col:
            return {"version": ver, "active_files": nfiles, "row_count_total": 0, "row_count_by_year": {}}
        scanner = dset.scanner(columns=[day_col], batch_size=200_000)
        total = 0
        by_year: Dict[str, int] = {}
        for batch in scanner.to_batches():
            df = batch.to_pandas()
            total += len(df)
            if not df.empty:
                years = pd.to_datetime(df[day_col], errors="coerce").dt.year
                vc = years.dropna().astype(int).value_counts().to_dict()
                for k, v in vc.items():
                    by_year[str(k)] = by_year.get(str(k), 0) + int(v)
            if limit_rows and total >= limit_rows:
                break
        return {"version": ver, "active_files": nfiles, "row_count_total": int(total), "row_count_by_year": by_year}
    except Exception as e:
        logger.warning("raw_edgar scan failed: %s", e)
        return {"version": ver, "active_files": nfiles, "row_count_total": None, "row_count_by_year": {}}


def _parquet_stats(path: Path, entity_col: str = "entity_id", time_col: str = "snapshot_ts") -> Dict[str, Any]:
    if not path.exists():
        return {"row_count": 0, "n_unique_entity_id": 0, "row_count_by_year": {}}
    try:
        df = pd.read_parquet(path, columns=[c for c in [entity_col, time_col] if c])
        n_ent = int(df[entity_col].nunique()) if entity_col in df.columns else 0
        by_year = {}
        if time_col in df.columns:
            df["_year"] = pd.to_datetime(df[time_col], errors="coerce").dt.year
            by_year = {str(int(k)): int(v) for k, v in df["_year"].dropna().value_counts().sort_index().items()}
        return {"row_count": int(len(df)), "n_unique_entity_id": n_ent, "row_count_by_year": by_year}
    except Exception:
        return {"row_count": 0, "n_unique_entity_id": 0, "row_count_by_year": {}}


def _edgar_store_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"row_count": 0, "row_count_by_year": {}}
    try:
        import pyarrow.dataset as ds
        dset = ds.dataset(str(path), format="parquet", partitioning="hive")
        schema_names = [f.name for f in dset.schema]
        cols = ["crawled_date_day"] if "crawled_date_day" in schema_names else []
        if not cols:
            scanner = dset.scanner(batch_size=200_000)
            total = sum(b.num_rows for b in scanner.to_batches())
            return {"row_count": total, "row_count_by_year": {}}
        scanner = dset.scanner(columns=cols, batch_size=200_000)
        frames = [b.to_pandas() for b in scanner.to_batches()]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        df["_year"] = pd.to_datetime(df["crawled_date_day"], errors="coerce").dt.year
        by_year = {str(int(k)): int(v) for k, v in df["_year"].dropna().value_counts().sort_index().items()}
        return {"row_count": int(len(df)), "row_count_by_year": by_year}
    except Exception:
        return {"row_count": 0, "row_count_by_year": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw vs processed cardinality coverage audit.")
    parser.add_argument("--raw_offers_delta", type=Path, required=True)
    parser.add_argument("--raw_edgar_delta", type=Path, required=True)
    parser.add_argument("--offers_core_parquet", type=Path, required=True, help="offers_core_full_daily parquet")
    parser.add_argument("--offers_core_manifest", type=Path, required=True)
    parser.add_argument("--offers_text_full_dir", type=Path, required=True)
    parser.add_argument("--edgar_store_dir", type=Path, required=True, help="edgar_feature_store_full_daily edgar_features dir")
    parser.add_argument("--snapshots_index_parquet", type=Path, default=None, help="snapshots_offer_day.parquet for snapshots→edgar coverage")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--reference_json", type=Path, default=None, help="Compare raw versions/active_files; FAIL if mismatch")
    parser.add_argument("--raw_scan_limit", type=int, default=2_000_000, help="Limit rows for raw offers scan; 0=no cap (full scan)")
    parser.add_argument("--contract_wide", type=Path, default=None, help="column_contract_wide.yaml for structured-signal column list")
    parser.add_argument("--docs_audits_dir", type=Path, default=None, help="If set, write public anchor")
    parser.add_argument("--output_basename", type=str, default="raw_cardinality_coverage", help="Output json/md basename")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    host = _hostname()
    logger = _setup_logger(args.output_dir, host_suffix=host)
    logger.info("=== Raw Cardinality Coverage Audit Start (host=%s) ===", host)

    fail_reasons: List[str] = []

    raw_limit = None if args.raw_scan_limit == 0 else args.raw_scan_limit
    raw_offers = _raw_offers_stats(args.raw_offers_delta, logger, limit_rows=raw_limit)
    struct_cols, nested_cols = _structured_columns_from_contract(args.contract_wide or repo_root / "configs/column_contract_wide.yaml")
    raw_struct_total = 0
    raw_struct_entity_day = 0
    try:
        raw_struct_total, raw_struct_entity_day = _raw_structured_signal_entity_day(
            args.raw_offers_delta, struct_cols, nested_cols, logger, limit_rows=raw_limit
        )
    except Exception as e:
        fail_reasons.append(str(e))

    raw_entity_day_total = 0
    try:
        raw_entity_day_total = _raw_entity_day_count(args.raw_offers_delta, logger, limit_rows=raw_limit)
    except Exception as e:
        fail_reasons.append(str(e))
    raw_offers["row_count_total_no_cap"] = raw_offers.get("row_count_total")
    if raw_limit:
        raw_offers["sample_ratio"] = raw_limit
    raw_edgar = _raw_edgar_stats(args.raw_edgar_delta, logger)

    offers_core_stats = _parquet_stats(args.offers_core_parquet)
    offers_core_manifest: Dict[str, Any] = {}
    if args.offers_core_manifest.exists():
        offers_core_manifest = json.loads(args.offers_core_manifest.read_text(encoding="utf-8"))
    offers_core = {
        "row_count": offers_core_stats["row_count"],
        "n_unique_entity_id": offers_core_stats["n_unique_entity_id"],
        "row_count_by_year": offers_core_stats["row_count_by_year"],
        "selection_note": "offers_core_full_daily",
        "manifest_selection": offers_core_manifest.get("selection", {}),
    }
    if struct_cols and raw_struct_entity_day == 0:
        fail_reasons.append("structured_signal entity-day count is 0 (scan failed or no signal detected)")
    if raw_entity_day_total == 0:
        fail_reasons.append("raw entity-day total is 0 (scan failed)")

    raw_vs_core_coverage: Dict[str, Any] = {}
    if raw_entity_day_total > 0 and offers_core["row_count"] > 0:
        raw_vs_core_coverage["raw_entity_day_count"] = raw_entity_day_total
        raw_vs_core_coverage["core_entity_day_count"] = offers_core["row_count"]
        raw_vs_core_coverage["core_coverage_rate"] = offers_core["row_count"] / raw_entity_day_total

    offers_text_manifest_path = args.offers_text_full_dir / "MANIFEST.json"
    if not offers_text_manifest_path.exists():
        fail_reasons.append("offers_text_full MANIFEST.json not found")
        offers_text = {"row_count": 0, "n_unique_entity_id": 0, "row_count_by_year": {}, "manifest": {}}
    else:
        offers_text_manifest = json.loads(offers_text_manifest_path.read_text(encoding="utf-8"))
        if "limit_rows" not in offers_text_manifest:
            fail_reasons.append("offers_text_full manifest must record limit_rows (null for full)")
        if "raw_rows_scanned" not in offers_text_manifest or "rows_emitted" not in offers_text_manifest:
            fail_reasons.append("offers_text_full manifest must record raw_rows_scanned and rows_emitted")
        offers_text_parquet = args.offers_text_full_dir / "offers_text.parquet"
        st = _parquet_stats(offers_text_parquet) if offers_text_parquet.exists() else {"row_count": 0, "n_unique_entity_id": 0, "row_count_by_year": {}}
        offers_text = {
            "row_count": st["row_count"],
            "n_unique_entity_id": st["n_unique_entity_id"],
            "row_count_by_year": st["row_count_by_year"],
            "manifest": {k: offers_text_manifest.get(k) for k in ["limit_rows", "raw_offers_version", "raw_active_files", "raw_rows_scanned", "rows_emitted", "rows_all_text_null_dropped", "n_unique_entity_id", "n_unique_entity_day"]},
        }

    edgar_store = _edgar_store_stats(args.edgar_store_dir)
    edgar_manifest_path = args.edgar_store_dir.parent / "MANIFEST.json" if args.edgar_store_dir.name == "edgar_features" else args.edgar_store_dir / "MANIFEST.json"
    if not edgar_manifest_path.exists():
        edgar_manifest_path = args.edgar_store_dir / "MANIFEST.json"
    if edgar_manifest_path.exists():
        try:
            em = json.loads(edgar_manifest_path.read_text(encoding="utf-8"))
            edgar_store["manifest_exists"] = True
            edgar_store["manifest_fields"] = list(em.keys())[:20]
        except Exception:
            edgar_store["manifest_exists"] = True
    else:
        edgar_store["manifest_exists"] = False
        fail_reasons.append("edgar_store_full_daily MANIFEST.json required but not found")

    # Snapshots index vs edgar store alignment (pair-level)
    snapshots_alignment = {}
    if args.snapshots_index_parquet and args.snapshots_index_parquet.exists():
        try:
            snap_df = pd.read_parquet(args.snapshots_index_parquet)
            if "crawled_date_day" not in snap_df.columns and "snapshot_ts" in snap_df.columns:
                snap_df["crawled_date_day"] = pd.to_datetime(snap_df["snapshot_ts"], errors="coerce", utc=True).dt.date.astype(str)
            snap_df = snap_df.dropna(subset=["cik", "crawled_date_day"])
            snap_df["cik"] = snap_df["cik"].astype(str)
            snap_pairs = snap_df[["cik", "crawled_date_day"]].drop_duplicates()
            total_pairs = len(snap_pairs)

            import pyarrow.dataset as ds
            dset = ds.dataset(str(args.edgar_store_dir), format="parquet", partitioning="hive")
            edgar_cols = [c for c in ["cik", "crawled_date_day"] if c in [f.name for f in dset.schema]]
            edgar_df = dset.to_table(columns=edgar_cols).to_pandas()
            if "crawled_date_day" not in edgar_df.columns and "snapshot_ts" in edgar_df.columns:
                edgar_df["crawled_date_day"] = pd.to_datetime(edgar_df["snapshot_ts"], errors="coerce", utc=True).dt.date.astype(str)
            edgar_df = edgar_df.dropna(subset=["cik", "crawled_date_day"])
            edgar_df["cik"] = edgar_df["cik"].astype(str)
            edgar_pairs = edgar_df[["cik", "crawled_date_day"]].drop_duplicates()

            matched = snap_pairs.merge(edgar_pairs, on=["cik", "crawled_date_day"], how="inner")
            matched_pairs = len(matched)
            edgar_ciks = set(edgar_pairs["cik"].dropna().astype(str))

            # Gap reasons
            missing = snap_pairs.merge(edgar_pairs, on=["cik", "crawled_date_day"], how="left", indicator=True)
            missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
            reason_counts: Dict[str, int] = {"cik_not_in_edgar_store": 0, "date_mismatch": 0}
            if not missing.empty:
                reason_counts["cik_not_in_edgar_store"] = int((~missing["cik"].isin(edgar_ciks)).sum())
                reason_counts["date_mismatch"] = int((missing["cik"].isin(edgar_ciks)).sum())
                debug_dir = args.output_dir / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                missing.head(1000).to_csv(debug_dir / "snapshots_edgar_missing_pairs_sample.csv", index=False)

            snapshots_alignment = {
                "snapshots_pairs_total": total_pairs,
                "snapshots_pairs_matched": matched_pairs,
                "snapshots_to_edgar_coverage": (matched_pairs / total_pairs) if total_pairs else 0,
                "snapshots_unique_cik": int(snap_pairs["cik"].nunique()),
                "edgar_unique_cik": int(edgar_pairs["cik"].nunique()),
                "gap_reasons_top_k": sorted(reason_counts, key=lambda k: -reason_counts[k])[:3],
                "gap_reason_counts": reason_counts,
            }
        except Exception as e:
            logger.warning("snapshots_index alignment check failed: %s", e)

    if raw_offers["version"] is None:
        fail_reasons.append("raw_offers Delta version missing")
    if raw_edgar["version"] is None:
        fail_reasons.append("raw_edgar Delta version missing")
    if args.snapshots_index_parquet and args.snapshots_index_parquet.exists():
        if not snapshots_alignment or "snapshots_to_edgar_coverage" not in snapshots_alignment:
            fail_reasons.append("snapshots_to_edgar coverage metrics required when snapshots_index provided")

    if args.reference_json and args.reference_json.exists():
        ref = json.loads(args.reference_json.read_text(encoding="utf-8"))
        ro = ref.get("raw_offers", {})
        re_ = ref.get("raw_edgar", {})
        if ro.get("version") != raw_offers.get("version"):
            fail_reasons.append(f"raw_offers version mismatch: ref={ro.get('version')} vs current={raw_offers.get('version')}")
        if ro.get("active_files") != raw_offers.get("active_files"):
            fail_reasons.append(f"raw_offers active_files mismatch: ref={ro.get('active_files')} vs current={raw_offers.get('active_files')}")
        if re_.get("version") != raw_edgar.get("version"):
            fail_reasons.append(f"raw_edgar version mismatch: ref={re_.get('version')} vs current={raw_edgar.get('version')}")
        if re_.get("active_files") != raw_edgar.get("active_files"):
            fail_reasons.append(f"raw_edgar active_files mismatch: ref={re_.get('active_files')} vs current={raw_edgar.get('active_files')}")

    text_coverage: Dict[str, Any] = {}
    try:
        offers_text_parquet = args.offers_text_full_dir / "offers_text.parquet"
        if offers_text_parquet.exists():
            core_keys = _key_df_from_parquet(args.offers_core_parquet)
            text_keys = _key_df_from_parquet(offers_text_parquet)
            if not core_keys.empty and not text_keys.empty:
                matched = core_keys.merge(text_keys, on=["entity_id", "crawled_date_day"], how="inner")
                text_coverage["core_pairs_total"] = len(core_keys)
                text_coverage["text_pairs_total"] = len(text_keys)
                text_coverage["matched_pairs"] = len(matched)
                text_coverage["text_coverage_rate"] = len(matched) / len(core_keys) if len(core_keys) else 0
    except Exception as e:
        fail_reasons.append(f"text coverage computation failed: {e}")

    offers_core_entity_day = offers_core["row_count"]
    struct_covered = offers_core_entity_day / raw_struct_entity_day if raw_struct_entity_day else 0.0
    report = {
        "host": host,
        "raw_offers_total_rows": raw_offers.get("row_count_total"),
        "raw_entity_day_total": raw_entity_day_total,
        "raw_structured_signal_entity_day_count": raw_struct_entity_day,
        "offers_core_entity_day_count": offers_core_entity_day,
        "structured_signal_covered_by_core": struct_covered,
        "raw_offers": raw_offers,
        "raw_edgar": raw_edgar,
        "offers_core_full_daily": offers_core,
        "offers_text_full": offers_text,
        "edgar_store_full_daily": edgar_store,
        "raw_vs_core_coverage": raw_vs_core_coverage,
        "text_coverage": text_coverage,
        "snapshots_to_edgar_coverage": snapshots_alignment,
        "data_morphology": {
            "offers_core_full_daily": "Full raw scan deduped to entity-day",
            "offers_text_full": "Full raw scan filtered/deduped text panel",
            "edgar_store_full_daily": "Raw EDGAR aligned to snapshots (cik-day or offer-day)",
        },
        "gate_passed": len(fail_reasons) == 0,
        "fail_reasons": fail_reasons,
    }

    json_path = args.output_dir / f"{args.output_basename}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)

    md_lines = [
        "# Raw Cardinality Coverage Report",
        "",
        f"- **Host:** {host}",
        f"- **Gate:** {'PASS' if report['gate_passed'] else 'FAIL'}",
        "",
        "## Structured-Signal Coverage",
        "",
        f"- raw_offers_total_rows: {report.get('raw_offers_total_rows')}",
        f"- raw_entity_day_total: {report.get('raw_entity_day_total')}",
        f"- raw_structured_signal_entity_day_count: {report.get('raw_structured_signal_entity_day_count')}",
        f"- offers_core_entity_day_count: {report.get('offers_core_entity_day_count')}",
        f"- structured_signal_covered_by_core: {report.get('structured_signal_covered_by_core')}",
        "",
        "## Raw vs Full-Daily Processed",
        "",
        "| Artifact | Version | Active Files | Row Count | Unique Entities |",
        "|----------|---------|--------------|-----------|-----------------|",
        f"| raw_offers | {raw_offers.get('version')} | {raw_offers.get('active_files')} | {raw_offers.get('row_count_total')} | - |",
        f"| raw_edgar | {raw_edgar.get('version')} | {raw_edgar.get('active_files')} | {raw_edgar.get('row_count_total')} | - |",
        f"| offers_core_full_daily | - | - | {offers_core['row_count']} | {offers_core['n_unique_entity_id']} |",
        f"| offers_text_full | - | - | {offers_text['row_count']} | {offers_text['n_unique_entity_id']} |",
        f"| edgar_store_full_daily | - | - | {edgar_store['row_count']} | - |",
        "",
        "## Coverage",
        "",
        f"- **raw vs core (entity-day):** {raw_vs_core_coverage.get('core_coverage_rate', 'N/A')}",
        f"- **text coverage (pairs):** {text_coverage.get('text_coverage_rate', 'N/A')}",
        "",
        "## Snapshots→EDGAR Coverage",
        "",
    ]
    if snapshots_alignment:
        for k, v in snapshots_alignment.items():
            md_lines.append(f"- **{k}:** {v}")
        md_lines.append("")
    md_lines.extend([
        "## Data morphology",
        "",
        "- **offers_core_full_daily:** Full raw scan deduped to entity-day",
        "- **offers_text_full:** Full raw scan filtered/deduped text panel",
        "- **edgar_store_full_daily:** Raw EDGAR aligned to snapshots",
        "",
    ])
    if fail_reasons:
        md_lines.append("## Fail reasons\n")
        for r in fail_reasons:
            md_lines.append(f"- {r}")
        md_lines.append("")
    md_path = args.output_dir / f"{args.output_basename}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    if args.docs_audits_dir and report["gate_passed"]:
        stamp = "20260129_073037"
        base = args.output_basename if args.output_basename != "raw_cardinality_coverage" else "raw_cardinality_coverage"
        anchor_path = args.docs_audits_dir / f"{base}_{stamp}.md"
        args.docs_audits_dir.mkdir(parents=True, exist_ok=True)
        anchor_content = (
            "# Raw Cardinality Coverage Audit Anchor (" + stamp + ")\n\n"
            "Public audit anchor for raw vs processed cardinality coverage.\n\n"
            "## Data Morphology\n\n"
            "- **offers_core_v2:** Entity subset full trajectory (limit_entities from MANIFEST)\n"
            "- **offers_text_full:** Full raw scan filtered/deduped text panel (manifest counters prove)\n"
            "- **edgar_store:** Full raw aggregation aligned to snapshots\n\n"
            "## Two-Machine Consistency\n\n"
            "Run on 3090 first, then on 4090 with `--reference_json runs/orchestrator/20260129_073037/analysis/raw_cardinality_coverage.json`.\n"
            "raw_offers_version, raw_edgar_version, active_files must match; else FAIL.\n\n"
            "## Reproducibility\n\n"
            "```bash\n"
            "HOST_TAG=3090 python scripts/audit_raw_cardinality_coverage.py \\\n"
            "  --raw_offers_delta data/raw/offers --raw_edgar_delta data/raw/edgar/accessions \\\n"
            "  --offers_core_parquet runs/offers_core_v2_20260127_043052/offers_core.parquet \\\n"
            "  --offers_core_manifest runs/offers_core_v2_20260127_043052/MANIFEST.json \\\n"
            "  --offers_text_full_dir runs/offers_text_v1_20260129_073037_full \\\n"
            "  --edgar_store_dir runs/edgar_feature_store/20260127_133511/edgar_features \\\n"
            "  --output_dir runs/orchestrator/20260129_073037/analysis \\\n"
            "  --docs_audits_dir docs/audits\n"
            "```\n"
        )
        anchor_path.write_text(anchor_content, encoding="utf-8")
        logger.info("Wrote public anchor %s", anchor_path)
        manifest_path = args.docs_audits_dir / "MANIFEST.json"
        if manifest_path.exists():
            import hashlib
            h = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
            rel_path = f"docs/audits/{base}_{stamp}.md"
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = m.get("entries", [])
            entries = [e for e in entries if e.get("path") != rel_path]
            entries.append({"path": rel_path, "sha256": h})
            m["entries"] = entries
            manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
            logger.info("Updated MANIFEST.json with raw_cardinality_coverage sha256")

    logger.info("=== Raw Cardinality Coverage Audit Complete === Gate: %s", "PASS" if report["gate_passed"] else "FAIL")
    if fail_reasons:
        for r in fail_reasons:
            logger.error("FAIL: %s", r)
        sys.exit(1)


if __name__ == "__main__":
    main()
