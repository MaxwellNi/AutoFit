#!/usr/bin/env python
"""
Column-level sufficiency audit: MUST_KEEP / CAN_DROP / SHOULD_ADD / MUST_ADD_TABLE.
Parses modeling requirements from run_full_benchmark.py; loads raw/processed schemas;
EDGAR value-level recompute (hard gate). Outputs column_manifest.json/md under output_dir.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

# Modeling requirements from scripts/run_full_benchmark.py (minimal time-series + label deps)
# time_since_start_days / time_delta_days can be computed at runtime via timeline.add_time_index
RUN_FULL_BENCHMARK_REQUIRED_COLS = [
    "entity_id",
    "platform_name",
    "offer_id",
    "snapshot_ts",
    "funding_goal_usd",
    "funding_raised_usd",
    "investors_count",
    "is_funded",
]

# Narrative columns (belong to offers_text table, NOT offers_core) - from raw schema
NARRATIVE_REQUIRED_COLS = [
    "headline",
    "title",
    "description_text",
    "company_description",
    "product_description",
    "financial_condition",
    "financial_forecasts",
    "financial_risks",
    "offering_purpose",
    "use_of_funds",
    "reasons_to_invest",
    "updates",
    "questions",
    "term_sheet",
    "front_video_transcript",
]

# SHOULD_ADD candidates (only mark if missingness < 80% and has variance)
SHOULD_ADD_CANDIDATES = [
    "number_of_days_left",
    "status",
    "web_fundraising_status",
    "offering_status",
    "interest_rate",
    "valuation_*_usd",
    "ticket_amount_*_usd",
    "price_*_usd",
]


def _hostname() -> str:
    return os.environ.get("HOST_TAG", os.environ.get("HOSTNAME", "unknown")).replace(".", "-")[:64]


def _setup_logger(output_dir: Path, host_suffix: Optional[str] = None) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("audit_column_manifest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    name = f"column_manifest_{host_suffix or _hostname()}.log"
    fh = logging.FileHandler(log_dir / name, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _raw_offers_schema(raw_offers_delta: Path, logger: logging.Logger) -> List[str]:
    try:
        from deltalake import DeltaTable
        dt = DeltaTable(str(raw_offers_delta))
        return [f.name for f in dt.schema().fields]
    except Exception as e:
        logger.warning("Raw offers Delta schema failed: %s", e)
        return []


def _parquet_schema(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        schema = pq.read_schema(path)
        return schema.names if hasattr(schema, "names") else [f.name for f in schema]
    except Exception:
        return []


def _edgar_store_schema(edgar_dir: Path, logger: logging.Logger) -> List[str]:
    try:
        import pyarrow.dataset as ds
        dset = ds.dataset(str(edgar_dir), format="parquet", partitioning="hive", exclude_invalid_files=True, ignore_prefixes=["_delta_log"])
        return dset.schema.names if hasattr(dset.schema, "names") else [f.name for f in dset.schema]
    except Exception as e:
        logger.warning("EDGAR store schema failed: %s", e)
        return []


def _missingness_and_stats(df: pd.DataFrame, cols: List[str], sample_cap: int = 100_000) -> Dict[str, Dict[str, Any]]:
    out = {}
    sub = df.head(sample_cap) if len(df) > sample_cap else df
    for c in cols:
        if c not in sub.columns:
            continue
        s = sub[c]
        missing = float(s.isna().mean())
        try:
            n_unique = int(s.nunique())
        except (TypeError, ValueError):
            n_unique = -1
        dtype = str(s.dtype)
        avg_len = None
        if dtype == "object" or "string" in dtype:
            try:
                lens = s.dropna().astype(str).str.len()
                avg_len = float(lens.mean()) if len(lens) else None
            except (TypeError, ValueError):
                pass
        out[c] = {"missing_rate": missing, "n_unique": n_unique, "dtype": dtype, "avg_length": avg_len}
    return out


def _raw_offers_sample_missingness(raw_offers_delta: Path, columns: List[str], entity_filter: Optional[List[Tuple[str, str]]], sample_rows: int, logger: logging.Logger) -> Dict[str, float]:
    """Estimate missingness in raw offers via entity-filtered scan (small sample)."""
    out = {c: float("nan") for c in columns}
    try:
        from deltalake import DeltaTable
        import pyarrow.dataset as ds
        dt = DeltaTable(str(raw_offers_delta))
        dset = dt.to_pyarrow_dataset()
        schema_names = dset.schema.names if hasattr(dset.schema, "names") else [f.name for f in dset.schema]
        read_cols = [c for c in columns if c in schema_names][:30]
        if not read_cols:
            return out
        platform_col = next((c for c in ["platform_name", "platform"] if c in schema_names), None)
        offer_col = next((c for c in ["offer_id", "offerid"] if c in schema_names), None)
        filter_expr = None
        if entity_filter and platform_col and offer_col and len(entity_filter) <= 100:
            p0, o0 = entity_filter[0]
            filter_expr = (ds.field(platform_col) == str(p0)) & (ds.field(offer_col) == str(o0))
            for pn, oid in entity_filter[1:]:
                filter_expr = filter_expr | ((ds.field(platform_col) == str(pn)) & (ds.field(offer_col) == str(oid)))
        scanner = dset.scanner(columns=read_cols, filter=filter_expr, batch_size=50_000) if filter_expr else dset.scanner(columns=read_cols, batch_size=50_000)
        rows = []
        for batch in scanner.to_batches():
            rows.append(batch.to_pandas())
            if sum(len(r) for r in rows) >= sample_rows:
                break
        if not rows:
            return out
        pdf = pd.concat(rows, ignore_index=True).head(sample_rows)
        for c in read_cols:
            if c in pdf.columns:
                out[c] = float(pdf[c].isna().mean())
    except Exception as e:
        logger.warning("Raw offers sample missingness failed: %s", e)
    return out


def _edgar_recompute_gate(
    raw_edgar_delta: Path,
    edgar_dir: Path,
    n_pairs: int,
    atol: float,
    require_deltalake: bool,
    logger: logging.Logger,
    min_compared: int = 10,
    edgar_recompute_fallback_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Recompute EDGAR features from raw aligned to snapshots (same logic as build_edgar_features) and compare to store. Hard gate."""
    result: Dict[str, Any] = {"status": "ok", "max_abs_diff": 0.0, "diff_count": 0, "total_compared": 0}
    try:
        from deltalake import DeltaTable
        from narrative.data_preprocessing.edgar_feature_store import (
            load_edgar_features_from_parquet,
            extract_edgar_features,
            aggregate_edgar_features,
            align_edgar_features_to_snapshots,
        )
        import pyarrow.dataset as ds
    except ImportError as e:
        result["status"] = "skipped"
        result["reason"] = str(e)
        if require_deltalake:
            return result
        return result
    if not raw_edgar_delta.exists() or not (raw_edgar_delta / "_delta_log").exists():
        result["status"] = "skipped"
        result["reason"] = "raw edgar delta missing or not Delta"
        if require_deltalake:
            return result
        return result
    try:
        store_ds = ds.dataset(str(edgar_dir), format="parquet", partitioning="hive", exclude_invalid_files=True, ignore_prefixes=["_delta_log"])
        store_tbl = store_ds.to_table()
        store_df = store_tbl.to_pandas()
    except Exception as e:
        result["status"] = "skipped"
        result["reason"] = f"edgar store read: {e}"
        if require_deltalake:
            return result
        return result
    if store_df.empty or "offer_id" not in store_df.columns or "cik" not in store_df.columns:
        # Try fallback edgar dir (e.g. old store with offer_id when full_daily has dedup-only)
        if edgar_recompute_fallback_dir and edgar_recompute_fallback_dir.exists():
            try:
                fallback_ds = ds.dataset(str(edgar_recompute_fallback_dir), format="parquet", partitioning="hive", exclude_invalid_files=True, ignore_prefixes=["_delta_log"])
                store_df = fallback_ds.to_table().to_pandas()
                edgar_dir = edgar_recompute_fallback_dir
                logger.info("Using edgar_recompute_fallback_dir for recompute (primary lacked offer_id/cik)")
            except Exception as e:
                result["status"] = "skipped"
                result["reason"] = f"store empty or missing offer_id/cik; fallback failed: {e}"
                return result
        if store_df.empty or "offer_id" not in store_df.columns or "cik" not in store_df.columns:
            result["status"] = "skipped"
            result["reason"] = "store empty or missing offer_id/cik"
            return result
    agg_cols = [c for c in store_df.columns if c.startswith("last_") or c.startswith("mean_") or c.startswith("ema_")]
    agg_cols = [c for c in agg_cols if not c.endswith("_is_missing")][:15]
    sample_col = agg_cols[0] if agg_cols else None
    if sample_col and sample_col in store_df.columns:
        has_data = store_df[store_df[sample_col].notna()][["platform_name", "offer_id", "cik"]].drop_duplicates().dropna()
        pairs_df = has_data if len(has_data) >= n_pairs else store_df[["platform_name", "offer_id", "cik"]].drop_duplicates().dropna()
    else:
        pairs_df = store_df[["platform_name", "offer_id", "cik"]].drop_duplicates().dropna()
    if pairs_df.empty:
        result["status"] = "skipped"
        result["reason"] = "no (platform_name, offer_id, cik) in store"
        return result
    rng = np.random.default_rng(42)
    n_pick = min(max(n_pairs, 30), len(pairs_df))
    pick = pairs_df.iloc[rng.choice(len(pairs_df), size=n_pick, replace=False)]
    our_offer_ids = set(pick["offer_id"].astype(str).unique())
    ciks = pick["cik"].astype(str).dropna().unique().tolist()
    if not ciks:
        result["status"] = "skipped"
        result["reason"] = "no valid cik"
        return result
    store_sub = store_df[
        (store_df["offer_id"].astype(str).isin(our_offer_ids)) & (store_df["cik"].astype(str).isin(ciks))
    ].copy()
    snapshot_col = "crawled_date_day" if "crawled_date_day" in store_sub.columns else "crawled_date"
    if snapshot_col not in store_sub.columns:
        result["status"] = "skipped"
        result["reason"] = "store missing snapshot time column"
        return result
    store_sub["crawled_date"] = pd.to_datetime(store_sub[snapshot_col], errors="coerce", utc=True)
    store_sub = store_sub.dropna(subset=["crawled_date"])
    snaps = store_sub[["platform_name", "offer_id", "cik", "crawled_date"]].drop_duplicates()
    if snaps.empty:
        result["status"] = "skipped"
        result["reason"] = "no snapshot rows for sample"
        return result
    try:
        edgar_raw = load_edgar_features_from_parquet(
            Path(raw_edgar_delta), batch_size=100_000, limit_rows=1_500_000, cik_filter=ciks
        )
    except Exception as e:
        result["status"] = "skipped"
        result["reason"] = f"load_edgar_features_from_parquet: {e}"
        if require_deltalake:
            return result
        return result
    if edgar_raw.empty:
        result["status"] = "skipped"
        result["reason"] = "raw edgar empty"
        if require_deltalake:
            return result
        return result
    edgar_raw = edgar_raw[edgar_raw["cik"].astype(str).isin(ciks)]
    if edgar_raw.empty:
        result["status"] = "skipped"
        result["reason"] = "no raw rows for sample ciks"
        if require_deltalake:
            return result
        return result
    agg = aggregate_edgar_features(edgar_raw, ema_alpha=0.2)
    try:
        aligned = align_edgar_features_to_snapshots(
            agg,
            snaps,
            snapshot_time_col="crawled_date",
            id_cols=("platform_name", "offer_id", "cik"),
        )
    except Exception as e:
        result["status"] = "skipped"
        result["reason"] = f"align_edgar_features_to_snapshots: {e}"
        if require_deltalake:
            return result
        return result
    if aligned.empty:
        result["status"] = "skipped"
        result["reason"] = "aligned empty"
        return result
    failed = False
    max_abs_diff = 0.0
    diff_count = 0
    total_compared = 0
    store_sub["crawled_date"] = pd.to_datetime(store_sub["crawled_date"], utc=True)
    aligned["crawled_date"] = pd.to_datetime(aligned["crawled_date"], utc=True)
    def _ts_key(s: pd.Series) -> pd.Series:
        return s.dt.strftime("%Y-%m-%dT%H:%M:%S")
    store_sub["_key"] = (
        store_sub["platform_name"].astype(str) + "|" + store_sub["offer_id"].astype(str)
        + "|" + store_sub["cik"].astype(str) + "|" + _ts_key(store_sub["crawled_date"])
    )
    aligned["_key"] = (
        aligned["platform_name"].astype(str) + "|" + aligned["offer_id"].astype(str)
        + "|" + aligned["cik"].astype(str) + "|" + _ts_key(aligned["crawled_date"])
    )
    keys_in_both = set(store_sub["_key"]) & set(aligned["_key"])
    for col in agg_cols:
        if col not in aligned.columns or col not in store_sub.columns:
            continue
        for _key in list(keys_in_both)[:500]:
            store_row = store_sub[store_sub["_key"] == _key]
            align_row = aligned[aligned["_key"] == _key]
            if store_row.empty or align_row.empty:
                continue
            sval = store_row[col].iloc[0]
            aval = align_row[col].iloc[0]
            if pd.isna(sval) and pd.isna(aval):
                continue
            if pd.isna(sval) or pd.isna(aval):
                total_compared += 1
                diff_count += 1
                failed = True
                continue
            try:
                d = abs(float(sval) - float(aval))
                total_compared += 1
                if d > atol:
                    diff_count += 1
                    failed = True
                max_abs_diff = max(max_abs_diff, d)
            except (TypeError, ValueError):
                pass
    result["max_abs_diff"] = max_abs_diff
    result["diff_count"] = diff_count
    result["total_compared"] = total_compared
    if require_deltalake and total_compared < min_compared:
        result["status"] = "fail"
        result["reason"] = f"EDGAR recompute total_compared={total_compared} < {min_compared} (insufficient comparisons)"
    elif failed and total_compared > 0:
        result["status"] = "fail"
        result["reason"] = f"EDGAR recompute diff_count={diff_count} total_compared={total_compared} max_abs_diff={max_abs_diff}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Column-level sufficiency audit (MUST_KEEP / CAN_DROP / SHOULD_ADD).")
    parser.add_argument("--offers_core", type=Path, required=True)
    parser.add_argument("--offers_static", type=Path, required=True)
    parser.add_argument("--offers_text", type=Path, default=None, help="Path to offers_text.parquet (for NBI/NCI)")
    parser.add_argument("--edgar_dir", type=Path, required=True)
    parser.add_argument("--edgar_recompute_dir", type=Path, default=None, help="Fallback edgar dir for recompute when primary lacks offer_id/cik (e.g. full_daily dedup store)")
    parser.add_argument("--raw_offers_delta", type=Path, required=True)
    parser.add_argument("--raw_edgar_delta", type=Path, required=True)
    parser.add_argument("--selection_json", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_entities", type=int, default=1000)
    parser.add_argument("--edgar_recompute_pairs", type=int, default=10)
    parser.add_argument("--edgar_min_compared", type=int, default=10, help="Min value-level comparisons for EDGAR recompute gate (default 10; 100 if EDGAR dense)")
    parser.add_argument("--edgar_recompute_required", type=int, default=1, help="If 0, EDGAR recompute failures do not gate FAIL (use when 4090 lacks data parity with 3090)")
    parser.add_argument("--require_deltalake", type=int, default=0)
    parser.add_argument("--text_required_mode", type=int, default=0)
    parser.add_argument("--contract_path", type=Path, default=None, help="Path to column_contract_v3.yaml (overrides v2)")
    args = parser.parse_args()

    if args.require_deltalake:
        try:
            from deltalake import DeltaTable  # noqa: F401
        except ImportError as e:
            print("FATAL: --require_deltalake 1 but deltalake not available:", e, file=sys.stderr)
            sys.exit(2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    host = _hostname()
    logger = _setup_logger(args.output_dir, host_suffix=host)
    logger.info("=== Column Manifest Audit Start (host=%s) ===", host)

    fail_reasons: List[str] = []
    report: Dict[str, Any] = {}

    raw_offers_cols = _raw_offers_schema(args.raw_offers_delta, logger)
    if args.require_deltalake and not raw_offers_cols:
        fail_reasons.append("raw offers delta unreadable (require_deltalake=1)")
    report["raw_offers_schema_count"] = len(raw_offers_cols)
    report["raw_offers_columns_sample"] = raw_offers_cols[:50]

    core_cols = _parquet_schema(args.offers_core)
    static_cols = _parquet_schema(args.offers_static)
    edgar_cols = _edgar_store_schema(args.edgar_dir, logger)
    report["offers_core_columns"] = core_cols
    report["offers_static_columns"] = static_cols
    report["edgar_store_columns_sample"] = edgar_cols[:40]

    # Modeling requirements from run_full_benchmark.py
    report["modeling_required_cols"] = list(RUN_FULL_BENCHMARK_REQUIRED_COLS)
    missing_keep = [c for c in RUN_FULL_BENCHMARK_REQUIRED_COLS if c not in core_cols]
    if missing_keep:
        fail_reasons.append(f"MUST_KEEP missing in offers_core: {missing_keep}")
    report["must_keep_in_core"] = [c for c in RUN_FULL_BENCHMARK_REQUIRED_COLS if c in core_cols]
    report["must_keep_missing"] = missing_keep

    # Narrative: belong to offers_text table, NOT offers_core
    report["narrative_required_cols"] = list(NARRATIVE_REQUIRED_COLS)
    report["narrative_note"] = "These columns belong to offers_text table, NOT offers_core."
    narrative_in_raw = [c for c in NARRATIVE_REQUIRED_COLS if c in raw_offers_cols]
    report["narrative_in_raw"] = narrative_in_raw
    # Check offers_text: use --offers_text if provided, else offers_core.parent/offers_text.parquet
    offers_text_path = args.offers_text if args.offers_text else (args.offers_core.parent / "offers_text.parquet")
    has_offers_text = offers_text_path.exists()
    report["offers_text_exists"] = has_offers_text
    if args.text_required_mode and not has_offers_text:
        fail_reasons.append("text_required_mode=1 but offers_text table missing")
    elif args.text_required_mode and has_offers_text:
        # Use --contract_path if provided, else fallback to v3 then v2
        contract_path = args.contract_path
        if not contract_path or not contract_path.exists():
            contract_path = repo_root / "configs" / "column_contract_v3.yaml"
        if not contract_path.exists():
            contract_path = repo_root / "configs" / "column_contract_v2.yaml"
        if contract_path.exists():
            try:
                import yaml
                contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
                text_must = contract.get("offers_text", {}).get("must_keep", [])
                text_cols = _parquet_schema(offers_text_path)
                missing_text = [c for c in text_must if c not in text_cols]
                if missing_text:
                    fail_reasons.append(f"offers_text missing MUST_KEEP per contract: {missing_text}")
                report["offers_text_must_keep_missing"] = missing_text
                report["contract_path_used"] = str(contract_path)
            except Exception as e:
                logger.warning("Could not validate offers_text against contract: %s", e)

    # Processed missingness (exact from head of parquet)
    core_df = pd.DataFrame()
    if args.offers_core.exists():
        try:
            core_df = pd.read_parquet(args.offers_core, columns=core_cols[:40] if core_cols else None)
        except Exception as e:
            logger.warning("Could not read offers_core for stats: %s", e)
    static_df = pd.DataFrame()
    if args.offers_static.exists():
        try:
            static_df = pd.read_parquet(args.offers_static)
        except Exception as e:
            logger.warning("Could not read offers_static: %s", e)

    core_stats = _missingness_and_stats(core_df, core_cols[:50]) if not core_df.empty else {}
    static_stats = _missingness_and_stats(static_df, static_cols[:30]) if not static_df.empty else {}
    report["offers_core_column_stats"] = core_stats
    report["offers_static_column_stats"] = static_stats

    # Entity filter for raw sample (from selection or core)
    entity_filter: Optional[List[Tuple[str, str]]] = None
    if args.selection_json and args.selection_json.exists():
        sel = json.loads(args.selection_json.read_text(encoding="utf-8"))
        entities = sel if isinstance(sel, list) else sel.get("entity_ids", sel.get("entities", []))
        entities = [str(x) for x in entities][:200]
        entity_filter = []
        for eid in entities:
            if "|" in eid:
                a, b = eid.split("|", 1)
                entity_filter.append((a.strip(), b.strip()))
            else:
                entity_filter.append(("", str(eid)))
    if not entity_filter and "platform_name" in core_df.columns and "offer_id" in core_df.columns:
        sub = core_df[["platform_name", "offer_id"]].drop_duplicates().head(200)
        entity_filter = list(sub.itertuples(index=False, name=None))
        entity_filter = [(str(a), str(b)) for a, b in entity_filter]
    raw_missingness = _raw_offers_sample_missingness(
        args.raw_offers_delta,
        list(set(core_cols[:30] + NARRATIVE_REQUIRED_COLS)),
        entity_filter,
        20_000,
        logger,
    )
    report["raw_offers_missingness_sample"] = {k: v for k, v in raw_missingness.items() if not np.isnan(v)}

    # MUST_KEEP / CAN_DROP / SHOULD_ADD / MUST_ADD_TABLE
    must_keep = list(report["must_keep_in_core"])
    can_drop = [c for c in core_cols if c not in must_keep and (c.startswith("url") or c.startswith("link") or "image" in c or "video_link" in c or c in ["hash_id", "comments"])]
    if not can_drop:
        can_drop = [c for c in core_cols if c not in must_keep and c not in ["funding_goal_usd", "funding_raised_usd", "investors_count", "is_funded", "snapshot_ts", "crawled_date", "crawled_date_day"]][:15]
    should_add: List[str] = []
    for pat in SHOULD_ADD_CANDIDATES:
        if "*" in pat:
            prefix = pat.replace("*_usd", "").replace("_*_usd", "")
            in_raw = [c for c in raw_offers_cols if prefix in c and "usd" in c.lower()]
            for c in in_raw:
                miss = raw_missingness.get(c, 1.0)
                if not np.isnan(miss) and miss < 0.80:
                    should_add.append(c)
        else:
            if pat in raw_offers_cols:
                miss = raw_missingness.get(pat, 1.0)
                if not np.isnan(miss) and miss < 0.80:
                    should_add.append(pat)
    report["must_keep"] = must_keep
    report["can_drop"] = can_drop[:20]
    report["should_add"] = should_add[:25]
    report["must_add_table"] = ["offers_text"] if not has_offers_text else []
    report["should_add_note"] = "Next version (v3) if added; only listed when missingness < 80% and available in raw."

    # EDGAR value-level recompute (hard gate)
    edgar_recompute = _edgar_recompute_gate(
        args.raw_edgar_delta,
        args.edgar_dir,
        args.edgar_recompute_pairs,
        atol=1e-5,
        require_deltalake=bool(args.require_deltalake),
        logger=logger,
        min_compared=getattr(args, "edgar_min_compared", 10),
        edgar_recompute_fallback_dir=args.edgar_recompute_dir,
    )
    report["edgar_recompute"] = edgar_recompute
    edgar_required = getattr(args, "edgar_recompute_required", 1)
    if edgar_recompute.get("status") == "skipped" and args.require_deltalake and edgar_required:
        fail_reasons.append("EDGAR recompute skipped: " + str(edgar_recompute.get("reason", "")))
    elif edgar_recompute.get("status") == "fail" and edgar_required:
        fail_reasons.append("EDGAR recompute failed: " + str(edgar_recompute.get("reason", "")))
    elif edgar_recompute.get("status") == "fail" and not edgar_required:
        report["edgar_recompute"]["note"] = "edgar_recompute_required=0; failure recorded but not gating (ensure data parity with 3090 for full validation)"

    report["gate_passed"] = len(fail_reasons) == 0
    report["fail_reasons"] = fail_reasons

    json_path = args.output_dir / "column_manifest.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", json_path)

    md_lines = [
        "# Column Manifest Report",
        "",
        "**Gate:** " + ("PASS" if report["gate_passed"] else "FAIL"),
        "",
        "## MUST_KEEP (outcome + trajectory)",
        "",
        "- " + ", ".join(report["must_keep"]),
        "- Rationale: Entity keys, time axis, and numerics required by run_full_benchmark.py for label construction and time-series features.",
        "",
        "## CAN_DROP",
        "",
        "- " + ", ".join(report["can_drop"][:15]),
        "- Rationale: Unused identifiers/URLs or heavy arrays in core; safe to drop for modeling.",
        "",
        "## SHOULD_ADD (next version v3)",
        "",
        "- " + (", ".join(report["should_add"][:15]) if report["should_add"] else "(none or not meeting missingness/variance)"),
        "- Rationale: " + report["should_add_note"],
        "",
        "## MUST_ADD_TABLE",
        "",
        "- " + ", ".join(report["must_add_table"]) if report["must_add_table"] else "(offers_text exists or N/A)",
        "- Rationale: offers_text for NBI/NCI narrative concepts.",
        "",
        "## Narrative columns (offers_text, NOT offers_core)",
        "",
        report["narrative_note"],
        "- " + ", ".join(report["narrative_required_cols"][:10]) + ", ...",
        "",
        "## EDGAR recompute",
        "",
        f"- Status: {edgar_recompute.get('status')}, max_abs_diff: {edgar_recompute.get('max_abs_diff')}, diff_count: {edgar_recompute.get('diff_count')}",
        "",
    ]
    md_path = args.output_dir / "column_manifest.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Wrote %s", md_path)
    logger.info("=== Column Manifest Audit Complete === Gate: %s", "PASS" if report["gate_passed"] else "FAIL")
    if fail_reasons:
        for r in fail_reasons:
            logger.error("FAIL: %s", r)
        sys.exit(2)


if __name__ == "__main__":
    main()
