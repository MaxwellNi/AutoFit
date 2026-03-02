#!/usr/bin/env python
"""
Audit processed artifacts: data integrity, coverage, alignment, and label health.

This script validates the integrity and completeness of offers_core, EDGAR feature
store, and benchmark outputs. It produces a structured JSON report and a markdown
summary for KDD 2026 submission readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


def _setup_logger(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("audit_processed_artifacts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_dir / "audit_processed_artifacts.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _load_offers_core(path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading offers_core from %s", path)
    df = pd.read_parquet(path)
    logger.info("offers_core loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


def _load_edgar_dataset(edgar_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading EDGAR feature store from %s", edgar_dir)
    dataset = ds.dataset(
        str(edgar_dir),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
        ignore_prefixes=["_delta_log"],
    )
    table = dataset.to_table()
    df = table.to_pandas()
    logger.info("EDGAR loaded: %d rows, %d columns", len(df), len(df.columns))
    return df


def _infer_year_from_timestamp(df: pd.DataFrame, time_col: str) -> pd.Series:
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    return ts.dt.year


def _audit_offers_core(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    logger.info("Auditing offers_core...")
    n_rows = len(df)
    n_cols = len(df.columns)
    key_cols = ["platform_name", "offer_id"]
    missing_keys = [c for c in key_cols if c not in df.columns]
    if missing_keys:
        logger.warning("Missing key columns: %s", missing_keys)
    
    key_missing_rate = {}
    for col in key_cols:
        if col in df.columns:
            key_missing_rate[col] = float(df[col].isna().mean())
    
    # Infer year from snapshot_ts or crawled_date
    year_col = None
    if "snapshot_ts" in df.columns:
        year_col = "snapshot_ts"
    elif "crawled_date" in df.columns:
        year_col = "crawled_date"
    
    coverage_by_year = {}
    if year_col:
        years = _infer_year_from_timestamp(df, year_col)
        for year in range(2022, 2027):
            mask = years == year
            coverage_by_year[str(year)] = {
                "rows": int(mask.sum()),
                "unique_entities": int(df.loc[mask, "offer_id"].nunique()) if "offer_id" in df.columns else 0,
            }
    else:
        logger.warning("No timestamp column found to infer year coverage")
    
    return {
        "total_rows": n_rows,
        "total_columns": n_cols,
        "key_columns": key_cols,
        "missing_key_columns": missing_keys,
        "key_missing_rate": key_missing_rate,
        "coverage_by_year": coverage_by_year,
    }


def _audit_edgar(df: pd.DataFrame, logger: logging.Logger) -> Dict[str, Any]:
    logger.info("Auditing EDGAR feature store...")
    n_rows = len(df)
    n_cols = len(df.columns)
    
    # Compute schema hash
    col_names = sorted(df.columns.tolist())
    col_str = ",".join(col_names)
    schema_hash = hashlib.sha256(col_str.encode("utf-8")).hexdigest()
    
    # Coverage by year
    coverage_by_year = {}
    if "snapshot_year" in df.columns:
        for year in range(2022, 2027):
            mask = df["snapshot_year"] == year
            coverage_by_year[str(year)] = {
                "rows": int(mask.sum()),
                "unique_entities": int(df.loc[mask, "offer_id"].nunique()) if "offer_id" in df.columns else 0,
            }
    else:
        logger.warning("No snapshot_year column in EDGAR; cannot compute year coverage")
    
    # Top missing columns
    missing_rates = {}
    for col in df.columns:
        rate = float(df[col].isna().mean())
        if rate > 0.5:
            missing_rates[col] = rate
    top_missing = sorted(missing_rates.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "total_rows": n_rows,
        "total_columns": n_cols,
        "schema_hash": schema_hash,
        "coverage_by_year": coverage_by_year,
        "top_missing_columns": dict(top_missing),
    }


def _audit_join_validity(
    offers_df: pd.DataFrame,
    edgar_df: pd.DataFrame,
    sample_size: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    logger.info("Auditing join validity with sample_size=%d", sample_size)
    if "offer_id" not in offers_df.columns or "offer_id" not in edgar_df.columns:
        logger.warning("Missing offer_id in one or both datasets; skipping join audit")
        return {"status": "skipped", "reason": "missing_offer_id"}
    
    unique_offers = offers_df["offer_id"].dropna().unique()
    if len(unique_offers) == 0:
        logger.warning("No valid offer_id in offers_core")
        return {"status": "skipped", "reason": "no_valid_offer_id"}
    
    sample_ids = np.random.choice(unique_offers, size=min(sample_size, len(unique_offers)), replace=False)
    edgar_ids = set(edgar_df["offer_id"].dropna().unique())
    
    matched = sum(1 for oid in sample_ids if oid in edgar_ids)
    match_rate = matched / len(sample_ids) if len(sample_ids) > 0 else 0.0
    
    logger.info("Join match rate: %.4f (%d/%d)", match_rate, matched, len(sample_ids))
    
    # Diagnosis
    if match_rate < 0.01:
        diagnosis = "Very low match rate suggests join key mismatch or time alignment issue."
    elif match_rate < 0.1:
        diagnosis = "Low match rate may indicate sparse EDGAR coverage or partial key overlap."
    else:
        diagnosis = "Match rate is acceptable; EDGAR coverage appears valid."
    
    return {
        "sample_size": len(sample_ids),
        "matched": matched,
        "match_rate": match_rate,
        "diagnosis": diagnosis,
    }


def _audit_label_health(bench_dirs: List[Path], logger: logging.Logger) -> Dict[str, Any]:
    logger.info("Auditing label health across %d benchmarks", len(bench_dirs))
    all_stats = {"train": [], "val": [], "test": []}
    
    for bench_dir in bench_dirs:
        metrics_path = bench_dir / "metrics.json"
        if not metrics_path.exists():
            logger.warning("Missing metrics.json in %s", bench_dir)
            continue
        
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for result in data.get("results", []):
            for split in ["train", "val", "test"]:
                split_key = f"{split}_metrics"
                if split_key in result:
                    y_vals = result[split_key].get("y_values", [])
                    if y_vals:
                        all_stats[split].extend(y_vals)
    
    summary = {}
    for split, vals in all_stats.items():
        if vals:
            arr = np.array(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                summary[split] = {
                    "count": len(arr),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "p99": float(np.percentile(arr, 99)),
                    "max": float(np.max(arr)),
                }
            else:
                summary[split] = {"count": 0}
        else:
            summary[split] = {"count": 0}
    
    # Risk assessment
    risk = []
    if summary.get("train", {}).get("max", 0) > 100 and summary.get("test", {}).get("max", 0) < 10:
        risk.append("Severe train-test distribution shift detected (train_max >> test_max).")
    if summary.get("train", {}).get("p99", 0) > 1000:
        risk.append("Extreme outliers in train labels; consider log1p or clipping.")
    
    risk_conclusion = " ".join(risk) if risk else "Label distributions appear stable across splits."
    
    return {
        "label_stats": summary,
        "risk_conclusion": risk_conclusion,
    }


def _generate_gate(report: Dict[str, Any], logger: logging.Logger) -> str:
    logger.info("Generating PASS/FAIL gate...")
    failures = []
    
    # Check offers_core
    offers = report.get("offers_core", {})
    if offers.get("total_rows", 0) == 0:
        failures.append("offers_core is empty")
    if offers.get("missing_key_columns"):
        failures.append(f"Missing key columns: {offers['missing_key_columns']}")
    
    coverage = offers.get("coverage_by_year", {})
    for year in ["2022", "2023", "2024", "2025", "2026"]:
        if coverage.get(year, {}).get("rows", 0) == 0:
            failures.append(f"No offers_core coverage for year {year}")
    
    # Check EDGAR
    edgar = report.get("edgar", {})
    if edgar.get("total_rows", 0) == 0:
        failures.append("EDGAR feature store is empty")
    
    edgar_coverage = edgar.get("coverage_by_year", {})
    for year in ["2022", "2023", "2024", "2025", "2026"]:
        if edgar_coverage.get(year, {}).get("rows", 0) == 0:
            failures.append(f"No EDGAR coverage for year {year}")
    
    # Check join
    join = report.get("join_validity", {})
    if join.get("match_rate", 0) < 0.001:
        failures.append("Join match rate < 0.1%; likely key/time mismatch")
    
    # Check labels
    labels = report.get("label_health", {})
    if "Severe train-test distribution shift" in labels.get("risk_conclusion", ""):
        failures.append("Severe train-test label distribution shift")
    
    if failures:
        logger.warning("GATE FAILED: %s", "; ".join(failures))
        return "FAIL"
    else:
        logger.info("GATE PASSED")
        return "PASS"


def _generate_markdown(report: Dict[str, Any], output_path: Path, logger: logging.Logger) -> None:
    logger.info("Generating markdown report...")
    lines = [
        "# Data Integrity Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Offers Core Coverage",
        "",
        f"- Total rows: {report['offers_core']['total_rows']:,}",
        f"- Total columns: {report['offers_core']['total_columns']}",
        f"- Key columns: {', '.join(report['offers_core']['key_columns'])}",
        f"- Missing key columns: {', '.join(report['offers_core']['missing_key_columns']) or 'None'}",
        "",
        "### Coverage by Year",
        "",
        "| Year | Rows | Unique Entities |",
        "| --- | --- | --- |",
    ]
    
    for year, stats in sorted(report["offers_core"]["coverage_by_year"].items()):
        lines.append(f"| {year} | {stats['rows']:,} | {stats['unique_entities']:,} |")
    
    lines.extend([
        "",
        "## 2. EDGAR Feature Store Coverage",
        "",
        f"- Total rows: {report['edgar']['total_rows']:,}",
        f"- Total columns: {report['edgar']['total_columns']}",
        f"- Schema hash: `{report['edgar']['schema_hash'][:16]}...`",
        "",
        "### Coverage by Year",
        "",
        "| Year | Rows | Unique Entities |",
        "| --- | --- | --- |",
    ])
    
    for year, stats in sorted(report["edgar"]["coverage_by_year"].items()):
        lines.append(f"| {year} | {stats['rows']:,} | {stats['unique_entities']:,} |")
    
    lines.extend([
        "",
        "### Top Missing Columns (>50% missing)",
        "",
    ])
    
    top_missing = report["edgar"].get("top_missing_columns", {})
    if top_missing:
        for col, rate in list(top_missing.items())[:5]:
            lines.append(f"- `{col}`: {rate:.2%}")
    else:
        lines.append("- None")
    
    lines.extend([
        "",
        "## 3. Join Validity",
        "",
        f"- Sample size: {report['join_validity']['sample_size']}",
        f"- Matched: {report['join_validity']['matched']}",
        f"- Match rate: {report['join_validity']['match_rate']:.4f}",
        f"- Diagnosis: {report['join_validity']['diagnosis']}",
        "",
        "## 4. Label Health",
        "",
    ])
    
    for split, stats in report["label_health"]["label_stats"].items():
        if stats.get("count", 0) > 0:
            lines.extend([
                f"### {split.capitalize()}",
                "",
                f"- Count: {stats['count']:,}",
                f"- Mean: {stats['mean']:.4f}",
                f"- Std: {stats['std']:.4f}",
                f"- P99: {stats['p99']:.4f}",
                f"- Max: {stats['max']:.4f}",
                "",
            ])
    
    lines.extend([
        f"**Risk Conclusion:** {report['label_health']['risk_conclusion']}",
        "",
        "## 5. Gate Decision",
        "",
        f"**Status:** {report['gate']}",
        "",
    ])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report written to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit processed artifacts for data integrity.")
    parser.add_argument("--offers_core", type=Path, required=True, help="Path to offers_core.parquet")
    parser.add_argument("--edgar_dir", type=Path, required=True, help="Path to EDGAR feature store directory")
    parser.add_argument("--bench_list", type=Path, required=True, help="Path to bench_dirs_all.txt")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for reports")
    parser.add_argument("--sample_size", type=int, default=200, help="Sample size for join validity check")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(args.output_dir)
    
    logger.info("=== Data Integrity Audit Start ===")
    logger.info("offers_core: %s", args.offers_core)
    logger.info("edgar_dir: %s", args.edgar_dir)
    logger.info("bench_list: %s", args.bench_list)
    logger.info("output_dir: %s", args.output_dir)
    
    # Load data
    offers_df = _load_offers_core(args.offers_core, logger)
    edgar_df = _load_edgar_dataset(args.edgar_dir, logger)
    
    bench_dirs = []
    if args.bench_list.exists():
        with open(args.bench_list, "r", encoding="utf-8") as f:
            bench_dirs = [Path(line.strip()) for line in f if line.strip()]
    logger.info("Loaded %d benchmark directories", len(bench_dirs))
    
    # Audit
    report = {
        "offers_core": _audit_offers_core(offers_df, logger),
        "edgar": _audit_edgar(edgar_df, logger),
        "join_validity": _audit_join_validity(offers_df, edgar_df, args.sample_size, logger),
        "label_health": _audit_label_health(bench_dirs, logger),
    }
    
    report["gate"] = _generate_gate(report, logger)
    
    # Write outputs
    json_path = args.output_dir / "data_integrity_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("JSON report written to %s", json_path)
    
    md_path = args.output_dir / "data_integrity_report.md"
    _generate_markdown(report, md_path, logger)
    
    logger.info("=== Data Integrity Audit Complete ===")
    
    if report["gate"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
