from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("alignment_quality")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _make_entity_key(df: pd.DataFrame, id_cols: Sequence[str]) -> pd.Series:
    return df[id_cols[0]].astype(str).fillna("NA") + "||" + df[id_cols[1]].astype(str).fillna("NA")


def _load_edgar_features(edgar_dir: Path, columns: Sequence[str]) -> pd.DataFrame:
    dataset = ds.dataset(
        str(edgar_dir),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
        ignore_prefixes=["_delta_log"],
    )
    cols = [c for c in columns if c in dataset.schema.names]
    if not cols:
        return pd.DataFrame()
    table = dataset.to_table(columns=cols)
    return table.to_pandas()


def _quantiles(series: pd.Series) -> Dict[str, float]:
    if series.empty:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    return {
        "p50": float(series.quantile(0.5)),
        "p90": float(series.quantile(0.9)),
        "p99": float(series.quantile(0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report alignment quality between offers_core and edgar features")
    parser.add_argument("--offers_core", type=Path, required=True)
    parser.add_argument("--edgar_features", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--snapshot_time_col", type=str, default="crawled_date")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("runs") / "alignment_report" / ts
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / "report_alignment_quality.log"
    logger = _setup_logger(log_path)

    logger.info("report_alignment_quality start")
    logger.info("offers_core=%s", args.offers_core)
    logger.info("edgar_features=%s", args.edgar_features)
    logger.info("output_dir=%s", output_dir)

    offers = pd.read_parquet(args.offers_core)
    if offers.empty:
        raise ValueError("offers_core is empty")

    id_cols = ["platform_name", "offer_id"]
    if not all(c in offers.columns for c in id_cols):
        raise KeyError(f"offers_core missing id columns: {id_cols}")

    offers["entity_id"] = _make_entity_key(offers, id_cols)

    if "snapshot_ts" in offers.columns:
        snapshot_ts = pd.to_datetime(offers["snapshot_ts"], errors="coerce", utc=True)
    else:
        snapshot_ts = pd.to_datetime(offers[args.snapshot_time_col], errors="coerce", utc=True)
        offers["snapshot_ts"] = snapshot_ts

    if "time_delta_days" not in offers.columns:
        offers = offers.sort_values(id_cols + ["snapshot_ts"], kind="mergesort")
        offers["time_delta_days"] = (
            offers.groupby(id_cols, sort=False)["snapshot_ts"]
            .diff()
            .dt.total_seconds()
            .div(86400.0)
            .fillna(0.0)
        )

    # join coverage (three sets: static/snapshots/edgar)
    static_set = set(offers["entity_id"].unique())
    snapshot_set = set(offers["entity_id"].unique())

    edgar_cols = [
        "platform_name",
        "offer_id",
        "cik",
        "edgar_valid",
        args.snapshot_time_col,
    ]
    edgar = _load_edgar_features(args.edgar_features, edgar_cols)
    if not edgar.empty and all(c in edgar.columns for c in id_cols):
        edgar["entity_id"] = _make_entity_key(edgar, id_cols)
        edgar_set = set(edgar["entity_id"].unique())
    else:
        edgar_set = set()

    union_set = static_set | snapshot_set | edgar_set
    n_static = len(static_set)
    n_snapshot = len(snapshot_set)
    n_edgar = len(edgar_set)
    n_union = len(union_set)
    n_static_snap = len(static_set & snapshot_set)
    n_static_edgar = len(static_set & edgar_set)
    n_snapshot_edgar = len(snapshot_set & edgar_set)
    n_all_three = len(static_set & snapshot_set & edgar_set)

    join_cov = {
        "n_static": n_static,
        "n_snapshot": n_snapshot,
        "n_edgar": n_edgar,
        "n_union": n_union,
        "n_static_snap": n_static_snap,
        "n_static_edgar": n_static_edgar,
        "n_snapshot_edgar": n_snapshot_edgar,
        "n_all_three": n_all_three,
        "ratio_all_three_over_union": (n_all_three / n_union) if n_union else 0.0,
        "ratio_all_three_over_static": (n_all_three / n_static) if n_static else 0.0,
        "ratio_edgar_over_static": (n_edgar / n_static) if n_static else 0.0,
    }
    pd.DataFrame([join_cov]).to_parquet(output_dir / "alignment_join_coverage.parquet", index=False)

    logger.info(
        "join coverage 三集合比例: n_static=%d n_snapshot=%d n_edgar=%d n_union=%d n_all_three=%d ratio_all_three_over_union=%.4f ratio_all_three_over_static=%.4f",
        n_static,
        n_snapshot,
        n_edgar,
        n_union,
        n_all_three,
        join_cov["ratio_all_three_over_union"],
        join_cov["ratio_all_three_over_static"],
    )

    # snapshot length
    snapshot_len = offers.groupby(id_cols, sort=False).size().reset_index(name="snapshot_len")
    snapshot_len.to_parquet(output_dir / "snapshot_length.parquet", index=False)
    snap_q = _quantiles(snapshot_len["snapshot_len"])
    logger.info(
        "snapshot_length quantiles: p50=%.3f p90=%.3f p99=%.3f",
        snap_q["p50"],
        snap_q["p90"],
        snap_q["p99"],
    )

    # delta-time stats
    delta = offers["time_delta_days"].dropna()
    delta_positive = delta[delta > 0]
    delta_used = delta_positive if len(delta_positive) else delta
    delta_q = _quantiles(delta_used)

    delta_stats = (
        offers.groupby(id_cols, sort=False)["time_delta_days"]
        .quantile([0.5, 0.9, 0.99])
        .unstack()
        .reset_index()
        .rename(columns={0.5: "p50", 0.9: "p90", 0.99: "p99"})
    )
    delta_stats.to_parquet(output_dir / "delta_time_stats.parquet", index=False)
    logger.info(
        "delta_time quantiles (days, %s): p50=%.6f p90=%.6f p99=%.6f",
        "positive-only" if len(delta_positive) else "all",
        delta_q["p50"],
        delta_q["p90"],
        delta_q["p99"],
    )

    # missingness profile
    total = len(offers)
    miss_rows = []
    for col in offers.columns:
        miss = int(offers[col].isna().sum())
        miss_rows.append(
            {
                "column": col,
                "missing_count": miss,
                "total_count": total,
                "missing_rate": float(miss / total) if total else 0.0,
            }
        )
    missingness = pd.DataFrame(miss_rows).sort_values("missing_rate", ascending=False)
    missingness.to_parquet(output_dir / "missingness_profile.parquet", index=False)

    # edgar_valid by year
    if not edgar.empty and "edgar_valid" in edgar.columns:
        edgar[args.snapshot_time_col] = pd.to_datetime(edgar[args.snapshot_time_col], errors="coerce", utc=True)
        edgar["snapshot_year"] = edgar[args.snapshot_time_col].dt.year
        by_year = (
            edgar.groupby("snapshot_year", dropna=False)["edgar_valid"]
            .agg(total="count", valid="sum")
            .reset_index()
        )
        by_year["valid_rate"] = by_year["valid"] / by_year["total"].replace(0, np.nan)
    else:
        by_year = pd.DataFrame(columns=["snapshot_year", "total", "valid", "valid_rate"])
    by_year.to_parquet(output_dir / "edgar_valid_by_year.parquet", index=False)

    logger.info("edgar_valid by year table:")
    for _, row in by_year.iterrows():
        logger.info(
            "  year=%s total=%d valid=%d valid_rate=%.4f",
            str(int(row["snapshot_year"])) if pd.notna(row["snapshot_year"]) else "NA",
            int(row["total"]),
            int(row["valid"]),
            float(row["valid_rate"]) if pd.notna(row["valid_rate"]) else 0.0,
        )

    logger.info("saved alignment_join_coverage=%s", output_dir / "alignment_join_coverage.parquet")
    logger.info("saved snapshot_length=%s", output_dir / "snapshot_length.parquet")
    logger.info("saved delta_time_stats=%s", output_dir / "delta_time_stats.parquet")
    logger.info("saved missingness_profile=%s", output_dir / "missingness_profile.parquet")
    logger.info("saved edgar_valid_by_year=%s", output_dir / "edgar_valid_by_year.parquet")
    logger.info("report_alignment_quality done")


if __name__ == "__main__":
    main()
