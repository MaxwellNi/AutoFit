#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _parse_horizons(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    horizons: List[int] = []
    for p in parts:
        try:
            horizons.append(int(p))
        except ValueError:
            raise ValueError(f"invalid horizon value: {p}")
    if not horizons:
        raise ValueError("no horizons provided")
    return horizons


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _entity_stats(counts: Iterable[int]) -> Dict[str, Any]:
    arr = np.asarray(list(counts), dtype=float)
    if arr.size == 0:
        return {"min": None, "median": None, "p90": None, "max": None}
    return {
        "min": int(np.min(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(np.max(arr)),
    }


def _merge_counts(target: Dict[str, int], src: Dict[str, int]) -> None:
    for k, v in src.items():
        target[k] = int(target.get(k, 0)) + int(v)


def _evaluate_pair(
    input_row: pd.Series,
    label_row: pd.Series,
    *,
    static_ratio_tol: float,
    min_label_delta_days: float,
    min_ratio_delta_rel: float,
) -> Tuple[Dict[str, int], bool, float | None]:
    drops = {
        "static_ratio": 0,
        "min_delta_days": 0,
        "min_ratio_delta_rel": 0,
        "ts_order_bad": 0,
    }

    input_goal = float(input_row.get("funding_goal_usd", np.nan))
    input_raised = float(input_row.get("funding_raised_usd", np.nan))
    label_goal = float(label_row.get("funding_goal_usd", np.nan))
    label_raised = float(label_row.get("funding_raised_usd", np.nan))

    with np.errstate(divide="ignore", invalid="ignore"):
        input_ratio = input_raised / input_goal if input_goal else np.nan
        label_ratio = label_raised / label_goal if label_goal else np.nan

    if np.isfinite(label_ratio) and np.isfinite(input_ratio):
        if abs(label_ratio - input_ratio) < static_ratio_tol:
            drops["static_ratio"] += 1
            return drops, False, None

    input_ts = input_row.get("snapshot_ts")
    label_ts = label_row.get("snapshot_ts")
    if pd.isna(input_ts) or pd.isna(label_ts):
        drops["ts_order_bad"] += 1
        return drops, False, None
    delta_days = (label_ts - input_ts).total_seconds() / 86400.0
    if delta_days <= 0:
        drops["ts_order_bad"] += 1
        return drops, False, None
    if min_label_delta_days > 0 and delta_days < min_label_delta_days:
        drops["min_delta_days"] += 1
        return drops, False, delta_days

    if np.isfinite(label_ratio) and np.isfinite(input_ratio):
        delta_abs = abs(label_ratio - input_ratio)
        delta_rel = delta_abs / max(1.0, abs(input_ratio))
        if min_ratio_delta_rel > 0 and delta_rel < min_ratio_delta_rel:
            drops["min_ratio_delta_rel"] += 1
            return drops, False, delta_days

    return drops, True, delta_days


def _evaluate_horizon(
    group: pd.DataFrame,
    horizon: int,
    *,
    static_ratio_tol: float,
    min_label_delta_days: float,
    min_ratio_delta_rel: float,
) -> Tuple[Dict[str, int], Dict[str, int], List[float]]:
    n = len(group)
    drops_sf1 = {
        "insufficient_future": 0,
        "static_ratio": 0,
        "min_delta_days": 0,
        "min_ratio_delta_rel": 0,
        "ts_order_bad": 0,
    }
    drops_sf0 = {
        "insufficient_future": 0,
        "static_ratio": 0,
        "min_delta_days": 0,
        "min_ratio_delta_rel": 0,
        "ts_order_bad": 0,
    }
    counts = {"strict_future_0": 0, "strict_future_1": 0}
    delta_days_list: List[float] = []

    for input_end in range(n):
        label_idx = input_end + horizon
        insufficient = label_idx >= n
        if insufficient:
            drops_sf1["insufficient_future"] += 1
            drops_sf0["insufficient_future"] += 1
            label_idx_sf0 = n - 1
        else:
            label_idx_sf0 = label_idx

        # strict_future=1
        if not insufficient:
            input_row = group.iloc[input_end]
            label_row = group.iloc[label_idx]
            drops, ok, delta_days = _evaluate_pair(
                input_row,
                label_row,
                static_ratio_tol=static_ratio_tol,
                min_label_delta_days=min_label_delta_days,
                min_ratio_delta_rel=min_ratio_delta_rel,
            )
            if ok:
                counts["strict_future_1"] += 1
                if delta_days is not None:
                    delta_days_list.append(delta_days)
            else:
                _merge_counts(drops_sf1, drops)

        # strict_future=0 (label clamped to last if insufficient)
        input_row = group.iloc[input_end]
        label_row = group.iloc[label_idx_sf0]
        drops, ok, _ = _evaluate_pair(
            input_row,
            label_row,
            static_ratio_tol=static_ratio_tol,
            min_label_delta_days=min_label_delta_days,
            min_ratio_delta_rel=min_ratio_delta_rel,
        )
        if ok:
            counts["strict_future_0"] += 1
        else:
            _merge_counts(drops_sf0, drops)

    return counts, drops_sf1, delta_days_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect strict_future feasibility for horizons.")
    parser.add_argument("--offers_core", required=True)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--horizons", type=str, default="14,21,30,45")
    parser.add_argument("--label_goal_min", type=float, default=50.0)
    parser.add_argument("--min_label_delta_days", type=float, default=1.0)
    parser.add_argument("--min_ratio_delta_rel", type=float, default=1e-4)
    parser.add_argument("--static_ratio_tol", type=float, default=1e-6)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    offers_path = Path(args.offers_core)
    file_stat = offers_path.stat()

    horizons = _parse_horizons(args.horizons)

    use_cols = [
        "entity_id",
        "snapshot_ts",
        "funding_goal_usd",
        "funding_raised_usd",
        "investors_count",
    ]
    df = pd.read_parquet(offers_path, columns=use_cols)
    missing = {c for c in ["entity_id", "snapshot_ts", "funding_goal_usd", "funding_raised_usd"] if c not in df.columns}
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")

    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)
    df = df[df["funding_goal_usd"] >= args.label_goal_min].copy()
    df = df.sort_values(["entity_id", "snapshot_ts"], kind="stable")

    entity_counts = df.groupby("entity_id", sort=False).size().tolist()
    entity_stats = _entity_stats(entity_counts)

    horizon_reports: List[Dict[str, Any]] = []
    for horizon in horizons:
        counts_sf0 = 0
        counts_sf1 = 0
        drops_sf1 = {
            "insufficient_future": 0,
            "static_ratio": 0,
            "min_delta_days": 0,
            "min_ratio_delta_rel": 0,
            "ts_order_bad": 0,
        }
        delta_days_all: List[float] = []

        for _, group in df.groupby("entity_id", sort=False):
            group = group.reset_index(drop=True)
            counts, drops, delta_days_list = _evaluate_horizon(
                group,
                horizon,
                static_ratio_tol=args.static_ratio_tol,
                min_label_delta_days=args.min_label_delta_days,
                min_ratio_delta_rel=args.min_ratio_delta_rel,
            )
            counts_sf0 += counts["strict_future_0"]
            counts_sf1 += counts["strict_future_1"]
            _merge_counts(drops_sf1, drops)
            delta_days_all.extend(delta_days_list)

        delta_stats = {
            "median": None,
            "p10": None,
            "p90": None,
            "pct_delta_days_lt_min": None,
        }
        if delta_days_all:
            arr = np.asarray(delta_days_all, dtype=float)
            delta_stats = {
                "median": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
                "pct_delta_days_lt_min": float((arr < args.min_label_delta_days).mean()),
            }

        horizon_reports.append(
            {
                "horizon": horizon,
                "strict_future_0_samples": int(counts_sf0),
                "strict_future_1_samples": int(counts_sf1),
                "drops_strict_future_1": drops_sf1,
                "delta_days": delta_stats,
            }
        )

    recommended_horizon = None
    reason = None
    for rep in sorted(horizon_reports, key=lambda r: r["horizon"]):
        samples = rep["strict_future_1_samples"]
        pct_lt = rep["delta_days"]["pct_delta_days_lt_min"]
        if samples >= 500 and (pct_lt is None or pct_lt <= 0.01):
            recommended_horizon = rep["horizon"]
            reason = "min_samples>=500 and pct_delta_days_lt_min<=0.01"
            break
    if recommended_horizon is None:
        reason = "no horizon met samples>=500 and pct_delta_days_lt_min<=0.01"

    report = {
        "offers_core": str(offers_path),
        "offers_core_size_bytes": int(file_stat.st_size),
        "offers_core_sha256": _sha256(offers_path),
        "n_rows": int(len(df)),
        "n_entities": int(df["entity_id"].nunique()),
        "snapshot_count_stats": entity_stats,
        "args": vars(args),
        "horizons": horizon_reports,
        "recommendation": {
            "recommended_horizon": recommended_horizon,
            "rule": "strict_future_1_samples>=500 and pct_delta_days_lt_min<=0.01",
            "reason": reason,
        },
    }

    json_path = out_dir / "feasibility_report.json"
    md_path = out_dir / "feasibility_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# strict_future feasibility report\n\n")
        f.write(f"offers_core: `{offers_path}`\n\n")
        f.write(f"rows: {report['n_rows']} entities: {report['n_entities']}\n\n")
        f.write("## Snapshot count per entity\n\n")
        f.write(f"- min: {entity_stats['min']}\n")
        f.write(f"- median: {entity_stats['median']}\n")
        f.write(f"- p90: {entity_stats['p90']}\n")
        f.write(f"- max: {entity_stats['max']}\n\n")
        f.write("## Horizon feasibility\n\n")
        f.write("| horizon | sf0_samples | sf1_samples | drops_sf1_insufficient | drops_sf1_static | drops_sf1_min_delta | drops_sf1_min_ratio_rel | drops_sf1_ts_bad | delta_days_median | pct_delta_days_lt_min |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for rep in report["horizons"]:
            drops = rep["drops_strict_future_1"]
            delta = rep["delta_days"]
            f.write(
                "| {h} | {sf0} | {sf1} | {ins} | {static} | {min_d} | {min_r} | {ts} | {med} | {pct} |\n".format(
                    h=rep["horizon"],
                    sf0=rep["strict_future_0_samples"],
                    sf1=rep["strict_future_1_samples"],
                    ins=drops["insufficient_future"],
                    static=drops["static_ratio"],
                    min_d=drops["min_delta_days"],
                    min_r=drops["min_ratio_delta_rel"],
                    ts=drops["ts_order_bad"],
                    med=delta["median"],
                    pct=delta["pct_delta_days_lt_min"],
                )
            )
        f.write("\n## Recommendation\n\n")
        f.write(f"recommended_horizon: {report['recommendation']['recommended_horizon']}\n\n")
        f.write(f"reason: {report['recommendation']['reason']}\n")

    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "offers_core": str(offers_path),
        "offers_core_sha256": report["offers_core_sha256"],
        "offers_core_size_bytes": report["offers_core_size_bytes"],
        "args": vars(args),
        "outputs": {
            "json": str(json_path),
            "md": str(md_path),
        },
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
