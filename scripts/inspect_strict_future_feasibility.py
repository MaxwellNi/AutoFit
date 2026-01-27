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


def _safe_ratio(raised: np.ndarray, goal: np.ndarray) -> np.ndarray:
    ratio = np.full_like(raised, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.isfinite(raised) & np.isfinite(goal) & (goal != 0)
        ratio[mask] = raised[mask] / goal[mask]
    return ratio


def _evaluate_horizon(
    group: pd.DataFrame,
    horizon: int,
    *,
    static_ratio_tol: float,
    min_label_delta_days: float,
    min_ratio_delta_rel: float,
) -> Tuple[Dict[str, int], Dict[str, int], List[float], List[float], List[float]]:
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
    input_keep_values: List[float] = []
    label_keep_values: List[float] = []

    if n == 0:
        return counts, drops_sf1, delta_days_list

    idx = np.arange(n)
    label_idx = idx + horizon
    insufficient = label_idx >= n
    drops_sf1["insufficient_future"] = int(insufficient.sum())
    drops_sf0["insufficient_future"] = int(insufficient.sum())

    label_idx_sf0 = np.minimum(label_idx, n - 1)

    goals = group["funding_goal_usd"].to_numpy(dtype=float)
    raised = group["funding_raised_usd"].to_numpy(dtype=float)
    ts = pd.to_datetime(group["snapshot_ts"], errors="coerce", utc=True).to_numpy(dtype="datetime64[ns]")

    # strict_future=1 valid indices
    valid_idx = idx[~insufficient]
    label_idx_valid = label_idx[~insufficient]
    if valid_idx.size:
        input_ratio = _safe_ratio(raised[valid_idx], goals[valid_idx])
        label_ratio = _safe_ratio(raised[label_idx_valid], goals[label_idx_valid])

        valid_ratio = np.isfinite(input_ratio) & np.isfinite(label_ratio)
        static_mask = valid_ratio & (np.abs(label_ratio - input_ratio) < static_ratio_tol)
        drops_sf1["static_ratio"] = int(static_mask.sum())

        delta_days = (ts[label_idx_valid] - ts[valid_idx]) / np.timedelta64(1, "D")
        ts_bad = ~np.isfinite(delta_days) | (delta_days <= 0)
        drops_sf1["ts_order_bad"] = int((~static_mask & ts_bad).sum())

        delta_days_valid = delta_days[~ts_bad]
        if delta_days_valid.size:
            delta_days_list.extend(delta_days_valid.astype(float).tolist())

        keep = valid_ratio & ~static_mask & ~ts_bad
        if min_label_delta_days > 0:
            min_delta_mask = delta_days < min_label_delta_days
            drops_sf1["min_delta_days"] = int((keep & min_delta_mask).sum())
            keep = keep & ~min_delta_mask

        if min_ratio_delta_rel > 0:
            delta_abs = np.abs(label_ratio - input_ratio)
            delta_rel = delta_abs / np.maximum(1.0, np.abs(input_ratio))
            min_ratio_mask = np.isfinite(delta_rel) & (delta_rel < min_ratio_delta_rel)
            drops_sf1["min_ratio_delta_rel"] = int((keep & min_ratio_mask).sum())
            keep = keep & ~min_ratio_mask

        counts["strict_future_1"] = int(keep.sum())
        if keep.any():
            input_keep_values.extend(input_ratio[keep].astype(float).tolist())
            label_keep_values.extend(label_ratio[keep].astype(float).tolist())

    # strict_future=0 (label clamped)
    input_ratio0 = _safe_ratio(raised, goals)
    label_ratio0 = _safe_ratio(raised[label_idx_sf0], goals[label_idx_sf0])
    valid_ratio0 = np.isfinite(input_ratio0) & np.isfinite(label_ratio0)
    static_mask0 = valid_ratio0 & (np.abs(label_ratio0 - input_ratio0) < static_ratio_tol)
    drops_sf0["static_ratio"] = int(static_mask0.sum())

    delta_days0 = (ts[label_idx_sf0] - ts[idx]) / np.timedelta64(1, "D")
    ts_bad0 = ~np.isfinite(delta_days0) | (delta_days0 <= 0)
    drops_sf0["ts_order_bad"] = int((~static_mask0 & ts_bad0).sum())

    keep0 = valid_ratio0 & ~static_mask0 & ~ts_bad0
    if min_label_delta_days > 0:
        min_delta_mask0 = delta_days0 < min_label_delta_days
        drops_sf0["min_delta_days"] = int((keep0 & min_delta_mask0).sum())
        keep0 = keep0 & ~min_delta_mask0

    if min_ratio_delta_rel > 0:
        delta_abs0 = np.abs(label_ratio0 - input_ratio0)
        delta_rel0 = delta_abs0 / np.maximum(1.0, np.abs(input_ratio0))
        min_ratio_mask0 = np.isfinite(delta_rel0) & (delta_rel0 < min_ratio_delta_rel)
        drops_sf0["min_ratio_delta_rel"] = int((keep0 & min_ratio_mask0).sum())
        keep0 = keep0 & ~min_ratio_mask0

    counts["strict_future_0"] = int(keep0.sum())

    return counts, drops_sf1, delta_days_list, input_keep_values, label_keep_values


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
        input_keep_all: List[float] = []
        label_keep_all: List[float] = []

        for _, group in df.groupby("entity_id", sort=False):
            group = group.reset_index(drop=True)
            counts, drops, delta_days_list, input_keep, label_keep = _evaluate_horizon(
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
            input_keep_all.extend(input_keep)
            label_keep_all.extend(label_keep)

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

        naive_rmse = None
        naive_std_y = None
        naive_near_perfect = None
        if label_keep_all:
            label_arr = np.asarray(label_keep_all, dtype=float)
            input_arr = np.asarray(input_keep_all, dtype=float)
            naive_std_y = float(np.std(label_arr)) if label_arr.size else None
            if label_arr.size:
                naive_rmse = float(np.sqrt(np.mean((label_arr - input_arr) ** 2)))
            if naive_std_y is not None and np.isfinite(naive_std_y):
                if naive_std_y == 0.0:
                    naive_near_perfect = True
                elif naive_rmse is not None:
                    naive_near_perfect = bool(naive_rmse <= 0.05 * naive_std_y)

        horizon_reports.append(
            {
                "horizon": horizon,
                "strict_future_0_samples": int(counts_sf0),
                "strict_future_1_samples": int(counts_sf1),
                "drops_strict_future_1": drops_sf1,
                "delta_days": delta_stats,
                "naive_progress_rmse": naive_rmse,
                "naive_progress_std_y": naive_std_y,
                "naive_progress_near_perfect": naive_near_perfect,
            }
        )

    pct_threshold = 0.05
    recommended_horizon = None
    reason = None
    candidates: List[Dict[str, Any]] = []
    for rep in sorted(horizon_reports, key=lambda r: r["horizon"]):
        samples = rep["strict_future_1_samples"]
        pct_lt = rep["delta_days"]["pct_delta_days_lt_min"]
        if samples >= 500 and (pct_lt is None or pct_lt <= pct_threshold):
            candidates.append(rep)
    non_trivial = [r for r in candidates if r.get("naive_progress_near_perfect") is False]
    if non_trivial:
        recommended_horizon = min(non_trivial, key=lambda r: r["horizon"])["horizon"]
        reason = "min_samples>=500, pct_delta_days_lt_min<=0.05, naive_progress_not_near_perfect"
    elif candidates:
        reason = "all candidates near-perfect; expand horizons or tighten min_ratio_delta_rel or change label"
    if recommended_horizon is None:
        reason = reason or f"no horizon met samples>=500 and pct_delta_days_lt_min<={pct_threshold}"

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
            "rule": "min_samples>=500, pct_delta_days_lt_min<=0.05, naive_progress_not_near_perfect",
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
        f.write("| horizon | sf0_samples | sf1_samples | drops_sf1_insufficient | drops_sf1_static | drops_sf1_min_delta | drops_sf1_min_ratio_rel | drops_sf1_ts_bad | delta_days_median | pct_delta_days_lt_min | naive_rmse | std_y | near_perfect |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for rep in report["horizons"]:
            drops = rep["drops_strict_future_1"]
            delta = rep["delta_days"]
            f.write(
                "| {h} | {sf0} | {sf1} | {ins} | {static} | {min_d} | {min_r} | {ts} | {med} | {pct} | {rmse} | {std} | {np} |\n".format(
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
                    rmse=rep.get("naive_progress_rmse"),
                    std=rep.get("naive_progress_std_y"),
                    np=rep.get("naive_progress_near_perfect"),
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
