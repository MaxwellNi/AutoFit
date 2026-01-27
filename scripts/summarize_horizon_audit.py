#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return None


def _alignment_index(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in report.get("runs", []):
        bench_dir = item.get("bench_dir")
        if bench_dir:
            indexed[str(bench_dir)] = item
    return indexed


def _sanity_index(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in report.get("benchmarks", []):
        bench_dir = item.get("bench_dir")
        if bench_dir:
            indexed[str(bench_dir)] = item
    return indexed


def _bench_dirs_from_alignment(alignment_report: Dict[str, Any]) -> List[str]:
    bench_dirs: List[str] = []
    for item in alignment_report.get("runs", []):
        if item.get("bench_dir"):
            bench_dirs.append(str(item["bench_dir"]))
    return bench_dirs


def _compute_near_perfect(naive_progress_rmse: Optional[float], std_y: Optional[float], ratio: float) -> Optional[bool]:
    if naive_progress_rmse is None or std_y is None:
        return None
    if not (isinstance(naive_progress_rmse, (int, float)) and isinstance(std_y, (int, float))):
        return None
    if not (std_y > 0):
        return None
    return bool(naive_progress_rmse < ratio * std_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize horizon learnability audit.")
    parser.add_argument("--sanity_report", type=Path, required=True)
    parser.add_argument("--alignment_audit", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--near_perfect_ratio", type=float, default=0.05)
    args = parser.parse_args()

    sanity_report = _load_json(args.sanity_report)
    alignment_report = _load_json(args.alignment_audit)

    sanity_index = _sanity_index(sanity_report)
    alignment_index = _alignment_index(alignment_report)

    bench_dirs = _bench_dirs_from_alignment(alignment_report)
    runs: List[Dict[str, Any]] = []
    for bench_dir in bench_dirs:
        cfg = _load_yaml(Path(bench_dir) / "configs" / "resolved_config.yaml")
        sanity = sanity_index.get(bench_dir, {})
        alignment = alignment_index.get(bench_dir, {})

        y_stats = sanity.get("y_stats", {})
        std_y = _to_float(y_stats.get("std")) if isinstance(y_stats, dict) else None

        naive_progress_rmse = None
        n_eval = None
        if sanity.get("naive_progress_baseline"):
            naive_progress_rmse = _to_float(sanity["naive_progress_baseline"].get("rmse"))
            n_eval = _to_int(sanity["naive_progress_baseline"].get("n_eval"))
        if n_eval is None and y_stats:
            n_eval = _to_int(y_stats.get("count"))

        best_rmse = None
        if sanity.get("best_model"):
            best_rmse = _to_float(sanity["best_model"].get("rmse"))

        warnings = sanity.get("warnings", []) or []
        strong_warning = any("STRONG WARNING" in str(w) for w in warnings)

        alignment_metrics = alignment.get("alignment", {}) if alignment else {}
        time_alignment = alignment.get("time_alignment", {}) if alignment else {}

        run_entry = {
            "bench_dir": str(bench_dir),
            "label_horizon": _to_int(cfg.get("label_horizon")),
            "use_edgar": _to_bool(cfg.get("use_edgar")),
            "limit_rows": _to_int(cfg.get("limit_rows")),
            "label_goal_min": _to_float(cfg.get("label_goal_min")),
            "dropped_due_to_static_ratio": _to_int(cfg.get("dropped_due_to_static_ratio")),
            "drop_counts": {
                "dropped_due_to_static_ratio": _to_int(cfg.get("dropped_due_to_static_ratio")),
                "dropped_due_to_min_delta_days": _to_int(cfg.get("dropped_due_to_min_delta_days")),
                "dropped_due_to_small_ratio_delta_abs": _to_int(cfg.get("dropped_due_to_small_ratio_delta_abs")),
                "dropped_due_to_small_ratio_delta_rel": _to_int(cfg.get("dropped_due_to_small_ratio_delta_rel")),
            },
            "n_eval": n_eval,
            "std_y": std_y,
            "naive_progress_rmse": naive_progress_rmse,
            "best_rmse": best_rmse,
            "alignment_corr": _to_float(alignment_metrics.get("corr")),
            "alignment_pct_abs_diff_lt_1e-6": _to_float(alignment_metrics.get("pct_abs_diff_lt_1e-6")),
            "alignment_max_abs_diff": _to_float(alignment_metrics.get("max_abs_diff")),
            "median_delta_days": _to_float(time_alignment.get("median_delta_days")),
            "strong_warning": strong_warning,
        }
        run_entry["near_perfect"] = _compute_near_perfect(
            run_entry["naive_progress_rmse"],
            run_entry["std_y"],
            args.near_perfect_ratio,
        )
        runs.append(run_entry)

    group_stats: Dict[Tuple[int, bool], Dict[str, Any]] = {}
    grouped: Dict[Tuple[int, bool], List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        horizon = run.get("label_horizon") or -1
        edgar = bool(run.get("use_edgar"))
        grouped[(horizon, edgar)].append(run)

    for key, items in grouped.items():
        horizon, edgar = key
        n_runs = len(items)
        n_eval_vals = [i["n_eval"] for i in items if i.get("n_eval") is not None]
        std_vals = [i["std_y"] for i in items if i.get("std_y") is not None]
        naive_vals = [i["naive_progress_rmse"] for i in items if i.get("naive_progress_rmse") is not None]
        best_vals = [i["best_rmse"] for i in items if i.get("best_rmse") is not None]
        align_corr_vals = [i["alignment_corr"] for i in items if i.get("alignment_corr") is not None]
        align_pct_vals = [i["alignment_pct_abs_diff_lt_1e-6"] for i in items if i.get("alignment_pct_abs_diff_lt_1e-6") is not None]
        align_max_vals = [i["alignment_max_abs_diff"] for i in items if i.get("alignment_max_abs_diff") is not None]
        dropped_vals = [i["dropped_due_to_static_ratio"] for i in items if i.get("dropped_due_to_static_ratio") is not None]
        median_delta_vals = [i["median_delta_days"] for i in items if i.get("median_delta_days") is not None]

        near_flags = [i["near_perfect"] for i in items if i.get("near_perfect") is not None]
        strong_flags = [i["strong_warning"] for i in items if i.get("strong_warning") is not None]

        group_stats[key] = {
            "label_horizon": horizon,
            "use_edgar": edgar,
            "n_runs": n_runs,
            "n_eval_mean": float(sum(n_eval_vals) / len(n_eval_vals)) if n_eval_vals else None,
            "dropped_static_mean": float(sum(dropped_vals) / len(dropped_vals)) if dropped_vals else None,
            "std_y_mean": float(sum(std_vals) / len(std_vals)) if std_vals else None,
            "naive_progress_rmse_mean": float(sum(naive_vals) / len(naive_vals)) if naive_vals else None,
            "best_rmse_mean": float(sum(best_vals) / len(best_vals)) if best_vals else None,
            "alignment_corr_mean": float(sum(align_corr_vals) / len(align_corr_vals)) if align_corr_vals else None,
            "alignment_pct_abs_diff_lt_1e-6_mean": float(sum(align_pct_vals) / len(align_pct_vals)) if align_pct_vals else None,
            "alignment_max_abs_diff_mean": float(sum(align_max_vals) / len(align_max_vals)) if align_max_vals else None,
            "median_delta_days_mean": float(sum(median_delta_vals) / len(median_delta_vals)) if median_delta_vals else None,
            "near_perfect_rate": float(sum(1 for v in near_flags if v) / len(near_flags)) if near_flags else None,
            "strong_warning_rate": float(sum(1 for v in strong_flags if v) / len(strong_flags)) if strong_flags else None,
        }

    horizons = sorted({k[0] for k in group_stats.keys() if k[0] >= 0})
    horizon_summary: Dict[int, Dict[str, Any]] = {}
    for horizon in horizons:
        groups = [v for (h, _), v in group_stats.items() if h == horizon]
        near_rates = [g["near_perfect_rate"] for g in groups if g.get("near_perfect_rate") is not None]
        strong_rates = [g["strong_warning_rate"] for g in groups if g.get("strong_warning_rate") is not None]
        horizon_summary[horizon] = {
            "near_perfect_rate": float(sum(near_rates) / len(near_rates)) if near_rates else None,
            "strong_warning_rate": float(sum(strong_rates) / len(strong_rates)) if strong_rates else None,
        }

    recommendation = "insufficient_data"
    recommended_horizon: Optional[int] = None
    for horizon in horizons:
        summary = horizon_summary[horizon]
        if summary["near_perfect_rate"] == 0.0 and (summary["strong_warning_rate"] in (0.0, None)):
            recommended_horizon = horizon
            break
    if recommended_horizon is not None:
        recommendation = f"recommend_h{recommended_horizon}"
    elif horizons:
        recommendation = f"recommend_increase_horizon_above_{max(horizons)}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "horizon_compare.json"
    md_path = args.output_dir / "horizon_compare.md"

    json_path.write_text(
        json.dumps(
            {
                "near_perfect_rule": f"naive_progress_rmse < {args.near_perfect_ratio} * std_y",
                "runs": runs,
                "group_summary": list(group_stats.values()),
                "horizon_summary": horizon_summary,
                "recommendation": recommendation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Horizon learnability audit\n\n")
        f.write(f"Near-perfect rule: naive_progress_rmse < {args.near_perfect_ratio} * std_y\n\n")
        f.write("| horizon | use_edgar | n_runs | n_eval_mean | dropped_static_mean | std_y_mean | naive_progress_rmse_mean | best_rmse_mean | alignment_corr_mean | alignment_max_abs_diff_mean | median_delta_days_mean | near_perfect_rate | strong_warning_rate |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for key in sorted(group_stats.keys(), key=lambda k: (k[0], k[1])):
            row = group_stats[key]
            f.write(
                "| {horizon} | {edgar} | {n_runs} | {n_eval_mean} | {dropped} | {std_y} | {naive} | {best} | {align_corr} | {align_max} | {median_delta} | {near_rate} | {warn_rate} |\n".format(
                    horizon=row.get("label_horizon"),
                    edgar=row.get("use_edgar"),
                    n_runs=row.get("n_runs"),
                    n_eval_mean=row.get("n_eval_mean"),
                    dropped=row.get("dropped_static_mean"),
                    std_y=row.get("std_y_mean"),
                    naive=row.get("naive_progress_rmse_mean"),
                    best=row.get("best_rmse_mean"),
                    align_corr=row.get("alignment_corr_mean"),
                    align_max=row.get("alignment_max_abs_diff_mean"),
                    median_delta=row.get("median_delta_days_mean"),
                    near_rate=row.get("near_perfect_rate"),
                    warn_rate=row.get("strong_warning_rate"),
                )
            )
        f.write("\n## Recommendation\n\n")
        f.write(f"{recommendation}\n")

    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
