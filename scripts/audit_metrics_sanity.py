#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _safe_float(value: object) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    return num


def _summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rmse_vals = []
    mae_vals = []
    r2_vals = []
    best_val_loss_vals = []
    for row in results:
        rmse_vals.append(_safe_float(row.get("rmse")))
        mae_vals.append(_safe_float(row.get("mae")))
        r2_vals.append(_safe_float(row.get("r2")))
        best_val_loss_vals.append(_safe_float(row.get("best_val_loss")))
    rmse_finite = [v for v in rmse_vals if v is not None]
    mae_finite = [v for v in mae_vals if v is not None]
    r2_finite = [v for v in r2_vals if v is not None]
    best_val_finite = [v for v in best_val_loss_vals if v is not None]
    return {
        "results_count": len(results),
        "rmse_finite_count": len(rmse_finite),
        "mae_finite_count": len(mae_finite),
        "r2_finite_count": len(r2_finite),
        "best_val_loss_finite_count": len(best_val_finite),
        "rmse_example": rmse_vals[:3],
        "best_val_loss_example": best_val_loss_vals[:3],
    }


def _bad_reasons(summary: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if summary["results_count"] == 0:
        reasons.append("no_results")
        return reasons
    if summary["rmse_finite_count"] == 0:
        reasons.append("all_rmse_nonfinite")
    if summary["best_val_loss_finite_count"] == 0:
        reasons.append("all_best_val_loss_nonfinite")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit metrics.json for non-finite metrics.")
    parser.add_argument("--bench_list", required=True, help="Path to bench_dirs_all.txt")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    bench_list = Path(args.bench_list)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_dirs = [Path(p) for p in bench_list.read_text(encoding="utf-8").splitlines() if p.strip()]
    report: Dict[str, Any] = {"runs": [], "bad_runs": []}

    for bench_dir in bench_dirs:
        metrics_path = bench_dir / "metrics.json"
        entry: Dict[str, Any] = {"bench_dir": str(bench_dir), "metrics_path": str(metrics_path)}
        if not metrics_path.exists():
            entry["ok"] = False
            entry["errors"] = ["missing_metrics.json"]
            report["runs"].append(entry)
            report["bad_runs"].append(entry)
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        results = metrics.get("results", []) or []
        summary = _summarize_results(results)
        reasons = _bad_reasons(summary)
        entry.update(summary)
        entry["exp_name"] = metrics.get("exp_name")
        entry["ok"] = len(reasons) == 0
        entry["bad_reasons"] = reasons
        report["runs"].append(entry)
        if reasons:
            report["bad_runs"].append(entry)

    json_path = output_dir / "sanity_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Metrics Sanity Report",
        "",
        "| exp_name | rmse_finite | best_val_finite | results | bad_reasons | bench_dir |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["runs"]:
        md_lines.append(
            f"| {row.get('exp_name')} | {row.get('rmse_finite_count')} | "
            f"{row.get('best_val_loss_finite_count')} | {row.get('results_count')} | "
            f"{','.join(row.get('bad_reasons', []))} | {row.get('bench_dir')} |"
        )
    md_path = output_dir / "sanity_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    if report["bad_runs"]:
        failure_path = output_dir / "FAILURE.md"
        failure_path.write_text(
            "# FAILURE\n\nBad runs detected in metrics sanity audit.\n",
            encoding="utf-8",
        )
        raise SystemExit("FATAL: metrics sanity audit found BAD runs.")

    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
