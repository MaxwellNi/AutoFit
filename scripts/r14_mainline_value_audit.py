#!/usr/bin/env python3
"""Audit observed mainline point-accuracy value against non-mainline R14 rows."""

from __future__ import annotations

import glob
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_mainline_value_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(str(ROOT / "runs/benchmarks/r14fcast_*/metrics.json"))):
        try:
            payload = json.load(open(path))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            payload = payload["results"]
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                mae = float(item.get("mae"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(mae):
                continue
            copied = dict(item)
            copied["mae"] = mae
            copied["_run"] = Path(path).parent.name
            copied["_metrics_path"] = str(Path(path).relative_to(ROOT))
            rows.append(copied)
    return rows


def _is_mainline(row: dict[str, Any]) -> bool:
    model = str(row.get("model_name") or row.get("model") or "")
    return row.get("category") == "mainline" or model.startswith("Mainline")


def _cell(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (row.get("task"), row.get("target"), row.get("horizon"), row.get("ablation"))


def main() -> int:
    rows = _rows()
    by_cell: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[_cell(row)].append(row)

    paired = []
    for cell, group in sorted(by_cell.items(), key=lambda item: str(item[0])):
        mainline = [row for row in group if _is_mainline(row)]
        non_mainline = [row for row in group if not _is_mainline(row)]
        if not mainline or not non_mainline:
            continue
        best_mainline = min(mainline, key=lambda row: row["mae"])
        best_non_mainline = min(non_mainline, key=lambda row: row["mae"])
        delta = float(best_non_mainline["mae"] - best_mainline["mae"])
        pct = float(delta / best_non_mainline["mae"] * 100.0) if best_non_mainline["mae"] else None
        paired.append({
            "task": cell[0],
            "target": cell[1],
            "horizon": cell[2],
            "ablation": cell[3],
            "mainline_wins": delta > 1e-9,
            "mae_delta_non_mainline_minus_mainline": delta,
            "mae_delta_pct_vs_best_non_mainline": pct,
            "best_mainline_model": best_mainline.get("model_name") or best_mainline.get("model"),
            "best_mainline_mae": best_mainline["mae"],
            "best_mainline_run": best_mainline.get("_run"),
            "best_non_mainline_model": best_non_mainline.get("model_name") or best_non_mainline.get("model"),
            "best_non_mainline_category": best_non_mainline.get("category"),
            "best_non_mainline_mae": best_non_mainline["mae"],
            "best_non_mainline_run": best_non_mainline.get("_run"),
        })

    pct_values = [row["mae_delta_pct_vs_best_non_mainline"] for row in paired if row["mae_delta_pct_vs_best_non_mainline"] is not None]
    task_summary = {}
    for task in sorted({row["task"] for row in paired}):
        subset = [row for row in paired if row["task"] == task]
        sub_pct = [row["mae_delta_pct_vs_best_non_mainline"] for row in subset if row["mae_delta_pct_vs_best_non_mainline"] is not None]
        task_summary[str(task)] = {
            "n_paired_cells": len(subset),
            "mainline_wins": sum(1 for row in subset if row["mainline_wins"]),
            "mainline_losses": sum(1 for row in subset if not row["mainline_wins"]),
            "mean_delta_pct_vs_best_non_mainline": float(statistics.mean(sub_pct)) if sub_pct else None,
        }

    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "scope": "r14fcast metrics only; exploratory best-observed pairing, not a locked leaderboard",
        "n_rows_scanned": len(rows),
        "n_paired_cells": len(paired),
        "mainline_wins": sum(1 for row in paired if row["mainline_wins"]),
        "mainline_losses": sum(1 for row in paired if not row["mainline_wins"]),
        "mainline_win_rate": (sum(1 for row in paired if row["mainline_wins"]) / len(paired)) if paired else None,
        "mean_delta_pct_vs_best_non_mainline": float(statistics.mean(pct_values)) if pct_values else None,
        "task_summary": task_summary,
        "paired_rows": paired,
        "top_mainline_wins": sorted(paired, key=lambda row: row["mae_delta_pct_vs_best_non_mainline"] or -999.0, reverse=True)[:10],
        "top_mainline_losses": sorted(paired, key=lambda row: row["mae_delta_pct_vs_best_non_mainline"] or 999.0)[:10],
        "interpretation": [
            "Current landed paired evidence does not support a blanket claim that mainline/shared trunk beats SOTA forecasters across heads.",
            "Paired cells are currently task1/funding-heavy; task2/task3 shared-head claims need explicit paired evidence before paper/deployment claims.",
            "Large wins from forced-source variants require separate leakage/activation/fairness review before being treated as deployable gains.",
        ],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text("# R14 Mainline Value Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("scope", "n_rows_scanned", "n_paired_cells", "mainline_wins", "mainline_losses", "mainline_win_rate", "mean_delta_pct_vs_best_non_mainline", "task_summary")}, indent=2, default=str))
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())