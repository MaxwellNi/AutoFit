#!/usr/bin/env python3
"""Summarize R14 hard-cell tail-level search runs."""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "audits"


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    patterns = [
        "runs/benchmarks/r14fcast_hardcell_matrix_main_*/metrics.json",
        "runs/benchmarks/r14fcast_hardcell_h30_lvl*/metrics.json",
    ]
    for pattern in patterns:
        for path in sorted(glob.glob(str(ROOT / pattern))):
            try:
                payload = json.load(open(path, encoding="utf-8"))
            except Exception:
                continue
            rows = payload.get("results", []) if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                coverage = _finite(row.get("nccopo_coverage_90_hard_cell_tail_guard"))
                if coverage is None:
                    continue
                marginal_width = _finite(row.get("nccopo_interval_width_mean"))
                hard_width = _finite(row.get("nccopo_interval_width_mean_hard_cell_tail_guard"))
                width_ratio = None
                if marginal_width and hard_width is not None and marginal_width > 0:
                    width_ratio = hard_width / marginal_width
                records.append(
                    {
                        "run": Path(path).parent.name,
                        "model": row.get("model_name") or row.get("model"),
                        "target": row.get("target"),
                        "horizon": row.get("horizon"),
                        "ablation": row.get("ablation"),
                        "upper_level": _finite(row.get("nccopo_hard_cell_upper_level")),
                        "coverage": coverage,
                        "marginal_width": marginal_width,
                        "hard_cell_width": hard_width,
                        "width_ratio_vs_marginal": width_ratio,
                        "status": row.get("nccopo_hard_cell_status"),
                        "metrics_path": str(Path(path).relative_to(ROOT)),
                        "mtime": os.path.getmtime(path),
                    }
                )
    return records


def _summarize_group(rows: list[dict[str, Any]], coverage_floor: float, max_width_ratio: float) -> dict[str, Any]:
    coverages = [float(row["coverage"]) for row in rows if row.get("coverage") is not None]
    ratios = [float(row["width_ratio_vs_marginal"]) for row in rows if row.get("width_ratio_vs_marginal") is not None]
    return {
        "n_records": len(rows),
        "n_unique_cells": len({(row.get("target"), row.get("horizon"), row.get("ablation")) for row in rows}),
        "coverage_min": min(coverages) if coverages else None,
        "coverage_mean": sum(coverages) / len(coverages) if coverages else None,
        "width_ratio_max": max(ratios) if ratios else None,
        "width_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "passes_coverage_floor": bool(coverages) and min(coverages) >= coverage_floor,
        "passes_width_guard": bool(ratios) and max(ratios) <= max_width_ratio,
        "records": sorted(rows, key=lambda row: (str(row.get("horizon")), str(row.get("ablation")), -float(row.get("mtime") or 0))),
    }


def main() -> int:
    coverage_floor = 0.88
    max_width_ratio = 3.0
    records = _load_records()
    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("horizon") != 30:
            continue
        level = row.get("upper_level")
        if level is None:
            continue
        by_level[f"{level:.3f}"].append(row)
    summaries = {
        level: _summarize_group(rows, coverage_floor, max_width_ratio)
        for level, rows in sorted(by_level.items())
    }
    eligible = [
        (level, summary)
        for level, summary in summaries.items()
        if summary["passes_coverage_floor"] and summary["passes_width_guard"]
    ]
    best_level = None
    if eligible:
        best_level = min(eligible, key=lambda item: (float(item[0]), item[1]["width_ratio_mean"] or 999.0))[0]
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": "passed" if best_level is not None else "not_passed",
        "scope": "R14 funding h30 hard-cell tail-level search; calibration-only records from formal benchmark runs.",
        "coverage_floor": coverage_floor,
        "max_width_ratio_vs_marginal": max_width_ratio,
        "best_level": best_level,
        "level_summaries": summaries,
        "interpretation": [
            "A level passes only if every landed h30 hard-cell record at that level clears coverage >= 0.88 and width ratio <= 3.0.",
            "This audit does not prove point-forecast improvement; it only selects the least conservative landed calibration setting under the width guard.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"r14_hardcell_level_search_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = out_json.with_suffix(".md")
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out_md.write_text(
        "# R14 Hard-Cell Level Search Summary\n\n```json\n"
        + json.dumps(report, indent=2, ensure_ascii=False, default=str)
        + "\n```\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "best_level": best_level,
                "levels": {
                    level: {
                        key: summary.get(key)
                        for key in ("n_records", "n_unique_cells", "coverage_min", "width_ratio_max", "passes_coverage_floor", "passes_width_guard")
                    }
                    for level, summary in summaries.items()
                },
                "out_json": str(out_json),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())