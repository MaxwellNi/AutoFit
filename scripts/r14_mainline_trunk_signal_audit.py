#!/usr/bin/env python3
"""Audit mainline trunk/guard signals for Round-14 forecast runs.

This script is intentionally evidence-limited: it summarizes landed metrics
that can indicate whether horizon/ablation/lane guards are active, but it does
not claim to prove causal trunk learning. The missing proof remains an
identical-row counterfactual run.
"""

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
OUT_JSON = ROOT / "runs" / "audits" / f"r14_mainline_trunk_signal_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def _metric_files() -> list[str]:
    patterns = [
        "runs/benchmarks/r14fcast_*/metrics.json",
        "runs/benchmarks/r14mond_*/metrics.json",
        "runs/benchmarks/r14stud_*/metrics.json",
    ]
    files: set[str] = set()
    for pattern in patterns:
        files.update(glob.glob(str(ROOT / pattern)))
    return sorted(files)


def _rows_from_file(path: str) -> list[dict[str, Any]]:
    try:
        payload = json.load(open(path))
    except Exception:
        return []
    if isinstance(payload, dict):
        payload = payload.get("results", payload.get("metrics", payload))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    rows = []
    for row in payload:
        if isinstance(row, dict):
            copied = dict(row)
            copied["_metrics_path"] = path
            rows.append(copied)
    return rows


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _bool_count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    true_count = sum(1 for row in rows if row.get(key) is True)
    false_count = sum(1 for row in rows if row.get(key) is False)
    none_count = len(rows) - true_count - false_count
    return {"true": true_count, "false": false_count, "missing": none_count}


def _range(values: list[float]) -> float | None:
    vals = [value for value in values if math.isfinite(value)]
    if not vals:
        return None
    return max(vals) - min(vals)


def main() -> int:
    files = _metric_files()
    all_rows: list[dict[str, Any]] = []
    for path in files:
        all_rows.extend(_rows_from_file(path))

    rows = [
        row for row in all_rows
        if row.get("category") == "mainline"
        and row.get("task") == "task1_outcome"
        and row.get("target") == "funding_raised_usd"
    ]

    cells = sorted({
        (row.get("ablation"), row.get("horizon"))
        for row in rows
        if row.get("ablation") is not None and row.get("horizon") is not None
    }, key=str)

    by_model_ablation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model = str(row.get("model_name") or row.get("model") or "?")
        ablation = str(row.get("ablation") or "?")
        by_model_ablation[(model, ablation)].append(row)

    horizon_checks = []
    for (model, ablation), group in sorted(by_model_ablation.items(), key=lambda item: item[0]):
        horizons = sorted({row.get("horizon") for row in group if row.get("horizon") is not None})
        if len(horizons) < 2:
            continue
        maes = [_finite(row.get("mae")) for row in group]
        c90s = [_finite(row.get("nccopo_coverage_90")) for row in group]
        dropped = [_finite(row.get("y_shift_dropped_tail_rows")) for row in group]
        horizon_checks.append({
            "model": model,
            "ablation": ablation,
            "horizons": horizons,
            "mae_range": _range([v for v in maes if v is not None]),
            "c90_range": _range([v for v in c90s if v is not None]),
            "y_shift_dropped_tail_rows_range": _range([v for v in dropped if v is not None]),
        })

    source_strengths = [
        _finite(row.get("lane_source_scale_strength"))
        for row in rows
        if _finite(row.get("lane_source_scale_strength")) is not None
    ]
    tail_weights = [
        _finite(row.get("lane_tail_weight_effective"))
        for row in rows
        if _finite(row.get("lane_tail_weight_effective")) is not None
    ]
    c90s = [_finite(row.get("nccopo_coverage_90")) for row in rows]
    mondrian = [_finite(row.get("nccopo_coverage_90_mondrian")) for row in rows]
    studentized = [_finite(row.get("nccopo_coverage_90_studentized")) for row in rows]

    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "n_metric_files_scanned": len(files),
        "n_mainline_funding_rows": len(rows),
        "cells_present_ablation_horizon": cells,
        "n_cells_present": len(cells),
        "guard_summary": {
            "lane_anchor_only_mode": _bool_count(rows, "lane_anchor_only_mode"),
            "lane_trunk_fallback_fitted": _bool_count(rows, "lane_trunk_fallback_fitted"),
            "lane_hurdle_engaged": _bool_count(rows, "lane_hurdle_engaged"),
            "source_scale_positive_rows": sum(1 for value in source_strengths if value > 0.0),
            "source_scale_observed_rows": len(source_strengths),
            "tail_weight_positive_rows": sum(1 for value in tail_weights if value > 0.0),
            "tail_weight_observed_rows": len(tail_weights),
        },
        "coverage_summary": {
            "marginal_n": sum(1 for value in c90s if value is not None),
            "marginal_mean": statistics.mean([v for v in c90s if v is not None]) if any(v is not None for v in c90s) else None,
            "mondrian_n": sum(1 for value in mondrian if value is not None),
            "studentized_n": sum(1 for value in studentized if value is not None),
        },
        "horizon_variation_summary": {
            "model_ablation_groups_with_2plus_horizons": len(horizon_checks),
            "mae_varies_groups": sum(1 for item in horizon_checks if (item.get("mae_range") or 0.0) > 1e-9),
            "c90_varies_groups": sum(1 for item in horizon_checks if (item.get("c90_range") or 0.0) > 1e-12),
            "y_shift_drop_varies_groups": sum(1 for item in horizon_checks if (item.get("y_shift_dropped_tail_rows_range") or 0.0) > 0.0),
        },
        "horizon_variation_examples": horizon_checks[:25],
        "interpretation_limits": [
            "This audit can show horizon-conditioned outputs and guard fields vary in landed metrics.",
            "It cannot prove causal trunk learning without an identical-row counterfactual run.",
            "If anchor_only/fallback rows are common, mainline success cannot be attributed to trunk alone.",
        ],
        "next_required_tests": [
            "Run identical-row horizon counterfactuals for selected mainline variants.",
            "Run same-row core_only/core_text/core_edgar/full ablation deltas with guard fields reported.",
            "Inspect studentized funding-only results once r14stud jobs land.",
        ],
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    lines = [f"# R14 Mainline Trunk Signal Audit — {report['timestamp_cest']}", ""]
    lines.append("## Summary")
    lines.append("```json")
    lines.append(json.dumps({k: report[k] for k in (
        "n_metric_files_scanned", "n_mainline_funding_rows", "n_cells_present",
        "guard_summary", "coverage_summary", "horizon_variation_summary",
        "interpretation_limits", "next_required_tests",
    )}, indent=2, default=str))
    lines.append("```")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"OK: {OUT_JSON}")
    print(f"OK: {OUT_MD}")
    print(json.dumps(report["horizon_variation_summary"], indent=2, default=str))
    print(json.dumps(report["guard_summary"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())