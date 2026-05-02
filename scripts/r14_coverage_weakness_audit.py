#!/usr/bin/env python3
"""Locate weak coverage cells across R14 NC-CoPo/CQR audit metrics."""

from __future__ import annotations

import glob
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_coverage_weakness_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")


def _metric_files() -> list[Path]:
    paths = set(
        glob.glob(str(ROOT / "runs/benchmarks/r13patch_*v2*/metrics.json"))
        + glob.glob(str(ROOT / "runs/benchmarks/r14fcast_*/metrics.json"))
        + glob.glob(str(ROOT / "runs/benchmarks/r14mond_*/metrics.json"))
        + glob.glob(str(ROOT / "runs/benchmarks/r14stud_*/metrics.json"))
    )
    return [Path(path) for path in sorted(paths)]


def _load_rows(path: Path, read_errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - audit records read failures
        read_errors.append({"path": str(path), "error_type": type(exc).__name__, "error": str(exc)})
        return []
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        data = data["results"]
    if not isinstance(data, list):
        read_errors.append({"path": str(path), "error_type": "UnexpectedJSON", "error": "metrics root is not a list"})
        return []
    return [row for row in data if isinstance(row, dict)]


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _record(path: Path, row: dict[str, Any], metric: str, value: float) -> dict[str, Any]:
    run = path.parent.name
    return {
        "run": run,
        "canonical_cqr_protocol": ("cqrrow" in run) or ("cqrgpd" in run),
        "metric": metric,
        "coverage": value,
        "gap_to_0_90": 0.90 - value,
        "gap_to_0_88": 0.88 - value,
        "model": row.get("model_name") or row.get("model"),
        "category": row.get("category"),
        "task": row.get("task"),
        "target": row.get("target"),
        "ablation": row.get("ablation"),
        "horizon": row.get("horizon"),
        "mae": row.get("mae"),
        "fairness_pass": row.get("fairness_pass"),
        "metrics_path": str(path.relative_to(ROOT)),
    }


def _summarize(records: list[dict[str, Any]], *, canonical_only: bool = False) -> dict[str, Any]:
    items = [item for item in records if not canonical_only or item.get("canonical_cqr_protocol")]
    values = [float(item["coverage"]) for item in items]
    weakest = sorted(items, key=lambda item: (item["coverage"], str(item.get("run"))))[:40]
    return {
        "n_records": len(items),
        "mean": statistics.mean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "below_0_88": sum(1 for value in values if value < 0.88),
        "below_0_90": sum(1 for value in values if value < 0.90),
        "weakest_40": weakest,
    }


def main() -> int:
    read_errors: list[dict[str, Any]] = []
    metric_fields = {
        "marginal": "nccopo_coverage_90",
        "mondrian": "nccopo_coverage_90_mondrian",
        "studentized": "nccopo_coverage_90_studentized",
        "cqr_lite": "nccopo_coverage_90_cqr_lite",
        "gpd_evt": "nccopo_coverage_90_gpd_evt",
        "drift_guard": "nccopo_coverage_90_drift_guard",
    }
    records_by_metric: dict[str, list[dict[str, Any]]] = {key: [] for key in metric_fields}
    files = _metric_files()
    for path in files:
        for row in _load_rows(path, read_errors):
            for metric, field in metric_fields.items():
                value = _finite(row.get(field))
                if value is not None:
                    records_by_metric[metric].append(_record(path, row, metric, value))

    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "n_metric_files": len(files),
        "metric_read_errors": read_errors,
        "summaries": {
            metric: _summarize(records)
            for metric, records in records_by_metric.items()
        },
        "canonical_cqr_lite_summary": _summarize(records_by_metric["cqr_lite"], canonical_only=True),
        "interpretation": [
            "This audit locates weak cells; it does not change pass/fail gates by itself.",
            "Canonical CQR-lite is reported separately because diagnostic probes can depress raw-all coverage.",
            "Cells below 0.88 are immediate pass-threshold blockers; cells below 0.90 are practical coverage gaps.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    OUT_MD.write_text("# R14 Coverage Weakness Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n", encoding="utf-8")
    print(json.dumps({
        "audit": str(OUT_JSON),
        "metric_read_errors": len(read_errors),
        "marginal": report["summaries"]["marginal"],
        "canonical_cqr_lite": report["canonical_cqr_lite_summary"],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())