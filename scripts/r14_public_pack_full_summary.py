#!/usr/bin/env python3
"""Summarize public-pack full-scope family shard manifests."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="public_pack_full_*_*/MANIFEST.json")
    parser.add_argument("--expected-families", default="ecl,ett,exchange,ili,solar,traffic,weather")
    parser.add_argument("--stem", default="r14_public_pack_full_summary")
    return parser.parse_args()


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get("metrics", {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def main() -> int:
    args = _parse_args()
    expected = sorted({item.strip() for item in args.expected_families.split(",") if item.strip()})
    manifests = sorted((ROOT / "runs" / "audits").glob(args.glob))
    family_rows = []
    all_results: list[dict[str, Any]] = []
    read_errors = []
    for manifest in manifests:
        try:
            data = json.loads(manifest.read_text())
        except Exception as exc:  # noqa: BLE001 - audit records read failures
            read_errors.append({"path": str(manifest), "error_type": type(exc).__name__, "error": str(exc)})
            continue
        cells = data.get("cells", []) if isinstance(data.get("cells"), list) else []
        results = data.get("results", []) if isinstance(data.get("results"), list) else []
        family = next(iter(sorted({str(cell.get("family")) for cell in cells if cell.get("family")})), None)
        if family is None:
            family = next(iter(sorted({str(row.get("family")) for row in results if row.get("family")})), manifest.parent.name)
        errors = [row for row in results if isinstance(row, dict) and row.get("error")]
        mae = [_metric(row, "mae") for row in results if isinstance(row, dict)]
        rmse = [_metric(row, "rmse") for row in results if isinstance(row, dict)]
        smape = [_metric(row, "smape") for row in results if isinstance(row, dict)]
        mae = [value for value in mae if value is not None]
        rmse = [value for value in rmse if value is not None]
        smape = [value for value in smape if value is not None]
        family_rows.append({
            "family": family,
            "manifest": str(manifest.relative_to(ROOT)),
            "n_cells": len(cells),
            "n_results": len(results),
            "n_errors": len(errors),
            "models": sorted({str(row.get("model_name")) for row in results if isinstance(row, dict) and row.get("model_name")}),
            "mae_mean": _mean(mae),
            "rmse_mean": _mean(rmse),
            "smape_mean": _mean(smape),
            "fairness_pass_rate": _mean([1.0 if row.get("fairness_pass") is True else 0.0 for row in results if isinstance(row, dict) and row.get("fairness_pass") is not None]),
        })
        all_results.extend(row for row in results if isinstance(row, dict))

    completed_families = sorted({row["family"] for row in family_rows})
    missing = sorted(set(expected) - set(completed_families))
    status = "passed" if not missing and family_rows and not any(row["n_errors"] for row in family_rows) and not read_errors else "partial"
    payload = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": status,
        "expected_families": expected,
        "completed_families": completed_families,
        "missing_families": missing,
        "n_manifests": len(manifests),
        "n_read_errors": len(read_errors),
        "read_errors": read_errors,
        "total_cells": sum(row["n_cells"] for row in family_rows),
        "total_results": sum(row["n_results"] for row in family_rows),
        "total_errors": sum(row["n_errors"] for row in family_rows),
        "overall_metric_means": {
            key: _mean([value for value in (_metric(row, key) for row in all_results) if value is not None])
            for key in ("mae", "rmse", "smape", "mape")
        },
        "family_rows": sorted(family_rows, key=lambda row: row["family"]),
        "limitations": [
            "This summarizes completed public-pack manifests only; missing family shards remain partial.",
            "Metrics are not cross-dataset normalized, so overall MAE/RMSE means are monitoring signals, not SOTA rankings.",
        ],
    }

    out_json = ROOT / "runs" / "audits" / f"{args.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    out_md.write_text("# R14 Public Pack Full Summary\n\n```json\n" + json.dumps(payload, indent=2, default=str) + "\n```\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "completed_families", "missing_families", "total_cells", "total_results", "total_errors", "overall_metric_means")}, indent=2, default=str))
    print(out_json)
    return 0 if status in {"passed", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())