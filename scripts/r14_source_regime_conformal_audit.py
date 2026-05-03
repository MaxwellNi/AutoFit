#!/usr/bin/env python3
"""Audit source as regime/context using landed R14 metric rows."""

from __future__ import annotations

import glob
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ABLATION_TO_REGIME = {
    "core_only": "no_source",
    "core_text": "text_regime",
    "core_edgar": "edgar_regime",
    "full": "full_source_regime",
}
METRICS = [
    "mae",
    "nccopo_coverage_90",
    "nccopo_coverage_90_cqr_lite",
    "nccopo_coverage_90_drift_guard",
    "nccopo_interval_width_mean",
    "nccopo_interval_width_mean_cqr_lite",
    "nccopo_interval_width_mean_drift_guard",
]


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _metric_files() -> list[Path]:
    patterns = [
        "runs/benchmarks/r13patch_*v2*/metrics.json",
        "runs/benchmarks/r14fcast_*/metrics.json",
        "runs/benchmarks/r14mond_*/metrics.json",
        "runs/benchmarks/r14stud_*/metrics.json",
    ]
    files: set[str] = set()
    for pattern in patterns:
        files.update(glob.glob(str(ROOT / pattern)))
    return [Path(path) for path in sorted(files)]


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _metric_files():
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            payload = payload["results"]
        if not isinstance(payload, list):
            continue
        run = path.parent.name
        diagnostic = any(token in run for token in ("source_scale", "telemetry", "driftguard"))
        mtime = os.path.getmtime(path)
        for item in payload:
            if not isinstance(item, dict):
                continue
            ablation = item.get("ablation")
            if ablation not in ABLATION_TO_REGIME:
                continue
            copied = dict(item)
            copied["source_regime"] = ABLATION_TO_REGIME[str(ablation)]
            copied["_run"] = run
            copied["_mtime"] = mtime
            copied["_diagnostic_run"] = diagnostic
            copied["_metrics_path"] = str(path.relative_to(ROOT))
            rows.append(copied)
    return rows


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"n_rows": len(rows)}
    for metric in METRICS:
        vals = [_finite(row.get(metric)) for row in rows]
        vals = [value for value in vals if value is not None]
        out[f"{metric}_mean"] = _mean(vals)
        out[f"{metric}_n"] = len(vals)
        if "coverage" in metric:
            out[f"{metric}_below_0_88"] = sum(1 for value in vals if value < 0.88)
            out[f"{metric}_below_0_90"] = sum(1 for value in vals if value < 0.90)
    return out


def _latest_by_key(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("model_name") or row.get("model"),
            row.get("category"),
            row.get("task"),
            row.get("target"),
            row.get("horizon"),
            row.get("ablation"),
        )
        if key not in latest or float(row.get("_mtime", 0.0)) >= float(latest[key].get("_mtime", 0.0)):
            latest[key] = row
    return latest


def _paired(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = _latest_by_key([row for row in rows if not row.get("_diagnostic_run")])
    by_base: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for key, row in latest.items():
        base = key[:-1]
        by_base[base][str(row.get("ablation"))] = row
    pairs = []
    for base, by_ablation in sorted(by_base.items(), key=lambda item: str(item[0])):
        core = by_ablation.get("core_only")
        if core is None:
            continue
        for ablation in ("core_text", "core_edgar", "full"):
            candidate = by_ablation.get(ablation)
            if candidate is None:
                continue
            pair = {
                "model": base[0],
                "category": base[1],
                "task": base[2],
                "target": base[3],
                "horizon": base[4],
                "source_ablation": ablation,
                "source_regime": ABLATION_TO_REGIME[ablation],
                "core_run": core.get("_run"),
                "source_run": candidate.get("_run"),
            }
            for metric in METRICS:
                core_val = _finite(core.get(metric))
                cand_val = _finite(candidate.get(metric))
                pair[f"core_{metric}"] = core_val
                pair[f"source_{metric}"] = cand_val
                pair[f"delta_{metric}_source_minus_core"] = None if core_val is None or cand_val is None else cand_val - core_val
            pairs.append(pair)
    return pairs


def _pair_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_pairs": len(pairs)}
    for metric in ("mae", "nccopo_coverage_90", "nccopo_coverage_90_cqr_lite", "nccopo_coverage_90_drift_guard"):
        key = f"delta_{metric}_source_minus_core"
        vals = [pair.get(key) for pair in pairs if pair.get(key) is not None]
        vals = [float(value) for value in vals]
        out[f"{key}_n"] = len(vals)
        out[f"{key}_mean"] = _mean(vals)
        if metric == "mae":
            out[f"{key}_wins_lower_mae"] = sum(1 for value in vals if value < -1e-9)
            out[f"{key}_losses_higher_mae"] = sum(1 for value in vals if value > 1e-9)
        else:
            out[f"{key}_wins_higher_coverage"] = sum(1 for value in vals if value > 1e-12)
            out[f"{key}_losses_lower_coverage"] = sum(1 for value in vals if value < -1e-12)
    return out


def main() -> int:
    rows = _rows()
    non_diag = [row for row in rows if not row.get("_diagnostic_run")]
    pairs = _paired(rows)
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "scope": "landed metric rows; source is interpreted as ablation/regime context, not residual multiplier",
        "n_rows_raw": len(rows),
        "n_rows_non_diagnostic": len(non_diag),
        "regime_summary_raw": {regime: _summarize([row for row in rows if row.get("source_regime") == regime]) for regime in sorted(set(ABLATION_TO_REGIME.values()))},
        "regime_summary_non_diagnostic": {regime: _summarize([row for row in non_diag if row.get("source_regime") == regime]) for regime in sorted(set(ABLATION_TO_REGIME.values()))},
        "paired_summary_vs_core_only": _pair_summary(pairs),
        "paired_rows_weakest_mae_losses": sorted(
            [pair for pair in pairs if pair.get("delta_mae_source_minus_core") is not None],
            key=lambda pair: float(pair["delta_mae_source_minus_core"]),
            reverse=True,
        )[:30],
        "paired_rows_best_mae_wins": sorted(
            [pair for pair in pairs if pair.get("delta_mae_source_minus_core") is not None],
            key=lambda pair: float(pair["delta_mae_source_minus_core"]),
        )[:30],
        "interpretation": [
            "This is a regime/context audit over landed rows, not a new source mechanism pass gate.",
            "A source regime is useful only if paired MAE/coverage improves without weak-cell regressions.",
            "Formal promotion still requires row-keyed strict counterfactuals and temporal reruns."
        ],
    }
    out_json = ROOT / "runs" / "audits" / f"r14_source_regime_conformal_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_md = out_json.with_suffix(".md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text("# R14 Source Regime Conformal Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n", encoding="utf-8")
    print(json.dumps({
        "n_rows_raw": report["n_rows_raw"],
        "n_rows_non_diagnostic": report["n_rows_non_diagnostic"],
        "paired_summary_vs_core_only": report["paired_summary_vs_core_only"],
    }, indent=2, default=str))
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())