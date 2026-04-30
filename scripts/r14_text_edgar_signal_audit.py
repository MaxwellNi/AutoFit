#!/usr/bin/env python3
"""Audit text/EDGAR signal contribution in landed Round-14 funding runs.

The audit is deliberately result-facing: it compares core_text/core_edgar/full
against the same model/category/horizon core_only row, then records whether
the extra information improved MAE or conformal coverage closeness. It also
summarizes artifact/join evidence, which proves availability but not efficacy.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_text_edgar_signal_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _rows_from_metrics(path: Path) -> list[dict[str, Any]]:
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
        if not isinstance(row, dict):
            continue
        copied = dict(row)
        copied["_metrics_path"] = str(path)
        copied["_metrics_mtime"] = path.stat().st_mtime
        rows.append(copied)
    return rows


def _metric_files() -> list[Path]:
    patterns = [
        "runs/benchmarks/r14fcast_*/metrics.json",
        "runs/benchmarks/r14mond_*/metrics.json",
        "runs/benchmarks/r14stud_*/metrics.json",
    ]
    files: set[Path] = set()
    for pattern in patterns:
        files.update(ROOT.glob(pattern))
    return sorted(files)


def _latest_funding_rows() -> list[dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in _metric_files():
        for row in _rows_from_metrics(path):
            if row.get("task") != "task1_outcome":
                continue
            if row.get("target") != "funding_raised_usd":
                continue
            ablation = row.get("ablation")
            if ablation not in {"core_only", "core_text", "core_edgar", "full"}:
                continue
            key = (
                row.get("category"),
                row.get("model_name") or row.get("model"),
                row.get("horizon"),
                ablation,
            )
            previous = rows.get(key)
            if previous is None or row["_metrics_mtime"] >= previous["_metrics_mtime"]:
                rows[key] = row
    return list(rows.values())


def _summarize_numbers(values: list[float]) -> dict[str, float | None]:
    vals = [value for value in values if math.isfinite(value)]
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def _summarize_pairs(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    by_ablation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_ablation_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in comparisons:
        by_ablation[item["ablation"]].append(item)
        by_ablation_category[f"{item['ablation']}::{item['category']}"].append(item)

    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        mae_deltas = [item["delta_mae"] for item in group if item["delta_mae"] is not None]
        c90_deltas = [item["delta_c90"] for item in group if item["delta_c90"] is not None]
        c90_closer = [item["delta_abs_c90_gap_to_090"] for item in group if item["delta_abs_c90_gap_to_090"] is not None]
        return {
            "n_pairs": len(group),
            "mae_wins_lower_is_better": sum(1 for value in mae_deltas if value < 0.0),
            "mae_losses": sum(1 for value in mae_deltas if value > 0.0),
            "mae_ties": sum(1 for value in mae_deltas if value == 0.0),
            "mae_delta": _summarize_numbers(mae_deltas),
            "c90_delta": _summarize_numbers(c90_deltas),
            "c90_closeness_wins_lower_abs_gap_is_better": sum(1 for value in c90_closer if value < 0.0),
            "c90_closeness_losses": sum(1 for value in c90_closer if value > 0.0),
            "c90_abs_gap_delta": _summarize_numbers(c90_closer),
        }

    return {
        "overall": one(comparisons),
        "by_ablation": {key: one(value) for key, value in sorted(by_ablation.items())},
        "by_ablation_category": {key: one(value) for key, value in sorted(by_ablation_category.items())},
    }


def audit_landed_result_deltas() -> dict[str, Any]:
    rows = _latest_funding_rows()
    by_base: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        base_key = (row.get("category"), row.get("model_name") or row.get("model"), row.get("horizon"))
        by_base[base_key][str(row.get("ablation"))] = row

    comparisons = []
    for (category, model, horizon), group in sorted(by_base.items(), key=str):
        baseline = group.get("core_only")
        if baseline is None:
            continue
        base_mae = _finite(baseline.get("mae"))
        base_c90 = _finite(baseline.get("nccopo_coverage_90"))
        for ablation in ("core_text", "core_edgar", "full"):
            row = group.get(ablation)
            if row is None:
                continue
            mae = _finite(row.get("mae"))
            c90 = _finite(row.get("nccopo_coverage_90"))
            delta_mae = None if mae is None or base_mae is None else mae - base_mae
            delta_c90 = None if c90 is None or base_c90 is None else c90 - base_c90
            delta_abs_gap = None
            if c90 is not None and base_c90 is not None:
                delta_abs_gap = abs(0.90 - c90) - abs(0.90 - base_c90)
            comparisons.append({
                "category": category,
                "model": model,
                "horizon": horizon,
                "ablation": ablation,
                "core_only_mae": base_mae,
                "ablation_mae": mae,
                "delta_mae": delta_mae,
                "core_only_c90": base_c90,
                "ablation_c90": c90,
                "delta_c90": delta_c90,
                "delta_abs_c90_gap_to_090": delta_abs_gap,
                "fairness_pass": row.get("fairness_pass"),
                "metrics_path": row.get("_metrics_path"),
            })

    source_strengths = [_finite(row.get("lane_source_scale_strength")) for row in rows]
    tail_weights = [_finite(row.get("lane_tail_weight_effective")) for row in rows]
    source_by_ablation: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get("lane_source_scale_strength"))
        if value is not None:
            source_by_ablation[str(row.get("ablation"))].append(value)

    return {
        "n_latest_funding_rows": len(rows),
        "n_comparable_pairs": len(comparisons),
        "summary": _summarize_pairs(comparisons),
        "comparisons": comparisons,
        "source_scale_activation": {
            "observed_rows": sum(1 for value in source_strengths if value is not None),
            "positive_rows": sum(1 for value in source_strengths if value is not None and value > 0.0),
            "positive_rows_by_ablation": {
                key: sum(1 for value in values if value > 0.0)
                for key, values in sorted(source_by_ablation.items())
            },
            "observed_rows_by_ablation": {
                key: len(values) for key, values in sorted(source_by_ablation.items())
            },
        },
        "tail_weight_activation": {
            "observed_rows": sum(1 for value in tail_weights if value is not None),
            "positive_rows": sum(1 for value in tail_weights if value is not None and value > 0.0),
        },
    }


def audit_embedding_artifacts() -> dict[str, Any]:
    root = ROOT / "runs" / "text_embeddings"
    metadata_path = root / "embedding_metadata.json"
    parquet_path = root / "text_embeddings.parquet"
    metadata: dict[str, Any] | None = None
    if metadata_path.exists():
        try:
            metadata = json.load(open(metadata_path))
        except Exception as exc:
            metadata = {"error": f"{type(exc).__name__}:{exc}"}

    parquet: dict[str, Any] = {"exists": parquet_path.exists()}
    if parquet_path.exists():
        parquet["size_bytes"] = parquet_path.stat().st_size
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(parquet_path)
            names = list(pf.schema.names)
            parquet.update({
                "num_rows": pf.metadata.num_rows,
                "num_columns": pf.metadata.num_columns,
                "n_text_emb_columns": sum(1 for name in names if name.startswith("text_emb_")),
                "schema_names_head": names[:12],
            })
        except Exception as exc:
            parquet["metadata_error"] = f"{type(exc).__name__}:{exc}"

    return {
        "embedding_metadata_path": str(metadata_path),
        "embedding_metadata_exists": metadata_path.exists(),
        "embedding_metadata": metadata,
        "parquet_path": str(parquet_path),
        "parquet": parquet,
    }


def audit_join_logs() -> dict[str, Any]:
    text_re = re.compile(r"Text embedding match rate:\s*([0-9.]+)%")
    edgar_re = re.compile(
        r"EDGAR as-of join:\s*([0-9,]+)/([0-9,]+) rows with CIK, match_rate=([0-9.]+)% on ([0-9,]+) features"
    )
    paths: list[Path] = []
    for directory in [ROOT / "slurm_logs", Path("/work/projects/eint/logs/phase15")]:
        if directory.exists():
            paths.extend(sorted(directory.glob("*.err")))

    text_rates: list[float] = []
    edgar_matches: list[dict[str, Any]] = []
    samples: list[str] = []
    for path in paths:
        try:
            with open(path, errors="ignore") as fh:
                for line in fh:
                    text_match = text_re.search(line)
                    if text_match:
                        text_rates.append(float(text_match.group(1)))
                        if len(samples) < 20:
                            samples.append(f"{path}:{line.strip()}")
                    edgar_match = edgar_re.search(line)
                    if edgar_match:
                        row = {
                            "rows_with_cik": int(edgar_match.group(1).replace(",", "")),
                            "total_rows": int(edgar_match.group(2).replace(",", "")),
                            "match_rate_percent": float(edgar_match.group(3)),
                            "n_features": int(edgar_match.group(4).replace(",", "")),
                            "path": str(path),
                        }
                        edgar_matches.append(row)
                        if len(samples) < 20:
                            samples.append(f"{path}:{line.strip()}")
        except Exception:
            continue

    latest_edgar = edgar_matches[-1] if edgar_matches else None
    return {
        "n_log_files_scanned": len(paths),
        "text_embedding_match_rate_percent": _summarize_numbers(text_rates),
        "n_text_match_log_lines": len(text_rates),
        "n_edgar_join_log_lines": len(edgar_matches),
        "latest_edgar_join": latest_edgar,
        "sample_lines": samples,
    }


def verdict(result_delta: dict[str, Any]) -> dict[str, Any]:
    by_ablation = result_delta["summary"]["by_ablation"]
    notes = []
    for ablation in ("core_text", "core_edgar", "full"):
        item = by_ablation.get(ablation, {})
        n_pairs = int(item.get("n_pairs") or 0)
        wins = int(item.get("mae_wins_lower_is_better") or 0)
        losses = int(item.get("mae_losses") or 0)
        if n_pairs == 0:
            notes.append(f"{ablation}: no core_only-paired landed rows")
        elif wins <= losses:
            notes.append(f"{ablation}: not validated as helpful on MAE ({wins} wins, {losses} losses)")
        else:
            notes.append(f"{ablation}: MAE-positive in current paired rows ({wins} wins, {losses} losses)")

    source = result_delta["source_scale_activation"]
    if source.get("observed_rows") and not source.get("positive_rows"):
        notes.append("source scaling is observed but never positive in landed rows; source-scale novelty is not currently evidenced")
    return {
        "claim_level": "artifact/join evidence exists; predictive usefulness remains mixed and must not be overclaimed",
        "notes": notes,
    }


def main() -> int:
    result_delta = audit_landed_result_deltas()
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "result_delta_audit": result_delta,
        "embedding_artifact_audit": audit_embedding_artifacts(),
        "join_log_audit": audit_join_logs(),
        "verdict": verdict(result_delta),
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    compact = {
        "result_delta_summary": result_delta["summary"],
        "source_scale_activation": result_delta["source_scale_activation"],
        "tail_weight_activation": result_delta["tail_weight_activation"],
        "embedding_artifact_audit": report["embedding_artifact_audit"],
        "join_log_audit": report["join_log_audit"],
        "verdict": report["verdict"],
    }
    lines = [f"# R14 Text/EDGAR Signal Audit — {report['timestamp_cest']}", "", "```json"]
    lines.append(json.dumps(compact, indent=2, default=str))
    lines.append("```")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"OK: {OUT_JSON}")
    print(f"OK: {OUT_MD}")
    print(json.dumps(compact["result_delta_summary"], indent=2, default=str))
    print(json.dumps(compact["source_scale_activation"], indent=2, default=str))
    print(json.dumps(compact["verdict"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())