#!/usr/bin/env python3
"""Audit a matched 1.5B-vs-7B text-embedding downstream benchmark pair."""

from __future__ import annotations

import glob
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_embedding_downstream_pair_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")

BASELINE_GLOB = "runs/benchmarks/r14fcast_embed15b_main_h14_ct_*/metrics.json"
CANDIDATE_GLOB = "runs/benchmarks/r14fcast_embed7bsh_main_h14_ct_*/metrics.json"
BASELINE_META = ROOT / "runs" / "text_embeddings" / "embedding_metadata.json"
CANDIDATE_META = ROOT / "runs" / "text_embeddings_gte_qwen2_7b_pca64_20260501_sharded" / "embedding_metadata.json"


def _latest(pattern: str) -> Path | None:
    paths = sorted(ROOT.glob(pattern))
    return paths[-1] if paths else None


def _load_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _model_name(row: dict[str, Any]) -> str:
    return str(row.get("model_name") or row.get("model") or "")


def _key(row: dict[str, Any]) -> tuple[str, str, str, int, str, int]:
    return (
        _model_name(row),
        str(row.get("task")),
        str(row.get("target")),
        int(row.get("horizon") or -1),
        str(row.get("ablation")),
        int(row.get("seed") or 42),
    )


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, str, int, str, int], dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            out[_key(row)] = row
    return out


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def main() -> int:
    baseline_path = _latest(BASELINE_GLOB)
    candidate_path = _latest(CANDIDATE_GLOB)
    baseline_rows = _load_json(baseline_path)
    candidate_rows = _load_json(candidate_path)
    baseline_rows = baseline_rows if isinstance(baseline_rows, list) else []
    candidate_rows = candidate_rows if isinstance(candidate_rows, list) else []

    baseline = _index(baseline_rows)
    candidate = _index(candidate_rows)
    common = sorted(set(baseline).intersection(candidate))
    comparisons = []
    mae_deltas = []
    mae_delta_pcts = []
    cqr_deltas = []
    marginal_deltas = []
    gpd_deltas = []
    source_positive_rows = 0

    for key in common:
        base = baseline[key]
        cand = candidate[key]
        base_mae = _finite(base.get("mae"))
        cand_mae = _finite(cand.get("mae"))
        mae_delta = None if base_mae is None or cand_mae is None else cand_mae - base_mae
        mae_delta_pct = None
        if mae_delta is not None and base_mae not in (None, 0.0):
            mae_delta_pct = mae_delta / base_mae * 100.0
            mae_deltas.append(float(mae_delta))
            mae_delta_pcts.append(float(mae_delta_pct))
        base_cqr = _finite(base.get("nccopo_coverage_90_cqr_lite"))
        cand_cqr = _finite(cand.get("nccopo_coverage_90_cqr_lite"))
        cqr_delta = None if base_cqr is None or cand_cqr is None else cand_cqr - base_cqr
        if cqr_delta is not None:
            cqr_deltas.append(float(cqr_delta))
        base_marginal = _finite(base.get("nccopo_coverage_90"))
        cand_marginal = _finite(cand.get("nccopo_coverage_90"))
        marginal_delta = None if base_marginal is None or cand_marginal is None else cand_marginal - base_marginal
        if marginal_delta is not None:
            marginal_deltas.append(float(marginal_delta))
        base_gpd = _finite(base.get("nccopo_coverage_90_gpd_evt"))
        cand_gpd = _finite(cand.get("nccopo_coverage_90_gpd_evt"))
        gpd_delta = None if base_gpd is None or cand_gpd is None else cand_gpd - base_gpd
        if gpd_delta is not None:
            gpd_deltas.append(float(gpd_delta))
        if (_finite(cand.get("lane_source_scale_strength")) or 0.0) > 0.0:
            source_positive_rows += 1
        comparisons.append({
            "model": key[0],
            "task": key[1],
            "target": key[2],
            "horizon": key[3],
            "ablation": key[4],
            "seed": key[5],
            "mae_1p5b": base_mae,
            "mae_7b": cand_mae,
            "mae_delta_7b_minus_1p5b": mae_delta,
            "mae_delta_pct_7b_minus_1p5b": mae_delta_pct,
            "cqr_lite_c90_1p5b": base_cqr,
            "cqr_lite_c90_7b": cand_cqr,
            "cqr_lite_c90_delta_7b_minus_1p5b": cqr_delta,
            "marginal_c90_delta_7b_minus_1p5b": marginal_delta,
            "gpd_evt_c90_delta_7b_minus_1p5b": gpd_delta,
        })

    mae_wins = sum(1 for item in comparisons if (item.get("mae_delta_7b_minus_1p5b") or 0.0) < 0.0)
    mae_losses = sum(1 for item in comparisons if (item.get("mae_delta_7b_minus_1p5b") or 0.0) > 0.0)
    n_pairs = len(comparisons)
    mae_win_rate = mae_wins / n_pairs if n_pairs else None
    mean_mae_delta_pct = _mean(mae_delta_pcts)
    mean_cqr_delta = _mean(cqr_deltas)

    passed = bool(
        n_pairs >= 10
        and mae_win_rate is not None
        and mae_win_rate >= 0.60
        and mean_mae_delta_pct is not None
        and mean_mae_delta_pct <= -1.0
        and mean_cqr_delta is not None
        and mean_cqr_delta >= -0.005
    )
    status = "passed" if passed else ("partial" if n_pairs else "not_passed")
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": status,
        "baseline": {
            "label": "gte-Qwen2-1.5B",
            "metrics_path": str(baseline_path) if baseline_path else None,
            "metadata_path": str(BASELINE_META),
            "metadata": _load_json(BASELINE_META),
        },
        "candidate": {
            "label": "gte-Qwen2-7B-sharded",
            "metrics_path": str(candidate_path) if candidate_path else None,
            "metadata_path": str(CANDIDATE_META),
            "metadata": _load_json(CANDIDATE_META),
        },
        "matched_protocol": {
            "task": "task1_outcome",
            "target": "funding_raised_usd",
            "horizon": 14,
            "ablation": "core_text",
            "category": "mainline",
            "same_model_set_required": True,
        },
        "summary": {
            "n_pairs": n_pairs,
            "mae_wins_7b_lower_is_better": mae_wins,
            "mae_losses_7b_lower_is_better": mae_losses,
            "mae_win_rate_7b": mae_win_rate,
            "mean_mae_delta_7b_minus_1p5b": _mean(mae_deltas),
            "mean_mae_delta_pct_7b_minus_1p5b": mean_mae_delta_pct,
            "mean_cqr_lite_c90_delta_7b_minus_1p5b": mean_cqr_delta,
            "mean_marginal_c90_delta_7b_minus_1p5b": _mean(marginal_deltas),
            "mean_gpd_evt_c90_delta_7b_minus_1p5b": _mean(gpd_deltas),
            "source_scale_positive_rows_7b": source_positive_rows,
        },
        "pass_rule": {
            "n_pairs_min": 10,
            "mae_win_rate_7b_min": 0.60,
            "mean_mae_delta_pct_7b_minus_1p5b_max": -1.0,
            "mean_cqr_lite_c90_delta_7b_minus_1p5b_min": -0.005,
        },
        "comparisons": comparisons,
        "claim_lock": "Only claim 7B downstream superiority if this audit is passed; otherwise describe it as a scoped matched comparison.",
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
    OUT_MD.write_text("# R14 Embedding Downstream Pair Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"OK: {OUT_JSON}")
    print(f"OK: {OUT_MD}")
    print(json.dumps({"status": status, **report["summary"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())