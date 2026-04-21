#!/usr/bin/env python3
"""Zero-training audit for ED-SSM Battle1/Battle3 anti-leakage checks.

This script does NOT train any model. It only reads:
1) Battle result artifacts under runs/edssm_battle*_fixed
2) Dataset slices produced by existing case builders
3) Feature-prep and join behavior used by battle scripts

Outputs a JSON report answering three urgent questions:
- Ghost funding illusion robustness
- Investors target physical scale sanity
- Look-ahead leakage checks for EDGAR joins and feature prep
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v740_alpha_minibenchmark import _build_case_frame, _make_temporal_config
from scripts.run_v740_alpha_smoke_slice import (
    _TARGET_LEAK_GROUPS,
    _join_edgar_asof,
    _prepare_features,
)


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_prediction_vectors(node: Any) -> List[str]:
    """Return JSON paths that look like per-sample prediction arrays."""
    hits: List[str] = []

    def _walk(cur: Any, path: str) -> None:
        if isinstance(cur, dict):
            for k, v in cur.items():
                _walk(v, f"{path}.{k}" if path else k)
            return
        if isinstance(cur, list):
            if cur and isinstance(cur[0], (int, float)) and len(cur) > 50:
                hits.append(path)
            for i, v in enumerate(cur[:3]):
                _walk(v, f"{path}[{i}]")

    _walk(node, "")
    return hits


def _investors_true_distribution() -> Dict[str, Any]:
    temporal = _make_temporal_config()
    out_cells: List[Dict[str, Any]] = []
    y_all: List[np.ndarray] = []

    for ablation in ["core_only", "core_edgar"]:
        for horizon in [1, 7]:
            case = {
                "task": "task3_risk_adjust",
                "ablation": ablation,
                "target": "investors_count",
                "horizon": horizon,
                "max_entities": 5000,
                "max_rows": 8000,
                "name": f"audit_t3_{ablation}_h{horizon}",
            }
            _, _, test = _build_case_frame(case, temporal)
            _, y_test = _prepare_features(test, "investors_count")
            y = y_test.to_numpy(dtype=np.float64)
            y = y[np.isfinite(y)]
            if len(y) == 0:
                stats = {
                    "n": 0,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                    "p95": None,
                    "p99": None,
                    "zero_ratio": None,
                }
            else:
                stats = {
                    "n": int(len(y)),
                    "min": float(np.min(y)),
                    "max": float(np.max(y)),
                    "mean": float(np.mean(y)),
                    "median": float(np.median(y)),
                    "p95": float(np.quantile(y, 0.95)),
                    "p99": float(np.quantile(y, 0.99)),
                    "zero_ratio": float(np.mean(y == 0.0)),
                }
                y_all.append(y)

            dup_key_rate = None
            if {"entity_id", "crawled_date_day"}.issubset(set(test.columns)):
                key_counts = test.groupby(["entity_id", "crawled_date_day"]).size()
                dup_key_rate = float(np.mean(key_counts.to_numpy() > 1)) if len(key_counts) else 0.0

            out_cells.append(
                {
                    "cell": f"task3_risk_adjust__{ablation}__investors_count__h{horizon}",
                    "stats": stats,
                    "dup_entity_day_rate": dup_key_rate,
                }
            )

    if y_all:
        y_cat = np.concatenate(y_all)
        aggregate = {
            "n": int(len(y_cat)),
            "min": float(np.min(y_cat)),
            "max": float(np.max(y_cat)),
            "mean": float(np.mean(y_cat)),
            "median": float(np.median(y_cat)),
            "p95": float(np.quantile(y_cat, 0.95)),
            "p99": float(np.quantile(y_cat, 0.99)),
            "zero_ratio": float(np.mean(y_cat == 0.0)),
        }
    else:
        aggregate = {"n": 0}

    return {"cells": out_cells, "aggregate": aggregate}


def _lookahead_checks() -> Dict[str, Any]:
    temporal = _make_temporal_config()

    src = inspect.getsource(_join_edgar_asof)
    static_contract = {
        "uses_merge_asof": "merge_asof" in src,
        "direction_backward": "direction=\"backward\"" in src,
        "join_key_crawled_date_day": "on=\"crawled_date_day\"" in src,
        "by_cik": "by=\"cik\"" in src,
        "tolerance_90d": "90D" in src,
    }

    runtime_cells: List[Dict[str, Any]] = []
    for task, target in [
        ("task2_forecast", "funding_raised_usd"),
        ("task3_risk_adjust", "investors_count"),
    ]:
        for horizon in [1, 7]:
            case = {
                "task": task,
                "ablation": "core_edgar",
                "target": target,
                "horizon": horizon,
                "max_entities": 5000,
                "max_rows": 8000,
                "name": f"audit_{task}_core_edgar_{target}_h{horizon}",
            }
            train, val, test = _build_case_frame(case, temporal)
            split_rows = []
            for split_name, df in [("train", train), ("val", val), ("test", test)]:
                row: Dict[str, Any] = {"split": split_name, "n_rows": int(len(df))}
                def _to_naive(s: pd.Series) -> pd.Series:
                    """Normalize to tz-naive datetime64[ns]."""
                    s = pd.to_datetime(s, errors="coerce", utc=True)
                    return s.dt.tz_convert(None)

                if "edgar_filed_date" in df.columns:
                    cdd = _to_naive(df["crawled_date_day"])
                    efd = _to_naive(df["edgar_filed_date"])
                    mask = efd.notna() & cdd.notna()
                    viol = (efd > cdd) & mask
                    row["edgar_filed_date_gt_crawled_date_day"] = int(viol.sum())
                    row["edgar_filed_date_compared_rows"] = int(mask.sum())
                else:
                    row["edgar_filed_date_gt_crawled_date_day"] = None
                    row["edgar_filed_date_compared_rows"] = 0

                if "cutoff_ts" in df.columns:
                    cdd = _to_naive(df["crawled_date_day"])
                    cut = _to_naive(df["cutoff_ts"])
                    mask = cut.notna() & cdd.notna()
                    # compare by date only to avoid same-day timestamp noise
                    viol = (cut.dt.normalize() > cdd.dt.normalize()) & mask
                    row["cutoff_ts_date_gt_crawled_date_day"] = int(viol.sum())
                    row["cutoff_ts_compared_rows"] = int(mask.sum())
                else:
                    row["cutoff_ts_date_gt_crawled_date_day"] = None
                    row["cutoff_ts_compared_rows"] = 0
                split_rows.append(row)

            # Feature-level leak exclusion check
            X_test, _ = _prepare_features(test, target)
            leak_group = _TARGET_LEAK_GROUPS.get(target, {target})
            leaked_feature_columns = sorted(set(X_test.columns).intersection(leak_group))

            runtime_cells.append(
                {
                    "cell": f"{task}__core_edgar__{target}__h{horizon}",
                    "splits": split_rows,
                    "leak_group_columns_present_in_X_test": leaked_feature_columns,
                }
            )

    return {
        "static_join_contract": static_contract,
        "runtime_cells": runtime_cells,
    }


def build_report(b1_path: Path, b3_path: Path) -> Dict[str, Any]:
    b1 = _load_json(b1_path)
    b3 = _load_json(b3_path)

    # Q1: Ghost-funding illusion with investors==0 check
    b3_pred_paths = _find_prediction_vectors(b3)
    can_count_investors_eq0 = len(b3_pred_paths) > 0

    q1_cells = []
    for c in b3.get("cells", []):
        e = c.get("edssm_mainline", {})
        q1_cells.append(
            {
                "cell": c.get("cell"),
                "n_valid": int(e.get("n_valid", 0)) if e.get("n_valid") is not None else None,
                "ghost_count": int(e.get("ghost_count", 0)) if e.get("ghost_count") is not None else None,
                "ghost_rate": _safe_float(e.get("ghost_rate")),
                "pred_investors_mean": _safe_float(e.get("pred_investors_mean")),
                "pred_funding_mean": _safe_float(e.get("pred_funding_mean")),
            }
        )

    q1 = {
        "question": "Can we directly count ED-SSM predictions where investors==0 from saved Battle3 artifacts?",
        "answer": "yes" if can_count_investors_eq0 else "no",
        "reason": (
            "Per-sample prediction arrays are present in battle3 artifact."
            if can_count_investors_eq0
            else "Battle3 JSON only stores aggregates (means/counts/rates), not per-row prediction vectors; exact investors==0 count is not recoverable post-hoc."
        ),
        "prediction_array_paths_detected": b3_pred_paths,
        "edssm_aggregate": {
            "aggregate_edssm_ghost_rate": _safe_float(b3.get("aggregate_edssm_ghost_rate")),
            "aggregate_edssm_inversion_rate": _safe_float(b3.get("aggregate_edssm_inversion_rate")),
        },
        "cells": q1_cells,
    }

    # Q2: investors true target scale audit
    q2 = _investors_true_distribution()

    # Q3: look-ahead / leakage checks
    q3 = _lookahead_checks()

    return {
        "audit": "edssm_battle_anti_leakage_metric_illusion",
        "artifacts": {
            "battle1": str(b1_path),
            "battle3": str(b3_path),
            "battle1_total": int(b1.get("total", 0)),
            "battle3_total": int(b3.get("total", 0)),
        },
        "q1_ghost_funding_illusion": q1,
        "q2_investors_true_target_scale": q2,
        "q3_lookahead_bias_check": q3,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-training anti-leakage audit for ED-SSM battles")
    ap.add_argument("--battle1-json", default="runs/edssm_battle1_fixed/battle1_revenge_results.json")
    ap.add_argument("--battle3-json", default="runs/edssm_battle3_fixed/battle3_audit_results.json")
    ap.add_argument("--output-json", default="runs/edssm_battle_audit/anti_leakage_metric_illusion_audit.json")
    args = ap.parse_args()

    report = build_report(Path(args.battle1_json), Path(args.battle3_json))
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # concise stdout summary for terminal users
    q1 = report["q1_ghost_funding_illusion"]
    q2 = report["q2_investors_true_target_scale"]["aggregate"]
    print(json.dumps(
        {
            "output_json": str(out_path),
            "q1_can_count_investors_eq0_from_artifact": q1["answer"],
            "q1_aggregate_edssm_ghost_rate": q1["edssm_aggregate"]["aggregate_edssm_ghost_rate"],
            "q2_investors_aggregate": q2,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
