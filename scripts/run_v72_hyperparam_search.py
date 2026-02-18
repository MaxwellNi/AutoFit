#!/usr/bin/env python3
"""
Build V7.2 hyperparameter-search planning artifacts from strict comparable evidence.

Outputs:
  - docs/benchmarks/block3_truth_pack/hyperparam_search_ledger.csv
  - docs/benchmarks/block3_truth_pack/best_config_by_model_target.json
  - docs/benchmarks/block3_truth_pack/compute_cost_report.csv

This script is evidence-first: it reads only materialized benchmark metrics and
produces a reproducible search plan with budget and cost baselines. It does not
use test feedback for parameter selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
RUNS_BENCH_ROOT = ROOT / "runs" / "benchmarks"
DEFAULT_BENCH_GLOB = "block3_20260203_225620*"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "benchmarks" / "block3_truth_pack"


def _resolve_bench_dirs(explicit: Sequence[Path], bench_glob: str) -> List[Path]:
    if explicit:
        out = []
        seen = set()
        for p in explicit:
            rp = p.resolve()
            if rp in seen or not rp.exists() or not rp.is_dir():
                continue
            out.append(rp)
            seen.add(rp)
        return sorted(out)

    out = []
    if RUNS_BENCH_ROOT.exists():
        for cand in sorted(RUNS_BENCH_ROOT.glob(bench_glob)):
            if not cand.is_dir():
                continue
            if cand.name.startswith("block3_preflight_"):
                continue
            if not cand.name.startswith("block3_20260203_225620"):
                continue
            out.append(cand.resolve())
    return out


def _to_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def _safe_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=True, sort_keys=True)


def _load_metrics(bench_dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bdir in bench_dirs:
        for mf in sorted(bdir.rglob("metrics.json")):
            try:
                payload = json.loads(mf.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                recs = payload.get("results", []) or []
            elif isinstance(payload, list):
                recs = payload
            else:
                recs = []
            try:
                rel = str(mf.relative_to(ROOT))
            except Exception:
                rel = str(mf)
            for rec in recs:
                if not isinstance(rec, dict):
                    continue
                item = dict(rec)
                item["_source_path"] = rel
                item["_bench_dir"] = bdir.name
                rows.append(item)
    return rows


def _strict_rows(rows: Iterable[Dict[str, Any]], min_coverage: float = 0.98) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        mae = _to_float(r.get("mae"))
        cov = _to_float(r.get("prediction_coverage_ratio"))
        if mae is None:
            continue
        if r.get("fairness_pass") is True and cov is not None and cov >= min_coverage:
            out.append(r)
    return out


def _target_family(target: str) -> str:
    if target == "investors_count":
        return "count"
    if target == "funding_raised_usd":
        return "heavy_tail"
    if target == "is_funded":
        return "binary"
    return "unknown"


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    arr = sorted(vals)
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return float(arr[mid])
    return float((arr[mid - 1] + arr[mid]) * 0.5)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _search_space_templates() -> Dict[str, Dict[str, Any]]:
    return {
        "LightGBM": {
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [63, 127, 255],
            "min_child_samples": [20, 40, 80],
            "feature_fraction": [0.7, 0.85, 1.0],
            "bagging_fraction": [0.7, 0.9, 1.0],
        },
        "XGBoost": {
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [6, 8, 10],
            "min_child_weight": [1.0, 2.0, 4.0],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        },
        "CatBoost": {
            "learning_rate": [0.01, 0.03, 0.05],
            "depth": [6, 8, 10],
            "l2_leaf_reg": [3.0, 6.0, 12.0],
            "bagging_temperature": [0.0, 1.0, 2.0],
        },
        "XGBoostPoisson": {
            "learning_rate": [0.01, 0.03, 0.05],
            "max_depth": [4, 6, 8],
            "min_child_weight": [1.0, 2.0, 4.0],
            "subsample": [0.7, 0.9, 1.0],
        },
        "LightGBMTweedie": {
            "learning_rate": [0.01, 0.03, 0.05],
            "num_leaves": [63, 127, 255],
            "tweedie_variance_power": [1.1, 1.3, 1.5],
            "min_child_samples": [20, 40, 80],
        },
        "RandomForest": {
            "n_estimators": [600, 900, 1200],
            "max_depth": [None, 20, 32],
            "min_samples_leaf": [1, 2, 4],
        },
        "NBEATS": {
            "max_steps": [1000, 2000, 3000],
            "learning_rate": [1e-4, 3e-4, 1e-3],
            "dropout_prob_theta": [0.0, 0.1, 0.2],
        },
        "NHITS": {
            "max_steps": [1000, 2000, 3000],
            "learning_rate": [1e-4, 3e-4, 1e-3],
            "dropout_prob_theta": [0.0, 0.1, 0.2],
        },
        "PatchTST": {
            "max_steps": [1000, 2000, 3000],
            "learning_rate": [1e-4, 3e-4, 1e-3],
            "dropout": [0.0, 0.1, 0.2],
            "hidden_size": [64, 128, 256],
        },
        "iTransformer": {
            "max_steps": [1000, 2000, 3000],
            "learning_rate": [1e-4, 3e-4, 1e-3],
            "dropout": [0.0, 0.1, 0.2],
            "d_model": [64, 128, 256],
        },
        "Chronos": {
            "prediction_length": [1, 7, 14, 30],
            "num_samples": [20, 50, 100],
            "temperature": [0.7, 0.9, 1.0],
        },
        "Moirai2": {
            "prediction_length": [1, 7, 14, 30],
            "num_samples": [20, 50, 100],
            "temperature": [0.7, 0.9, 1.0],
        },
        "TabPFNClassifier": {
            "n_estimators": [1],
            "ensemble_size": [8, 16, 32],
            "subsample": [5000, 10000, 20000],
        },
        "TabPFNRegressor": {
            "n_estimators": [1],
            "ensemble_size": [8, 16, 32],
            "subsample": [5000, 10000, 20000],
        },
        "AutoFitV72": {
            "top_k": [6, 8, 10],
            "search_budget": [96],
            "dynamic_weighting": [True],
            "count_safe_mode": [True],
            "champion_anchor": [True],
        },
    }


def _default_best_config(model_name: str, target_family: str) -> Dict[str, Any]:
    # Evidence-first defaults (no test feedback): stable center points from each grid.
    defaults: Dict[str, Dict[str, Any]] = {
        "LightGBM": {
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 40,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.9,
        },
        "XGBoost": {
            "learning_rate": 0.03,
            "max_depth": 8,
            "min_child_weight": 2.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        "CatBoost": {
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 6.0,
            "bagging_temperature": 1.0,
        },
        "XGBoostPoisson": {
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 2.0,
            "subsample": 0.9,
        },
        "LightGBMTweedie": {
            "learning_rate": 0.03,
            "num_leaves": 127,
            "tweedie_variance_power": 1.3 if target_family == "count" else 1.5,
            "min_child_samples": 40,
        },
        "RandomForest": {
            "n_estimators": 900,
            "max_depth": None,
            "min_samples_leaf": 2,
        },
        "NBEATS": {
            "max_steps": 2000,
            "learning_rate": 3e-4,
            "dropout_prob_theta": 0.1,
        },
        "NHITS": {
            "max_steps": 2000,
            "learning_rate": 3e-4,
            "dropout_prob_theta": 0.1,
        },
        "PatchTST": {
            "max_steps": 2000,
            "learning_rate": 3e-4,
            "dropout": 0.1,
            "hidden_size": 128,
        },
        "iTransformer": {
            "max_steps": 2000,
            "learning_rate": 3e-4,
            "dropout": 0.1,
            "d_model": 128,
        },
        "Chronos": {
            "num_samples": 50,
            "temperature": 0.9,
        },
        "Moirai2": {
            "num_samples": 50,
            "temperature": 0.9,
        },
        "TabPFNClassifier": {
            "ensemble_size": 16,
            "subsample": 10000,
        },
        "TabPFNRegressor": {
            "ensemble_size": 16,
            "subsample": 10000,
        },
        "AutoFitV72": {
            "top_k": 8,
            "search_budget": 96,
            "dynamic_weighting": True,
            "count_safe_mode": True,
            "champion_anchor": True,
        },
    }
    return dict(defaults.get(model_name, {}))


def _lane_model_priority() -> Dict[str, List[str]]:
    return {
        "count": [
            "AutoFitV72",
            "NHITS",
            "NBEATS",
            "PatchTST",
            "iTransformer",
            "XGBoostPoisson",
            "LightGBMTweedie",
            "LightGBM",
            "XGBoost",
            "CatBoost",
            "Moirai2",
            "Chronos",
        ],
        "heavy_tail": [
            "AutoFitV72",
            "PatchTST",
            "NHITS",
            "NBEATS",
            "Chronos",
            "Moirai2",
            "LightGBM",
            "XGBoost",
            "CatBoost",
            "TabPFNRegressor",
        ],
        "binary": [
            "AutoFitV72",
            "PatchTST",
            "NHITS",
            "LightGBM",
            "XGBoost",
            "CatBoost",
            "TabPFNClassifier",
        ],
    }


def _build_cost_report(strict_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {"train": [], "infer": []}
    )
    sources: Dict[Tuple[str, str, str], str] = {}
    for r in strict_rows:
        model = str(r.get("model_name", ""))
        category = str(r.get("category", ""))
        target = str(r.get("target", ""))
        if not model or not category or not target:
            continue
        k = (model, category, target)
        t = _to_float(r.get("train_time_seconds"))
        p = _to_float(r.get("inference_time_seconds"))
        if t is not None:
            buckets[k]["train"].append(t)
        if p is not None:
            buckets[k]["infer"].append(p)
        if k not in sources:
            sources[k] = str(r.get("_source_path", ""))

    rows_out: List[Dict[str, Any]] = []
    for (model, category, target), stats in sorted(buckets.items()):
        train_med = _median(stats["train"])
        infer_med = _median(stats["infer"])
        recs = max(len(stats["train"]), len(stats["infer"]), 1)
        rows_out.append(
            {
                "model_name": model,
                "category": category,
                "target": target,
                "strict_records": recs,
                "train_time_median_seconds": train_med,
                "inference_time_median_seconds": infer_med,
                "evidence_path": sources.get((model, category, target), ""),
            }
        )
    return rows_out


def _build_plan_artifacts(
    strict_rows: Sequence[Dict[str, Any]],
    search_budget: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    space_templates = _search_space_templates()
    lane_priority = _lane_model_priority()
    cost_rows = _build_cost_report(strict_rows)

    best_by_model_target: Dict[Tuple[str, str], Dict[str, Any]] = {}
    category_by_model: Dict[str, str] = {}
    evidence_by_model: Dict[str, str] = {}

    for r in strict_rows:
        model = str(r.get("model_name", ""))
        target = str(r.get("target", ""))
        mae = _to_float(r.get("mae"))
        if not model or not target or mae is None:
            continue
        key = (model, target)
        prev = best_by_model_target.get(key)
        if prev is None or mae < float(prev["mae"]):
            best_by_model_target[key] = {
                "mae": mae,
                "source": str(r.get("_source_path", "")),
            }
            category_by_model[model] = str(r.get("category", ""))
            evidence_by_model[model] = str(r.get("_source_path", ""))

    ledger_rows: List[Dict[str, Any]] = []
    best_config_json: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "train_val_oof_only",
        "search_budget_per_model": int(search_budget),
        "targets": {},
    }

    lane_order = {"count": 0, "heavy_tail": 1, "binary": 2, "unknown": 9}
    for target in sorted({str(r.get("target", "")) for r in strict_rows if r.get("target")}):
        family = _target_family(target)
        model_priority = lane_priority.get(family, [])
        target_payload: Dict[str, Any] = {}
        for rank, model_name in enumerate(model_priority, start=1):
            if model_name not in space_templates:
                continue
            base = best_by_model_target.get((model_name, target))
            status = "planned"
            trials_executed = 0
            best_mae_observed = None
            source = evidence_by_model.get(model_name, "")
            if base is not None:
                status = "planned_with_evidence"
                best_mae_observed = float(base["mae"])
                source = base["source"]

            best_cfg = _default_best_config(model_name, family)
            target_payload[model_name] = {
                "target_family": family,
                "category": category_by_model.get(model_name, "unknown"),
                "status": status,
                "search_budget": int(search_budget),
                "trials_executed": trials_executed,
                "best_config": best_cfg,
                "best_mae_observed_strict": best_mae_observed,
                "search_space": space_templates[model_name],
                "evidence_path": source or "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
            }

            ledger_rows.append(
                {
                    "target": target,
                    "target_family": family,
                    "priority_rank": rank,
                    "model_name": model_name,
                    "category": category_by_model.get(model_name, "unknown"),
                    "search_budget": int(search_budget),
                    "trials_executed": trials_executed,
                    "status": status,
                    "best_mae_observed_strict": best_mae_observed,
                    "search_space_json": _safe_json(space_templates[model_name]),
                    "best_config_json": _safe_json(best_cfg),
                    "selection_scope": "train_val_oof_only",
                    "evidence_path": source or "docs/benchmarks/block3_truth_pack/condition_leaderboard.csv",
                }
            )

        best_config_json["targets"][target] = target_payload

    ledger_rows = sorted(
        ledger_rows,
        key=lambda r: (
            lane_order.get(str(r["target_family"]), 99),
            str(r["target"]),
            int(r["priority_rank"]),
            str(r["model_name"]),
        ),
    )
    return ledger_rows, best_config_json, cost_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build V7.2 hyperparameter search artifacts.")
    p.add_argument("--bench-dir", action="append", default=[], help="Benchmark directory (repeatable).")
    p.add_argument("--bench-glob", type=str, default=DEFAULT_BENCH_GLOB, help="Glob under runs/benchmarks.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    p.add_argument("--search-budget", type=int, default=96, help="Trials budget per model.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bench_dirs = _resolve_bench_dirs([Path(p) for p in args.bench_dir], bench_glob=args.bench_glob)
    rows = _load_metrics(bench_dirs)
    strict = _strict_rows(rows, min_coverage=0.98)

    ledger_rows, best_cfg_json, cost_rows = _build_plan_artifacts(
        strict_rows=strict,
        search_budget=int(args.search_budget),
    )

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "hyperparam_search_ledger.csv"
    best_cfg_path = out_dir / "best_config_by_model_target.json"
    cost_path = out_dir / "compute_cost_report.csv"

    _write_csv(
        ledger_path,
        ledger_rows,
        [
            "target",
            "target_family",
            "priority_rank",
            "model_name",
            "category",
            "search_budget",
            "trials_executed",
            "status",
            "best_mae_observed_strict",
            "search_space_json",
            "best_config_json",
            "selection_scope",
            "evidence_path",
        ],
    )
    best_cfg_path.write_text(json.dumps(best_cfg_json, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_csv(
        cost_path,
        cost_rows,
        [
            "model_name",
            "category",
            "target",
            "strict_records",
            "train_time_median_seconds",
            "inference_time_median_seconds",
            "evidence_path",
        ],
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bench_dirs": [str(p) for p in bench_dirs],
        "strict_records": len(strict),
        "search_budget": int(args.search_budget),
        "outputs": {
            "hyperparam_search_ledger": str(ledger_path),
            "best_config_by_model_target": str(best_cfg_path),
            "compute_cost_report": str(cost_path),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
