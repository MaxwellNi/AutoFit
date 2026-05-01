#!/usr/bin/env python3
"""Audit whether strict text/EDGAR counterfactual evidence exists.

This script distinguishes two evidence levels:
1. Paired ablation proxy: aggregate core_text/core_edgar/full vs core_only.
2. Strict row-key audit: predictions share model/task/target/horizon and the
    same entity/date/source_row_index rows, allowing identical-row deltas.
"""

from __future__ import annotations

import glob
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_text_counterfactual_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")


def _latest_text_audit():
    paths = sorted(glob.glob(str(ROOT / "runs/audits/r14_text_edgar_signal_audit_*.json")))
    if not paths:
        return None
    return json.load(open(paths[-1]))


KEY_COLS = ["model", "task", "target", "horizon", "entity_id", "crawled_date_day", "source_row_index"]
LOOSE_KEY_COLS = ["model", "task", "target", "horizon", "entity_id", "crawled_date_day"]
VALUE_COLS = ["y_true", "y_pred", "ablation"]


def _rowkey_prediction_files() -> list[Path]:
    files: list[Path] = []
    for path_str in glob.glob(str(ROOT / "runs/benchmarks/r14fcast*/predictions.parquet")):
        path = Path(path_str)
        try:
            import pyarrow.parquet as pq
            names = set(pq.read_schema(path).names)
        except Exception:
            continue
        if set(KEY_COLS + VALUE_COLS).issubset(names):
            files.append(path)
    return sorted(files)


def _read_prediction_sample(path: Path, max_rows_per_file: int) -> pd.DataFrame:
    cols = KEY_COLS + VALUE_COLS
    frame = pd.read_parquet(path, columns=cols)
    frame["crawled_date_day"] = pd.to_datetime(frame["crawled_date_day"], errors="coerce").dt.strftime("%Y-%m-%d")
    if max_rows_per_file > 0 and len(frame) > max_rows_per_file:
        threshold = max(1, int((max_rows_per_file / len(frame)) * np.iinfo("uint64").max))
        hashes = pd.util.hash_pandas_object(frame[KEY_COLS], index=False).to_numpy(dtype=np.uint64, copy=False)
        frame = frame.loc[hashes <= threshold].copy()
    frame["_pred_path"] = str(path)
    return frame


def _summarize(values: pd.Series) -> dict[str, float | None]:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def _overlap_counts_by_horizon(base: pd.DataFrame, alt: pd.DataFrame, keys: list[str]) -> dict[str, int]:
    if base.empty or alt.empty:
        return {}
    base_keys = base[keys].drop_duplicates()
    alt_keys = alt[keys].drop_duplicates()
    joined = base_keys.merge(alt_keys, on=keys, how="inner")
    if joined.empty:
        return {}
    return {str(int(k)): int(v) for k, v in joined.groupby("horizon", observed=True).size().sort_index().items()}


def audit_strict_rowkey_counterfactual(max_rows_per_file: int = 250_000) -> dict:
    files = _rowkey_prediction_files()
    if not files:
        return {
            "status": "not_passed",
            "reason": "no_predictions_with_row_keys",
            "rowkey_prediction_files": [],
            "comparisons": [],
        }
    frames = []
    for path in files:
        try:
            frames.append(_read_prediction_sample(path, max_rows_per_file=max_rows_per_file))
        except Exception as exc:
            frames.append(pd.DataFrame({"_read_error": [f"{path}: {type(exc).__name__}:{exc}"]}))
    data = pd.concat([f for f in frames if not f.empty and "_read_error" not in f.columns], ignore_index=True) if frames else pd.DataFrame()
    if data.empty:
        return {
            "status": "not_passed",
            "reason": "rowkey_files_unreadable_or_empty",
            "rowkey_prediction_files": [str(path) for path in files],
            "comparisons": [],
        }
    comparisons = []
    for ablation in ("core_text", "core_edgar", "full"):
        base = data[data["ablation"] == "core_only"].copy()
        alt = data[data["ablation"] == ablation].copy()
        loose_overlap_by_horizon = _overlap_counts_by_horizon(base, alt, LOOSE_KEY_COLS)
        strict_key_overlap_by_horizon = _overlap_counts_by_horizon(base, alt, KEY_COLS)
        loose_overlap = int(sum(loose_overlap_by_horizon.values()))
        strict_key_overlap = int(sum(strict_key_overlap_by_horizon.values()))
        if base.empty or alt.empty:
            comparisons.append({
                "ablation": ablation,
                "status": "missing_pair",
                "n_overlap": 0,
                "n_overlap_without_source_row_index": loose_overlap,
                "overlap_by_horizon": strict_key_overlap_by_horizon,
                "overlap_without_source_row_index_by_horizon": loose_overlap_by_horizon,
            })
            continue
        merged = base.merge(
            alt,
            on=KEY_COLS,
            suffixes=("_core_only", f"_{ablation}"),
            how="inner",
        )
        if merged.empty:
            comparisons.append({
                "ablation": ablation,
                "status": "no_rowkey_overlap",
                "n_overlap": 0,
                "n_overlap_without_source_row_index": loose_overlap,
                "overlap_by_horizon": strict_key_overlap_by_horizon,
                "overlap_without_source_row_index_by_horizon": loose_overlap_by_horizon,
            })
            continue
        y_true_equal = np.isclose(
            pd.to_numeric(merged["y_true_core_only"], errors="coerce"),
            pd.to_numeric(merged[f"y_true_{ablation}"], errors="coerce"),
            equal_nan=True,
        )
        pred_delta = pd.to_numeric(merged[f"y_pred_{ablation}"], errors="coerce") - pd.to_numeric(merged["y_pred_core_only"], errors="coerce")
        abs_err_base = (pd.to_numeric(merged["y_pred_core_only"], errors="coerce") - pd.to_numeric(merged["y_true_core_only"], errors="coerce")).abs()
        abs_err_alt = (pd.to_numeric(merged[f"y_pred_{ablation}"], errors="coerce") - pd.to_numeric(merged[f"y_true_{ablation}"], errors="coerce")).abs()
        comparisons.append({
            "ablation": ablation,
            "status": "passed" if len(merged) >= 1000 and float(np.mean(y_true_equal)) >= 0.999 else "partial",
            "n_overlap": int(len(merged)),
            "n_overlap_unique_strict_keys": strict_key_overlap,
            "n_overlap_without_source_row_index": loose_overlap,
            "overlap_by_horizon": strict_key_overlap_by_horizon,
            "overlap_without_source_row_index_by_horizon": loose_overlap_by_horizon,
            "y_true_equal_rate": float(np.mean(y_true_equal)),
            "model_count": int(merged["model"].nunique()),
            "horizons": sorted([int(x) for x in merged["horizon"].dropna().unique()]),
            "prediction_delta": _summarize(pred_delta),
            "abs_error_delta_alt_minus_core_only": _summarize(abs_err_alt - abs_err_base),
        })
    comparable = [item for item in comparisons if item.get("status") in {"passed", "partial"}]
    all_passed = len(comparable) == 3 and all(item.get("status") == "passed" for item in comparable)
    any_partial = any(item.get("status") in {"passed", "partial"} for item in comparisons)
    return {
        "status": "passed" if all_passed else ("partial" if any_partial else "not_passed"),
        "max_rows_per_file": max_rows_per_file,
        "rowkey_prediction_files": [str(path) for path in files],
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit strict row-key text/EDGAR counterfactual evidence")
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=250_000,
        help="Stable hash sample cap per predictions parquet; use 0 for a full audit.",
    )
    args = parser.parse_args()
    text_audit = _latest_text_audit() or {}
    paired = text_audit.get("result_delta_audit", {}).get("summary", {}).get("by_ablation", {})
    strict = audit_strict_rowkey_counterfactual(max_rows_per_file=args.max_rows_per_file)
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": strict.get("status", "not_passed"),
        "paired_ablation_proxy_available": bool(paired),
        "paired_ablation_proxy": paired,
        "strict_rowkey_counterfactual": strict,
        "strict_counterfactual_artifacts": sorted(glob.glob(str(ROOT / "runs/audits/*text*counterfactual*.json"))),
        "remaining_limitations": [
            "Strict row-key overlap proves same-row ablation differences, not controlled token-level text perturbation.",
            "A full pass still needs the newly submitted core_text/core_edgar/full row-key jobs to land across h7/h14/h30.",
            "Token/embedding perturbation remains a separate future audit.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
    OUT_MD.write_text("# R14 Text Counterfactual Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"OK: {OUT_JSON}")
    print(f"OK: {OUT_MD}")
    print(json.dumps({
        "status": report["status"],
        "paired_ablation_proxy_available": report["paired_ablation_proxy_available"],
        "strict_status": strict.get("status"),
        "n_rowkey_files": len(strict.get("rowkey_prediction_files", [])),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())