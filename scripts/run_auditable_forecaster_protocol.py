#!/usr/bin/env python3
"""Wrap arbitrary forecaster predictions with an auditable conformal layer.

Input is a CSV or parquet file with at least:
  row key column, split column, true target column, prediction column.

The script fits the conformal layer on calibration rows and emits row-keyed
test predictions with intervals plus an audit JSON/Markdown summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from narrative.block3.auditable_forecaster import (
    AuditableConformalProtocolLayer,
    AuditableProtocolConfig,
)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported input suffix: {path.suffix}; use csv or parquet")


def _optional_array(frame: pd.DataFrame, column: str | None) -> np.ndarray | None:
    if not column:
        return None
    if column not in frame.columns:
        raise ValueError(f"missing optional column: {column}")
    values = frame[column].to_numpy()
    if pd.isna(values).any():
        raise ValueError(f"optional column {column} contains missing values")
    return values


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")


def _metadata(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("--metadata-json must decode to an object")
    return data


def _finite_numeric(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite numeric for selected rows")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV/parquet predictions table")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for audit outputs")
    parser.add_argument("--row-key-col", default="row_key")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--calibration-split", default="cal")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--pred-col", default="y_pred")
    parser.add_argument("--sigma-col", default="")
    parser.add_argument("--group-col", default="")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--studentized", action="store_true")
    parser.add_argument("--metadata-json", default="{}")
    args = parser.parse_args()

    frame = _read_table(args.input)
    _require_columns(frame, [args.row_key_col, args.split_col, args.y_col, args.pred_col])
    if not (0.0 < float(args.alpha) < 1.0):
        raise ValueError("--alpha must be in (0, 1)")

    cal = frame.loc[frame[args.split_col].astype(str) == str(args.calibration_split)].copy()
    test = frame.loc[frame[args.split_col].astype(str) == str(args.test_split)].copy()
    if len(cal) < 2:
        raise ValueError("calibration split must contain at least 2 rows")
    if test.empty:
        raise ValueError("test split is empty")

    layer = AuditableConformalProtocolLayer(
        AuditableProtocolConfig(
            alpha=float(args.alpha),
            use_studentized=bool(args.studentized),
            row_key_name=args.row_key_col,
            metadata=_metadata(args.metadata_json),
        )
    )
    layer.fit(
        _finite_numeric(cal[args.y_col], args.y_col),
        _finite_numeric(cal[args.pred_col], args.pred_col),
        row_keys_cal=cal[args.row_key_col].tolist(),
        sigma_hat_cal=_optional_array(cal, args.sigma_col or None),
        groups_cal=_optional_array(cal, args.group_col or None),
    )
    result = layer.predict(
        _finite_numeric(test[args.pred_col], args.pred_col),
        row_keys=test[args.row_key_col].tolist(),
        sigma_hat=_optional_array(test, args.sigma_col or None),
        groups=_optional_array(test, args.group_col or None),
    )
    y_test = pd.to_numeric(test[args.y_col], errors="coerce").to_numpy(dtype=np.float64)
    score = None
    if np.all(np.isfinite(y_test)):
        score = layer.score(y_test, result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "auditable_predictions.csv"
    audit_path = args.output_dir / "auditable_protocol_audit.json"
    md_path = args.output_dir / "auditable_protocol_audit.md"
    result.frame.to_csv(prediction_path, index=False)
    audit = {
        "timestamp_cest": datetime.now().isoformat(),
        "input_path": str(args.input),
        "prediction_path": str(prediction_path),
        "status": "passed" if result.summary.get("prediction_row_keys_unique") else "not_passed",
        "config": {
            "alpha": float(args.alpha),
            "studentized": bool(args.studentized),
            "row_key_col": args.row_key_col,
            "split_col": args.split_col,
            "calibration_split": args.calibration_split,
            "test_split": args.test_split,
            "y_col": args.y_col,
            "pred_col": args.pred_col,
            "sigma_col": args.sigma_col or None,
            "group_col": args.group_col or None,
        },
        "fit_summary": result.summary,
        "test_score": score,
    }
    audit_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    md_path.write_text(
        "# Auditable Forecaster Protocol Audit\n\n```json\n"
        + json.dumps(audit, indent=2, default=str)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": audit["status"], "audit_path": str(audit_path), "prediction_path": str(prediction_path), "test_score": score}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())