#!/usr/bin/env python3
"""Diagnose residual/tail structure behind Round-14 coverage failures."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from r14_interval_pilot_lib import ROOT, load_predictions, write_report


def _describe(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    data = load_predictions(max_files=8, max_rows_per_group=50000)
    rows = []
    for (model, horizon, target), group in data.groupby(["model", "horizon", "target"], sort=True):
        if len(group) < 1000:
            continue
        y_pred = group["y_pred"].to_numpy(dtype=float)
        abs_resid = group["abs_residual"].to_numpy(dtype=float)
        quantiles = np.unique(np.quantile(y_pred, np.linspace(0.0, 1.0, 6)))
        decile_rows = []
        if len(quantiles) > 2:
            bins = np.digitize(y_pred, quantiles[1:-1], right=False)
            global_q90 = float(np.quantile(abs_resid, 0.90))
            for bin_id in range(len(quantiles) - 1):
                vals = abs_resid[bins == bin_id]
                if vals.size == 0:
                    continue
                decile_rows.append({
                    "bin": int(bin_id),
                    "n": int(vals.size),
                    "abs_residual": _describe(vals),
                    "share_above_global_q90": float(np.mean(vals > global_q90)),
                })
        rows.append({
            "model": str(model),
            "horizon": int(horizon) if horizon == horizon else None,
            "target": str(target),
            "n_rows": int(len(group)),
            "residual": _describe(group["residual"]),
            "abs_residual": _describe(abs_resid),
            "positive_target_rate": float(np.mean(group["y_true"].to_numpy(dtype=float) > 0.0)),
            "y_pred_bin_diagnostics": decile_rows,
        })
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "scope": "diagnostic_from_existing_predictions_not_temporal_validation",
        "n_prediction_rows_loaded": int(len(data)),
        "n_groups": int(len(rows)),
        "sigma_hat_available_in_predictions": False,
        "sigma_hat_note": "Existing prediction artifacts do not store sigma_hat; run a dedicated nccopo_inputs audit to validate studentized scaling quality.",
        "rows": rows,
        "limitations": [
            "Existing prediction artifacts lack stable row keys and sigma_hat.",
            "This diagnoses residual/tail concentration only; it does not prove a final interval method.",
        ],
    }
    out_json, out_md = write_report(report, "r14_sigma_residual_diagnostic")
    print(f"OK: {out_json}")
    print(f"OK: {out_md}")
    print(json.dumps({k: report[k] for k in ("scope", "n_prediction_rows_loaded", "n_groups", "sigma_hat_available_in_predictions", "sigma_hat_note")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())