#!/usr/bin/env python3
"""Run an auditable Bike Sharing public event-proxy validation pilot.

Bike Sharing Demand is not a replacement for M5 or a credentialed medical
event panel. Its role is narrower: it is a public, credential-free time-series
panel with explicit event/state proxies (holiday, workingday, weather, hour)
and a count target. The artifact therefore records scoped evidence instead of
promoting a broad cross-domain generalization claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_bikeshare_external_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")
OUT_DATA = ROOT / "runs" / "external_validation" / "bikeshare" / "bike_sharing_demand.parquet"

OPENML_ID = 44063
RANDOM_STATE = 42


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_bikeshare() -> pd.DataFrame:
    cache = ROOT / "runs" / "external_validation" / "openml_cache"
    cache.mkdir(parents=True, exist_ok=True)
    frame = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="pandas", data_home=str(cache)).frame
    data = frame.copy().reset_index(drop=True)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["row_time_id"] = np.arange(len(data), dtype=np.int64)
    return data


def _split_chronological(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = data.sort_values("row_time_id").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.60)
    cal_end = int(n * 0.80)
    return {
        "train": ordered.iloc[:train_end].copy(),
        "cal": ordered.iloc[train_end:cal_end].copy(),
        "test": ordered.iloc[cal_end:].copy(),
    }


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=RANDOM_STATE,
    )


def _asymmetric_interval(cal_y: np.ndarray, cal_pred: np.ndarray, test_pred: np.ndarray, alpha: float = 0.10):
    lower_resid = np.clip(cal_pred - cal_y, 0.0, None)
    upper_resid = np.clip(cal_y - cal_pred, 0.0, None)
    q_lower = float(np.quantile(lower_resid, 1.0 - alpha / 2.0))
    q_upper = float(np.quantile(upper_resid, 1.0 - alpha / 2.0))
    return test_pred - q_lower, test_pred + q_upper, q_lower, q_upper


def _symmetric_interval(cal_y: np.ndarray, cal_pred: np.ndarray, test_pred: np.ndarray, alpha: float = 0.10):
    q = float(np.quantile(np.abs(cal_y - cal_pred), 1.0 - alpha))
    return test_pred - q, test_pred + q, q


def _coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def _fit_predict(splits: dict[str, pd.DataFrame], feature_cols: list[str]) -> dict[str, np.ndarray | float]:
    target = "count"
    train = splits["train"].dropna(subset=feature_cols + [target]).copy()
    cal = splits["cal"].dropna(subset=feature_cols + [target]).copy()
    test = splits["test"].dropna(subset=feature_cols + [target]).copy()

    reg = _model()
    reg.fit(train[feature_cols], np.log1p(train[target].to_numpy(dtype=float)))
    cal_pred = np.expm1(reg.predict(cal[feature_cols])).clip(min=0.0)
    test_pred = np.expm1(reg.predict(test[feature_cols])).clip(min=0.0)
    cal_y = cal[target].to_numpy(dtype=float)
    test_y = test[target].to_numpy(dtype=float)
    return {
        "n_train": int(len(train)),
        "n_cal": int(len(cal)),
        "n_test": int(len(test)),
        "cal_y": cal_y,
        "cal_pred": cal_pred,
        "test_y": test_y,
        "test_pred": test_pred,
        "mae": float(mean_absolute_error(test_y, test_pred)),
        "rmse": float(mean_squared_error(test_y, test_pred) ** 0.5),
    }


def _evaluate(data: pd.DataFrame, *, max_rows: int | None) -> dict:
    if max_rows is not None and len(data) > max_rows:
        data = data.sort_values("row_time_id").iloc[:max_rows].copy()
    splits = _split_chronological(data)

    core_cols = ["season", "year", "month", "hour", "temp", "feel_temp", "humidity", "windspeed"]
    event_cols = core_cols + ["holiday", "workingday", "weather"]
    target = "count"

    core = _fit_predict(splits, core_cols)
    event = _fit_predict(splits, event_cols)
    test_y = event["test_y"]
    cal_y = event["cal_y"]
    cal_pred = event["cal_pred"]
    test_pred = event["test_pred"]

    lo_m, hi_m, q_m = _symmetric_interval(cal_y, cal_pred, test_pred)
    lo_c, hi_c, q_l, q_u = _asymmetric_interval(cal_y, cal_pred, test_pred)
    positive = data.loc[data[target] > 0, target].to_numpy(dtype=float)
    event_mae_delta = float(core["mae"] - event["mae"])
    event_mae_delta_pct = float(event_mae_delta / core["mae"] * 100.0) if core["mae"] else None
    return {
        "target": target,
        "core_feature_cols": core_cols,
        "event_state_feature_cols": event_cols,
        "event_proxy_cols": ["holiday", "workingday", "weather", "hour", "month"],
        "split_protocol": "deterministic original-row chronological split: 60% train, 20% calibration, 20% test",
        "n_rows": int(len(data)),
        "n_train": int(event["n_train"]),
        "n_cal": int(event["n_cal"]),
        "n_test": int(event["n_test"]),
        "zero_rate": float(np.mean(data[target].to_numpy(dtype=float) == 0.0)),
        "positive_count": int(len(positive)),
        "positive_tail_quantiles": {
            str(q): float(np.quantile(positive, q)) for q in (0.5, 0.9, 0.95, 0.99)
        } if len(positive) else {},
        "core_point_mae": float(core["mae"]),
        "event_state_point_mae": float(event["mae"]),
        "event_state_mae_delta_vs_core": event_mae_delta,
        "event_state_mae_delta_pct_vs_core": event_mae_delta_pct,
        "event_state_point_rmse": float(event["rmse"]),
        "marginal_coverage_90": _coverage(test_y, lo_m, hi_m),
        "marginal_width_mean": float(np.mean(hi_m - lo_m)),
        "marginal_q_90": q_m,
        "cqr_lite_coverage_90": _coverage(test_y, lo_c, hi_c),
        "cqr_lite_width_mean": float(np.mean(hi_c - lo_c)),
        "cqr_lite_q_lower_95": q_l,
        "cqr_lite_q_upper_95": q_u,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None, help="Optional deterministic prefix cap for quick reruns.")
    args = parser.parse_args()

    fetch_error = None
    data = None
    validation = None
    try:
        data = _fetch_bikeshare()
        OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(OUT_DATA, index=False)
        validation = _evaluate(data, max_rows=args.max_rows)
    except Exception as exc:  # pragma: no cover - external network/env dependent
        fetch_error = f"{type(exc).__name__}: {exc}"

    data_hash = _sha256_file(OUT_DATA) if OUT_DATA.exists() else None
    cqr = None if validation is None else validation.get("cqr_lite_coverage_90")
    event_delta = None if validation is None else validation.get("event_state_mae_delta_vs_core")
    status = "partial" if validation is not None else "not_passed"
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": status,
        "scope_status": {
            "public_dataset_acquired": data is not None,
            "public_event_proxy_pilot": bool(validation is not None),
            "event_state_point_improves_over_core": bool(event_delta is not None and event_delta > 0.0),
            "cqr_lite_near_90_coverage": bool(cqr is not None and cqr >= 0.88),
            "heavy_tail_financial_claim": False,
            "broad_cross_industry_claim": False,
        },
        "dataset": {
            "name": "Bike_Sharing_Demand",
            "source": "OpenML",
            "data_id": OPENML_ID,
            "local_parquet": str(OUT_DATA) if OUT_DATA.exists() else None,
            "local_parquet_sha256": data_hash,
            "n_rows": None if data is None else int(len(data)),
            "columns": None if data is None else list(map(str, data.columns)),
        },
        "validation": validation,
        "fetch_error": fetch_error,
        "required_for_pass": [
            "Public dataset present locally with source ID and hash recorded.",
            "Deterministic chronological split protocol documented and rerunnable.",
            "Core-vs-event-state paired point forecast comparison reported.",
            "Marginal and CQR-lite conformal coverage audit generated.",
            "For broad generalization: add at least one heavier-tailed event-rich public panel such as M5/calendar or a credentialed clinical event panel.",
        ],
        "interpretation": "This is real public event-proxy evidence for the event-state adapter and conformal layer. It is not a heavy-tail financial or universal forecasting proof.",
        "next_action": "Use this as a public event-proxy rung; keep M5 or another event-rich heavy-tail panel as the next generalization blocker.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
    OUT_MD.write_text("# R14 Bike Sharing External Validation Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"OK: {OUT_JSON}")
    print(f"OK: {OUT_MD}")
    print(json.dumps({
        "status": report["status"],
        "public_dataset_acquired": report["scope_status"]["public_dataset_acquired"],
        "event_state_point_improves_over_core": report["scope_status"]["event_state_point_improves_over_core"],
        "cqr_lite_near_90_coverage": report["scope_status"]["cqr_lite_near_90_coverage"],
        "n_rows": report["dataset"]["n_rows"],
        "event_state_mae_delta_vs_core": None if validation is None else validation.get("event_state_mae_delta_vs_core"),
        "cqr_lite_coverage_90": cqr,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())