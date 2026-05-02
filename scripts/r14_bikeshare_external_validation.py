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
    if OUT_DATA.exists():
        return pd.read_parquet(OUT_DATA).reset_index(drop=True)
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


def _quantile_model(quantile: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=160,
        learning_rate=0.05,
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


def _higher_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    level = min(1.0, np.ceil((len(values) + 1) * (1.0 - alpha)) / len(values))
    return float(np.quantile(values, level, method="higher"))


def _temporal_drift_guard_interval(
    cal_y: np.ndarray,
    cal_pred: np.ndarray,
    test_pred: np.ndarray,
    alpha: float = 0.10,
    holdout_fraction: float = 0.20,
) -> dict[str, float]:
    """Calibration-only temporal guard for exchangeability drift.

    The guard uses the first calibration block to estimate the nominal residual
    quantile, checks coverage on the later calibration holdout, then raises the
    final residual quantile level by the observed holdout undercoverage. Test
    labels are never used to choose the inflation.
    """
    residuals = np.abs(np.asarray(cal_y, dtype=float) - np.asarray(cal_pred, dtype=float))
    split = int(len(residuals) * (1.0 - holdout_fraction))
    split = min(max(split, 1), len(residuals) - 1)
    early = residuals[:split]
    late_y = np.asarray(cal_y, dtype=float)[split:]
    late_pred = np.asarray(cal_pred, dtype=float)[split:]
    nominal_level = 1.0 - alpha
    early_q = float(np.quantile(early, nominal_level, method="higher"))
    holdout_coverage = float(np.mean(np.abs(late_y - late_pred) <= early_q))
    adjusted_level = min(0.99, nominal_level + max(0.0, nominal_level - holdout_coverage))
    guarded_q = float(np.quantile(residuals, adjusted_level, method="higher"))
    test_pred = np.asarray(test_pred, dtype=float)
    return {
        "drift_guard_nominal_level": nominal_level,
        "drift_guard_holdout_fraction": holdout_fraction,
        "drift_guard_holdout_coverage_at_nominal": holdout_coverage,
        "drift_guard_adjusted_quantile_level": adjusted_level,
        "drift_guard_conformity_q": guarded_q,
        "drift_guard_lower": test_pred - guarded_q,
        "drift_guard_upper": test_pred + guarded_q,
    }


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


def _fit_quantile_cqr(splits: dict[str, pd.DataFrame], feature_cols: list[str], alpha: float = 0.10) -> dict[str, float]:
    target = "count"
    train = splits["train"].dropna(subset=feature_cols + [target]).copy()
    cal = splits["cal"].dropna(subset=feature_cols + [target]).copy()
    test = splits["test"].dropna(subset=feature_cols + [target]).copy()

    lower = _quantile_model(alpha / 2.0)
    upper = _quantile_model(1.0 - alpha / 2.0)
    y_train_log = np.log1p(train[target].to_numpy(dtype=float))
    lower.fit(train[feature_cols], y_train_log)
    upper.fit(train[feature_cols], y_train_log)

    cal_lo = np.expm1(lower.predict(cal[feature_cols])).clip(min=0.0)
    cal_hi = np.expm1(upper.predict(cal[feature_cols])).clip(min=0.0)
    test_lo = np.expm1(lower.predict(test[feature_cols])).clip(min=0.0)
    test_hi = np.expm1(upper.predict(test[feature_cols])).clip(min=0.0)
    cal_lo, cal_hi = np.minimum(cal_lo, cal_hi), np.maximum(cal_lo, cal_hi)
    test_lo, test_hi = np.minimum(test_lo, test_hi), np.maximum(test_lo, test_hi)
    cal_y = cal[target].to_numpy(dtype=float)
    test_y = test[target].to_numpy(dtype=float)
    scores = np.maximum(cal_lo - cal_y, cal_y - cal_hi)
    q = _higher_quantile(scores, alpha=alpha)
    lo = test_lo - q
    hi = test_hi + q
    return {
        "quantile_cqr_coverage_90": _coverage(test_y, lo, hi),
        "quantile_cqr_width_mean": float(np.mean(hi - lo)),
        "quantile_cqr_conformity_q_90": q,
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
    event_quantile = _fit_quantile_cqr(splits, event_cols)
    test_y = event["test_y"]
    cal_y = event["cal_y"]
    cal_pred = event["cal_pred"]
    test_pred = event["test_pred"]

    lo_m, hi_m, q_m = _symmetric_interval(cal_y, cal_pred, test_pred)
    lo_c, hi_c, q_l, q_u = _asymmetric_interval(cal_y, cal_pred, test_pred)
    drift_guard = _temporal_drift_guard_interval(cal_y, cal_pred, test_pred)
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
        "drift_guard_coverage_90": _coverage(test_y, drift_guard["drift_guard_lower"], drift_guard["drift_guard_upper"]),
        "drift_guard_width_mean": float(np.mean(drift_guard["drift_guard_upper"] - drift_guard["drift_guard_lower"])),
        "drift_guard_holdout_fraction": drift_guard["drift_guard_holdout_fraction"],
        "drift_guard_holdout_coverage_at_nominal": drift_guard["drift_guard_holdout_coverage_at_nominal"],
        "drift_guard_adjusted_quantile_level": drift_guard["drift_guard_adjusted_quantile_level"],
        "drift_guard_conformity_q": drift_guard["drift_guard_conformity_q"],
        **event_quantile,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None, help="Optional deterministic prefix cap for quick reruns.")
    args = parser.parse_args()

    fetch_error = None
    cache_write_error = None
    data = None
    validation = None
    try:
        data = _fetch_bikeshare()
        OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
        try:
            data.to_parquet(OUT_DATA, index=False)
        except PermissionError as exc:
            cache_write_error = f"{type(exc).__name__}: {exc}"
        validation = _evaluate(data, max_rows=args.max_rows)
    except Exception as exc:  # pragma: no cover - external network/env dependent
        fetch_error = f"{type(exc).__name__}: {exc}"

    data_hash = _sha256_file(OUT_DATA) if OUT_DATA.exists() else None
    cqr = None if validation is None else validation.get("cqr_lite_coverage_90")
    quantile_cqr = None if validation is None else validation.get("quantile_cqr_coverage_90")
    drift_guard = None if validation is None else validation.get("drift_guard_coverage_90")
    event_delta = None if validation is None else validation.get("event_state_mae_delta_vs_core")
    full_scope = args.max_rows is None
    status = "passed" if (
        full_scope
        and validation is not None
        and event_delta is not None
        and event_delta > 0.0
        and drift_guard is not None
        and drift_guard >= 0.88
    ) else ("partial" if validation is not None else "not_passed")
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": status,
        "scope_status": {
            "full_scope_run": full_scope,
            "max_rows": args.max_rows,
            "public_dataset_acquired": data is not None,
            "public_event_proxy_pilot": bool(validation is not None),
            "event_state_point_improves_over_core": bool(event_delta is not None and event_delta > 0.0),
            "cqr_lite_near_90_coverage": bool(cqr is not None and cqr >= 0.88),
            "quantile_cqr_near_90_coverage": bool(quantile_cqr is not None and quantile_cqr >= 0.88),
            "drift_guard_near_90_coverage": bool(drift_guard is not None and drift_guard >= 0.88),
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
        "cache_write_error": cache_write_error,
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
        "full_scope_run": full_scope,
        "event_state_point_improves_over_core": report["scope_status"]["event_state_point_improves_over_core"],
        "cqr_lite_near_90_coverage": report["scope_status"]["cqr_lite_near_90_coverage"],
        "n_rows": report["dataset"]["n_rows"],
        "event_state_mae_delta_vs_core": None if validation is None else validation.get("event_state_mae_delta_vs_core"),
        "cqr_lite_coverage_90": cqr,
        "quantile_cqr_coverage_90": quantile_cqr,
        "drift_guard_coverage_90": drift_guard,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())