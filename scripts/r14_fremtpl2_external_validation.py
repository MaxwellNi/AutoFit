#!/usr/bin/env python3
"""Run an auditable freMTPL2 public external-validation pilot.

The goal is deliberately narrow: freMTPL2 is a public insurance panel with
zero-inflated and heavy-tailed claim amounts. It is useful evidence for the
heavy-tail/coverage part of the method contract, but it is not an event-stream
or cross-industry proof by itself. The output therefore records a scoped status
instead of promoting a broad generalization claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "audits" / f"r14_fremtpl2_external_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
OUT_MD = OUT_JSON.with_suffix(".md")
OUT_DATA = ROOT / "runs" / "external_validation" / "fremtpl2" / "fremtpl2_policy_claims.parquet"

FREQ_OPENML_ID = 41214
SEV_OPENML_ID = 41215
RANDOM_STATE = 42


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_fremtpl2() -> pd.DataFrame:
    if OUT_DATA.exists():
        return pd.read_parquet(OUT_DATA).reset_index(drop=True)
    cache = ROOT / "runs" / "external_validation" / "openml_cache"
    cache.mkdir(parents=True, exist_ok=True)
    freq = fetch_openml(data_id=FREQ_OPENML_ID, as_frame=True, parser="pandas", data_home=str(cache)).frame
    sev = fetch_openml(data_id=SEV_OPENML_ID, as_frame=True, parser="pandas", data_home=str(cache)).frame

    freq = freq.copy()
    sev = sev.copy()
    freq["IDpol"] = pd.to_numeric(freq["IDpol"], errors="coerce").astype("Int64")
    sev["IDpol"] = pd.to_numeric(sev["IDpol"], errors="coerce").astype("Int64")
    sev["ClaimAmount"] = pd.to_numeric(sev["ClaimAmount"], errors="coerce")

    sev_agg = (
        sev.dropna(subset=["IDpol"])
        .groupby("IDpol", as_index=False)
        .agg(claim_amount_total=("ClaimAmount", "sum"), claim_amount_count=("ClaimAmount", "size"))
    )
    data = freq.merge(sev_agg, on="IDpol", how="left")
    data["claim_amount_total"] = data["claim_amount_total"].fillna(0.0)
    data["claim_amount_count"] = data["claim_amount_count"].fillna(0).astype(int)
    data["ClaimNb"] = pd.to_numeric(data["ClaimNb"], errors="coerce")
    data["Exposure"] = pd.to_numeric(data["Exposure"], errors="coerce")
    return data


def _split_by_policy_id(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = data.sort_values("IDpol").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * 0.60)
    cal_end = int(n * 0.80)
    return {
        "train": ordered.iloc[:train_end].copy(),
        "cal": ordered.iloc[train_end:cal_end].copy(),
        "test": ordered.iloc[cal_end:].copy(),
    }


def _build_model(categorical_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_cols,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=80,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("model", model)])


def _build_quantile_model(categorical_cols: list[str], quantile: float) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical_cols,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("model", model)])


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


def _evaluate_amount(data: pd.DataFrame, *, max_rows: int | None) -> dict:
    if max_rows is not None and len(data) > max_rows:
        data = data.sort_values("IDpol").iloc[:max_rows].copy()
    splits = _split_by_policy_id(data)

    feature_cols = [
        "Exposure",
        "Area",
        "VehPower",
        "VehAge",
        "DrivAge",
        "BonusMalus",
        "VehBrand",
        "VehGas",
        "Density",
        "Region",
    ]
    categorical_cols = ["Area", "VehBrand", "VehGas", "Region"]
    target = "claim_amount_total"

    train = splits["train"].dropna(subset=feature_cols + [target]).copy()
    cal = splits["cal"].dropna(subset=feature_cols + [target]).copy()
    test = splits["test"].dropna(subset=feature_cols + [target]).copy()
    pipe = _build_model(categorical_cols)
    pipe.fit(train[feature_cols], np.log1p(train[target].to_numpy(dtype=float)))

    cal_pred = np.expm1(pipe.predict(cal[feature_cols])).clip(min=0.0)
    test_pred = np.expm1(pipe.predict(test[feature_cols])).clip(min=0.0)
    cal_y = cal[target].to_numpy(dtype=float)
    test_y = test[target].to_numpy(dtype=float)

    lo_m, hi_m, q_m = _symmetric_interval(cal_y, cal_pred, test_pred)
    lo_c, hi_c, q_l, q_u = _asymmetric_interval(cal_y, cal_pred, test_pred)
    lower = _build_quantile_model(categorical_cols, 0.05)
    upper = _build_quantile_model(categorical_cols, 0.95)
    lower.fit(train[feature_cols], np.log1p(train[target].to_numpy(dtype=float)))
    upper.fit(train[feature_cols], np.log1p(train[target].to_numpy(dtype=float)))
    cal_lo = np.expm1(lower.predict(cal[feature_cols])).clip(min=0.0)
    cal_hi = np.expm1(upper.predict(cal[feature_cols])).clip(min=0.0)
    test_lo = np.expm1(lower.predict(test[feature_cols])).clip(min=0.0)
    test_hi = np.expm1(upper.predict(test[feature_cols])).clip(min=0.0)
    cal_lo, cal_hi = np.minimum(cal_lo, cal_hi), np.maximum(cal_lo, cal_hi)
    test_lo, test_hi = np.minimum(test_lo, test_hi), np.maximum(test_lo, test_hi)
    conformity = np.maximum(cal_lo - cal_y, cal_y - cal_hi)
    q_cqr = _higher_quantile(conformity, alpha=0.10)
    lo_q = test_lo - q_cqr
    hi_q = test_hi + q_cqr
    positive = data.loc[data[target] > 0, target].to_numpy(dtype=float)
    tail_q = {str(q): float(np.quantile(positive, q)) for q in (0.5, 0.9, 0.95, 0.99)} if len(positive) else {}
    return {
        "target": target,
        "feature_cols": feature_cols,
        "excluded_as_leakage": ["ClaimNb", "claim_amount_count"],
        "split_protocol": "deterministic policy-id ordered split: 60% train, 20% calibration, 20% test",
        "n_rows": int(len(data)),
        "n_train": int(len(train)),
        "n_cal": int(len(cal)),
        "n_test": int(len(test)),
        "zero_rate": float(np.mean(data[target].to_numpy(dtype=float) == 0.0)),
        "positive_count": int(len(positive)),
        "positive_tail_quantiles": tail_q,
        "point_mae": float(mean_absolute_error(test_y, test_pred)),
        "point_rmse": float(mean_squared_error(test_y, test_pred) ** 0.5),
        "marginal_coverage_90": _coverage(test_y, lo_m, hi_m),
        "marginal_width_mean": float(np.mean(hi_m - lo_m)),
        "marginal_q_90": q_m,
        "cqr_lite_coverage_90": _coverage(test_y, lo_c, hi_c),
        "cqr_lite_width_mean": float(np.mean(hi_c - lo_c)),
        "cqr_lite_q_lower_95": q_l,
        "cqr_lite_q_upper_95": q_u,
        "quantile_cqr_coverage_90": _coverage(test_y, lo_q, hi_q),
        "quantile_cqr_width_mean": float(np.mean(hi_q - lo_q)),
        "quantile_cqr_conformity_q_90": q_cqr,
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
        data = _fetch_fremtpl2()
        OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
        try:
            data.to_parquet(OUT_DATA, index=False)
        except PermissionError as exc:
            cache_write_error = f"{type(exc).__name__}: {exc}"
        validation = _evaluate_amount(data, max_rows=args.max_rows)
    except Exception as exc:  # pragma: no cover - external network/env dependent
        fetch_error = f"{type(exc).__name__}: {exc}"

    data_hash = _sha256_file(OUT_DATA) if OUT_DATA.exists() else None
    coverage = None if validation is None else validation.get("cqr_lite_coverage_90")
    quantile_cqr = None if validation is None else validation.get("quantile_cqr_coverage_90")
    full_scope = args.max_rows is None
    # Scope is intentionally partial: freMTPL2 validates public insurance
    # heavy-tail/zero-inflation coverage, not event-stream generalization.
    status = "passed" if (full_scope and validation is not None and quantile_cqr is not None and quantile_cqr >= 0.88) else ("partial" if validation is not None else "not_passed")
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": status,
        "scope_status": {
            "full_scope_run": full_scope,
            "max_rows": args.max_rows,
            "public_dataset_acquired": data is not None,
            "public_insurance_heavy_tail_pilot": bool(full_scope and validation is not None and ((coverage is not None and coverage >= 0.88) or (quantile_cqr is not None and quantile_cqr >= 0.88))),
            "quantile_cqr_near_90_coverage": bool(quantile_cqr is not None and quantile_cqr >= 0.88),
            "event_stream_generalization": False,
            "broad_cross_industry_claim": False,
        },
        "dataset": {
            "name": "freMTPL2",
            "source": "OpenML",
            "frequency_data_id": FREQ_OPENML_ID,
            "severity_data_id": SEV_OPENML_ID,
            "local_parquet": str(OUT_DATA) if OUT_DATA.exists() else None,
            "local_parquet_sha256": data_hash,
            "n_rows": None if data is None else int(len(data)),
            "columns": None if data is None else list(map(str, data.columns)),
        },
        "validation": validation,
        "fetch_error": fetch_error,
        "cache_write_error": cache_write_error,
        "required_for_pass": [
            "At least one public dataset present locally with source IDs and hash recorded.",
            "A deterministic split protocol documented and rerunnable.",
            "Marginal and CQR-lite conformal coverage audit generated.",
            "For broad generalization: add an event-rich public dataset (for example M5/calendar events or a credentialed MIMIC event panel).",
            "For event-state claims: show that only the schema adapter changes, not the method contract.",
        ],
        "interpretation": "This is real public insurance evidence for zero-inflated heavy-tailed coverage. It is not, by itself, evidence for event-stream or broad cross-industry generalization.",
        "next_action": "Keep this as a public insurance pilot; add an event-rich public dataset before upgrading external validation beyond partial.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
    OUT_MD.write_text("# R14 freMTPL2 External Validation Audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n")
    print(f"OK: {OUT_JSON}")
    print(f"OK: {OUT_MD}")
    print(json.dumps({
        "status": report["status"],
        "public_dataset_acquired": report["scope_status"]["public_dataset_acquired"],
        "full_scope_run": full_scope,
        "public_insurance_heavy_tail_pilot": report["scope_status"]["public_insurance_heavy_tail_pilot"],
        "n_rows": report["dataset"]["n_rows"],
        "cqr_lite_coverage_90": None if validation is None else validation.get("cqr_lite_coverage_90"),
        "quantile_cqr_coverage_90": quantile_cqr,
        "next_action": report["next_action"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())