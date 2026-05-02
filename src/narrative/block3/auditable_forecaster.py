"""Post-hoc auditable conformal layer for arbitrary forecasters.

This module is intentionally small: it does not train a forecasting model.
It consumes calibration predictions from any forecaster, fits a conformal
interval rule, and emits prediction/audit tables with stable row keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .models.calibration import NCCoPoCalibrator, NCCoPoConfig


@dataclass(frozen=True)
class AuditableProtocolConfig:
    alpha: float = 0.10
    use_studentized: bool = False
    row_key_name: str = "row_key"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditableForecastResult:
    frame: pd.DataFrame
    summary: dict[str, Any]


def _as_1d(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _row_keys(row_keys: Iterable[Any] | None, n_rows: int, name: str) -> list[Any]:
    if row_keys is None:
        return list(range(n_rows))
    keys = list(row_keys)
    if len(keys) != n_rows:
        raise ValueError(f"{name} length {len(keys)} != {n_rows}")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} must be unique for strict row-key audit")
    return keys


class AuditableConformalProtocolLayer:
    """Conformal/audit layer that wraps predictions from any forecaster.

    Typical use:
      1. Train any SOTA forecaster however it normally trains.
      2. Produce calibration predictions and test predictions.
      3. Fit this layer on calibration `(y, y_pred)`.
      4. Emit test predictions with intervals and row-key audit fields.
    """

    def __init__(self, config: AuditableProtocolConfig | None = None) -> None:
        self.config = config or AuditableProtocolConfig()
        self._calibrator: NCCoPoCalibrator | None = None
        self._fit_summary: dict[str, Any] = {}

    def fit(
        self,
        y_cal: Sequence[float] | np.ndarray,
        y_pred_cal: Sequence[float] | np.ndarray,
        *,
        row_keys_cal: Iterable[Any] | None = None,
        sigma_hat_cal: Sequence[float] | np.ndarray | None = None,
        groups_cal: Sequence[int] | np.ndarray | None = None,
    ) -> "AuditableConformalProtocolLayer":
        y = _as_1d(y_cal, "y_cal")
        pred = _as_1d(y_pred_cal, "y_pred_cal")
        if y.shape != pred.shape:
            raise ValueError("y_cal and y_pred_cal must have the same shape")
        keys = _row_keys(row_keys_cal, len(y), "row_keys_cal")
        sigma = None if sigma_hat_cal is None else _as_1d(sigma_hat_cal, "sigma_hat_cal")
        if sigma is not None and sigma.shape != y.shape:
            raise ValueError("sigma_hat_cal must match y_cal shape")
        groups = None if groups_cal is None else np.asarray(groups_cal).reshape(-1)
        if groups is not None and groups.shape[0] != y.shape[0]:
            raise ValueError("groups_cal must match y_cal length")

        self._calibrator = NCCoPoCalibrator(
            NCCoPoConfig(
                alpha=float(self.config.alpha),
                use_studentized=bool(self.config.use_studentized),
                mondrian_groups=groups,
            )
        ).fit(y, pred, sigma_hat_cal=sigma)
        result = self._calibrator.result
        self._fit_summary = {
            "status": "fitted",
            "alpha": float(self.config.alpha),
            "n_calibration_rows": int(len(y)),
            "unique_row_keys": int(len(set(keys))),
            "use_studentized": bool(self.config.use_studentized),
            "cal_coverage": None if result is None else float(result.cal_coverage),
            "conformal_q": None if result is None else float(result.conformal_q),
            "metadata": dict(self.config.metadata),
        }
        return self

    def predict(
        self,
        y_pred: Sequence[float] | np.ndarray,
        *,
        row_keys: Iterable[Any] | None = None,
        sigma_hat: Sequence[float] | np.ndarray | None = None,
        groups: Sequence[int] | np.ndarray | None = None,
    ) -> AuditableForecastResult:
        if self._calibrator is None:
            raise RuntimeError("AuditableConformalProtocolLayer must be fitted first")
        pred = _as_1d(y_pred, "y_pred")
        keys = _row_keys(row_keys, len(pred), "row_keys")
        sigma = None if sigma_hat is None else _as_1d(sigma_hat, "sigma_hat")
        if sigma is not None and sigma.shape != pred.shape:
            raise ValueError("sigma_hat must match y_pred shape")
        group_array = None if groups is None else np.asarray(groups).reshape(-1)
        if group_array is not None and group_array.shape[0] != pred.shape[0]:
            raise ValueError("groups must match y_pred length")

        lower, upper = self._calibrator.predict_interval(pred, sigma_hat=sigma, groups=group_array)
        frame = pd.DataFrame(
            {
                self.config.row_key_name: keys,
                "y_pred": pred,
                "interval_lower": lower,
                "interval_upper": upper,
                "interval_width": upper - lower,
            }
        )
        summary = {
            **self._fit_summary,
            "n_prediction_rows": int(len(frame)),
            "prediction_row_keys_unique": bool(frame[self.config.row_key_name].is_unique),
            "interval_width_mean": float(frame["interval_width"].mean()),
        }
        return AuditableForecastResult(frame=frame, summary=summary)

    def score(self, y_true: Sequence[float] | np.ndarray, result: AuditableForecastResult) -> dict[str, Any]:
        y = _as_1d(y_true, "y_true")
        if len(y) != len(result.frame):
            raise ValueError("y_true length must match prediction frame length")
        pred = result.frame["y_pred"].to_numpy(dtype=np.float64)
        lower = result.frame["interval_lower"].to_numpy(dtype=np.float64)
        upper = result.frame["interval_upper"].to_numpy(dtype=np.float64)
        return {
            "n": int(len(y)),
            "mae": float(np.mean(np.abs(y - pred))),
            "coverage": float(np.mean((y >= lower) & (y <= upper))),
            "interval_width_mean": float(np.mean(upper - lower)),
            "row_keys_unique": bool(result.frame[self.config.row_key_name].is_unique),
        }


def strict_row_key_delta(
    left: AuditableForecastResult,
    right: AuditableForecastResult,
    *,
    row_key_name: str = "row_key",
) -> dict[str, Any]:
    """Compare two protocol outputs only on exact shared row keys."""
    left_frame = left.frame[[row_key_name, "y_pred"]].rename(columns={"y_pred": "left_pred"})
    right_frame = right.frame[[row_key_name, "y_pred"]].rename(columns={"y_pred": "right_pred"})
    merged = left_frame.merge(right_frame, on=row_key_name, how="inner", validate="one_to_one")
    delta = merged["right_pred"].to_numpy(dtype=np.float64) - merged["left_pred"].to_numpy(dtype=np.float64)
    return {
        "n_left": int(len(left_frame)),
        "n_right": int(len(right_frame)),
        "n_overlap": int(len(merged)),
        "strict_overlap_rate_left": float(len(merged) / max(len(left_frame), 1)),
        "prediction_delta_mean": float(delta.mean()) if len(delta) else None,
        "prediction_delta_abs_mean": float(np.abs(delta).mean()) if len(delta) else None,
    }


__all__ = [
    "AuditableProtocolConfig",
    "AuditableForecastResult",
    "AuditableConformalProtocolLayer",
    "strict_row_key_delta",
]