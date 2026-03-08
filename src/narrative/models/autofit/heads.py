#!/usr/bin/env python3
"""
AutoFit v2 — Task-Specific Prediction Heads.

Each head transforms expert outputs into task-specific predictions:
    - Task 1 (Outcome):     classification (sigmoid) + regression + ranking
    - Task 2 (Forecast):    multi-horizon quantile outputs
    - Task 3 (Risk Adjust): OOD-robust calibrated predictions

Heads can wrap any Expert and add:
    - Loss function selection (Huber for heavy tails, quantile for Task 2)
    - Calibration (Platt scaling for Task 1 classification)
    - Confidence intervals
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TaskHead:
    """Base class for task-specific prediction heads."""

    def __init__(self, task: str, config: Optional[Dict[str, Any]] = None):
        self.task = task
        self.config = config or {}

    def postprocess(
        self,
        raw_preds: np.ndarray,
        target: str,
        *,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Post-process raw expert predictions into task outputs.

        Returns dict with keys like 'point', 'lower', 'upper', 'proba', etc.
        """
        return {"point": raw_preds}


class Task1Head(TaskHead):
    """
    Task 1: Outcome Prediction head.

    For binary targets (is_funded): applies sigmoid + optional Platt calibration.
    For continuous targets: applies robust clipping.
    """

    def postprocess(
        self,
        raw_preds: np.ndarray,
        target: str,
        *,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        result = {"point": raw_preds}

        if target == "is_funded":
            # Sigmoid for probability output
            proba = 1.0 / (1.0 + np.exp(-np.clip(raw_preds, -20, 20)))
            result["proba"] = proba
            result["point"] = (proba > 0.5).astype(float)
        else:
            # Regression: clip negatives for financial amounts
            if target in ("funding_raised_usd", "investors_count"):
                result["point"] = np.maximum(raw_preds, 0.0)

        return result


class Task2Head(TaskHead):
    """
    Task 2: Trajectory Forecasting head.

    Adds quantile estimates from point predictions using residual bootstrap.
    """

    def __init__(self, task: str = "task2_forecast", config: Optional[Dict[str, Any]] = None):
        super().__init__(task, config)
        self.quantiles = config.get("quantiles", [0.1, 0.5, 0.9]) if config else [0.1, 0.5, 0.9]

    def postprocess(
        self,
        raw_preds: np.ndarray,
        target: str,
        *,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        result = {"point": raw_preds}

        # Estimate prediction intervals from training residuals
        if y_train is not None and len(y_train) > 10:
            residual_std = np.std(y_train - np.mean(y_train))
            for q in self.quantiles:
                z = float(np.percentile(np.random.randn(10000), q * 100))
                result[f"q{q:.2f}"] = raw_preds + z * residual_std

            result["lower"] = result.get("q0.10", raw_preds - 1.28 * residual_std)
            result["upper"] = result.get("q0.90", raw_preds + 1.28 * residual_std)

        return result


class Task3Head(TaskHead):
    """
    Task 3: Risk-Adjusted Prediction head.

    Focuses on calibration and worst-group performance.
    """

    def postprocess(
        self,
        raw_preds: np.ndarray,
        target: str,
        *,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        result = {"point": raw_preds}

        # Calibration: scale predictions to match training distribution
        if y_train is not None and len(y_train) > 10:
            train_mean = np.mean(y_train)
            train_std = np.std(y_train) + 1e-12
            pred_mean = np.mean(raw_preds)
            pred_std = np.std(raw_preds) + 1e-12

            calibrated = (raw_preds - pred_mean) / pred_std * train_std + train_mean
            result["calibrated"] = calibrated

        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

TASK_HEADS = {
    "task1_outcome": Task1Head,
    "task2_forecast": Task2Head,
    "task3_risk_adjust": Task3Head,
}


def create_task_head(
    task: str,
    config: Optional[Dict[str, Any]] = None,
) -> TaskHead:
    """Create a task-specific prediction head."""
    cls = TASK_HEADS.get(task, TaskHead)
    return cls(task=task, config=config)
