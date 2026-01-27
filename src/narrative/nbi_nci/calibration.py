from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None  # type: ignore


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / float(temperature)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> float:
    if torch is None:
        raise ImportError("torch is required for temperature scaling")
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    temperature = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().item())


def fit_isotonic(probs: np.ndarray, labels: np.ndarray):
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception as e:
        raise ImportError("scikit-learn is required for isotonic regression") from e
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, labels)
    return iso


def apply_isotonic(model, probs: np.ndarray) -> np.ndarray:
    return model.predict(probs)


def reliability_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if probs.ndim != 1:
        probs = probs.reshape(-1)
    if labels.ndim != 1:
        labels = labels.reshape(-1)

    preds = (probs >= 0.5).astype(int)
    acc = float((preds == labels).mean())
    brier = float(np.mean((probs - labels) ** 2))
    nll = float(-(labels * np.log(probs + 1e-8) + (1 - labels) * np.log(1 - probs + 1e-8)).mean())

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc_bin = labels[mask].mean()
        ece += (mask.mean()) * abs(conf - acc_bin)

    return {
        "accuracy": acc,
        "brier": brier,
        "nll": nll,
        "ece": float(ece),
    }


def export_reliability_metrics(metrics: Dict[str, float], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out_path


__all__ = [
    "temperature_scale",
    "fit_temperature",
    "fit_isotonic",
    "apply_isotonic",
    "reliability_metrics",
    "export_reliability_metrics",
]
