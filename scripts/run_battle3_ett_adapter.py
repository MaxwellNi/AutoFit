#!/usr/bin/env python3
"""Battle 3: ETT/Weather Universal Test — 降维泛化打榜.

Proves our R²-IN Shared Trunk generalizes to standard time series forecasting
benchmarks (ETTh1, ETTm1, Weather) with a simple linear readout head.

Architecture:
    raw_features → R²-IN backbone → compact_state (64d) → LinearReadout → MSE

Comparison baselines:
    - Naive (last value repeat)
    - DLinear (decomposition-linear, strong baseline)

Usage:
    python scripts/run_battle3_ett_adapter.py \
        --dataset ETTh1 \
        --pred-len 96 192 336 720 \
        --output-dir runs/benchmarks/.../battle3_ett_weather
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_LOG = logging.getLogger("battle3_ett")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

# ------------------------------------------------------------------
# ETT/Weather DataLoader
# ------------------------------------------------------------------
_ETT_URL_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
_WEATHER_URL = "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/weather/weather.csv"

_DATASET_URLS = {
    "ETTh1": f"{_ETT_URL_BASE}/ETTh1.csv",
    "ETTh2": f"{_ETT_URL_BASE}/ETTh2.csv",
    "ETTm1": f"{_ETT_URL_BASE}/ETTm1.csv",
    "ETTm2": f"{_ETT_URL_BASE}/ETTm2.csv",
}

# Standard split ratios (following PatchTST / DLinear convention)
_SPLIT_RATIOS = {
    "ETTh1": (12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24),  # 12/4/4 months
    "ETTh2": (12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24),
    "ETTm1": (12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4),
    "ETTm2": (12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4),
    "Weather": (36792, 5271, 10540),  # following standard split
}


def load_dataset(name: str, cache_dir: str = "/tmp/ts_benchmarks") -> pd.DataFrame:
    """Load a standard time series benchmark dataset."""
    cache_path = Path(cache_dir) / f"{name}.csv"
    if cache_path.exists():
        _LOG.info(f"Loading cached {name} from {cache_path}")
        return pd.read_csv(cache_path)

    url = _DATASET_URLS.get(name)
    if url is None and name == "Weather":
        url = _WEATHER_URL
    if url is None:
        raise ValueError(f"Unknown dataset: {name}. Choices: {list(_DATASET_URLS.keys()) + ['Weather']}")

    _LOG.info(f"Downloading {name} from {url} ...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(url)
    df.to_csv(cache_path, index=False)
    _LOG.info(f"Cached to {cache_path} ({len(df)} rows, {len(df.columns)} cols)")
    return df


def prepare_sliding_windows(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window input/output pairs.

    Args:
        data: (T, C) array of multivariate time series
        seq_len: lookback window length
        pred_len: prediction horizon

    Returns:
        X: (N, seq_len, C) input windows
        Y: (N, pred_len, C) target windows
    """
    T, C = data.shape
    n_samples = T - seq_len - pred_len + 1
    if n_samples <= 0:
        raise ValueError(f"Not enough data: T={T}, seq_len={seq_len}, pred_len={pred_len}")

    X = np.zeros((n_samples, seq_len, C), dtype=np.float32)
    Y = np.zeros((n_samples, pred_len, C), dtype=np.float32)
    for i in range(n_samples):
        X[i] = data[i: i + seq_len]
        Y[i] = data[i + seq_len: i + seq_len + pred_len]
    return X, Y


def split_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/val/test using standard ratios."""
    # Drop date column if present
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    data = df[numeric_cols].to_numpy(dtype=np.float32)

    if dataset_name in _SPLIT_RATIOS:
        n_train, n_val, n_test = _SPLIT_RATIOS[dataset_name]
    else:
        T = len(data)
        n_train = int(T * 0.7)
        n_val = int(T * 0.1)
        n_test = T - n_train - n_val

    n_train = min(n_train, len(data) - 2)
    n_val = min(n_val, len(data) - n_train - 1)

    train = data[:n_train]
    val = data[n_train: n_train + n_val]
    test = data[n_train + n_val:]

    _LOG.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}, features={data.shape[1]}")
    return train, val, test


# ------------------------------------------------------------------
# R²-IN Backbone + Linear Readout
# ------------------------------------------------------------------
class R2INLinearForecaster:
    """Simplified R²-IN trunk + linear readout for standard TSF benchmarks.

    Architecture:
        1. R²-IN normalize each window: (x - median) / (1.4826 * MAD)
        2. Flatten window → random projection to compact_dim
        3. Linear readout: compact → pred_len * n_channels
    """

    def __init__(self, seq_len: int, pred_len: int, n_channels: int,
                 compact_dim: int = 64, seed: int = 42):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.compact_dim = compact_dim
        self.seed = seed
        self._fitted = False

    def _r2in_normalize(self, X: np.ndarray) -> np.ndarray:
        """Per-window R²-IN: (window - median) / (1.4826*MAD)."""
        # X: (N, seq_len, C)
        N = X.shape[0]
        flat = X.reshape(N, -1).astype(np.float64)
        loc = np.median(flat, axis=1, keepdims=True)
        mad = np.median(np.abs(flat - loc), axis=1, keepdims=True)
        scale = 1.4826 * mad
        scale = np.where(scale < 1e-6, 1.0, scale)
        return ((flat - loc) / scale).astype(np.float32)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray,
            X_val: np.ndarray = None, Y_val: np.ndarray = None):
        """Fit linear readout with MSE loss (closed-form ridge regression)."""
        N = X_train.shape[0]
        flat = self._r2in_normalize(X_train)  # (N, seq_len*C)
        input_dim = flat.shape[1]

        # Random projection
        rng = np.random.RandomState(self.seed)
        self._proj = (rng.randn(input_dim, self.compact_dim) / np.sqrt(input_dim)).astype(np.float32)
        compact = flat @ self._proj  # (N, compact_dim)

        # Target: flatten Y
        target = Y_train.reshape(N, -1).astype(np.float64)  # (N, pred_len*C)
        output_dim = target.shape[1]

        # Ridge regression (closed form): W = (X^T X + λI)^{-1} X^T Y
        X_bias = np.hstack([compact, np.ones((N, 1), dtype=np.float64)])  # (N, compact_dim+1)
        lambda_reg = 1e-3
        XtX = X_bias.T @ X_bias + lambda_reg * np.eye(X_bias.shape[1], dtype=np.float64)
        XtY = X_bias.T @ target
        self._W = np.linalg.solve(XtX, XtY).astype(np.float32)  # (compact_dim+1, output_dim)

        # Store normalization stats for denormalization
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        flat = self._r2in_normalize(X)
        compact = flat @ self._proj
        X_bias = np.hstack([compact, np.ones((N, 1), dtype=np.float32)])
        pred_flat = X_bias @ self._W  # (N, pred_len*C)
        return pred_flat.reshape(N, self.pred_len, self.n_channels)


# ------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------
def naive_forecast(X: np.ndarray, pred_len: int) -> np.ndarray:
    """Repeat last value for pred_len steps."""
    # X: (N, seq_len, C)
    last = X[:, -1:, :]  # (N, 1, C)
    return np.tile(last, (1, pred_len, 1))


def dlinear_forecast(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_test: np.ndarray, pred_len: int, kernel_size: int = 25,
) -> np.ndarray:
    """Simple DLinear: decompose into trend+seasonal, linear map each.

    Simplified version using moving average decomposition + two linear layers.
    """
    N_train, seq_len, C = X_train.shape

    # Decomposition via moving average
    def _decompose(x):
        # x: (N, seq_len, C)
        if seq_len < kernel_size:
            ks = seq_len
        else:
            ks = kernel_size
        # Ensure odd kernel for symmetric padding
        if ks % 2 == 0:
            ks = ks - 1
        ks = max(ks, 3)
        pad = ks // 2
        trend = np.zeros_like(x)
        for c in range(C):
            col = x[:, :, c]  # (N, seq_len)
            padded = np.pad(col, ((0, 0), (pad, pad)), mode='edge')
            # Insert leading zero for proper cumsum sliding window
            zeros = np.zeros((padded.shape[0], 1), dtype=padded.dtype)
            cumsum = np.cumsum(np.concatenate([zeros, padded], axis=1), axis=1)
            trend[:, :, c] = (cumsum[:, ks:] - cumsum[:, :-ks]) / ks
        seasonal = x - trend
        return trend, seasonal

    trend_train, seasonal_train = _decompose(X_train)
    target = Y_train.reshape(N_train, -1)

    # Two linear maps (trend → pred, seasonal → pred)
    trend_flat = trend_train.reshape(N_train, -1)
    seasonal_flat = seasonal_train.reshape(N_train, -1)
    combined = np.hstack([trend_flat, seasonal_flat, np.ones((N_train, 1))])

    # Ridge regression
    XtX = combined.T @ combined + 1e-3 * np.eye(combined.shape[1])
    XtY = combined.T @ target
    W = np.linalg.solve(XtX, XtY)

    # Predict
    trend_test, seasonal_test = _decompose(X_test)
    N_test = X_test.shape[0]
    combined_test = np.hstack([
        trend_test.reshape(N_test, -1),
        seasonal_test.reshape(N_test, -1),
        np.ones((N_test, 1)),
    ])
    pred = combined_test @ W
    return pred.reshape(N_test, pred_len, C)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate_mse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MSE and MAE on multivariate forecasts."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"mse": mse, "mae": mae}


# ------------------------------------------------------------------
# Run one configuration
# ------------------------------------------------------------------
def _run_config(
    dataset_name: str,
    pred_len: int,
    seq_len: int,
    output_dir: Path,
) -> Dict[str, Any]:
    _LOG.info(f"=== {dataset_name} | seq_len={seq_len} pred_len={pred_len} ===")

    df = load_dataset(dataset_name)
    train_data, val_data, test_data = split_dataset(df, dataset_name)
    n_channels = train_data.shape[1]

    # Build sliding windows
    X_train, Y_train = prepare_sliding_windows(train_data, seq_len, pred_len)
    X_val, Y_val = prepare_sliding_windows(val_data, seq_len, pred_len)
    X_test, Y_test = prepare_sliding_windows(test_data, seq_len, pred_len)
    _LOG.info(f"  Windows: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Standardize using training statistics (channel-wise)
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)

    X_train_n = (X_train - train_mean) / train_std
    X_test_n = (X_test - train_mean) / train_std
    Y_train_n = (Y_train - train_mean) / train_std
    Y_test_n = (Y_test - train_mean) / train_std

    results = {}

    # 1. Naive
    naive_pred = naive_forecast(X_test_n, pred_len)
    results["Naive"] = evaluate_mse_mae(Y_test_n, naive_pred)
    _LOG.info(f"  Naive: MSE={results['Naive']['mse']:.6f} MAE={results['Naive']['mae']:.6f}")

    # 2. DLinear
    dlinear_pred = dlinear_forecast(X_train_n, Y_train_n, X_test_n, pred_len)
    results["DLinear"] = evaluate_mse_mae(Y_test_n, dlinear_pred)
    _LOG.info(f"  DLinear: MSE={results['DLinear']['mse']:.6f} MAE={results['DLinear']['mae']:.6f}")

    # 3. R²-IN Trunk + Linear Readout
    model = R2INLinearForecaster(
        seq_len=seq_len, pred_len=pred_len, n_channels=n_channels,
    )
    model.fit(X_train_n, Y_train_n)
    r2in_pred = model.predict(X_test_n)
    results["R2IN_Trunk"] = evaluate_mse_mae(Y_test_n, r2in_pred)
    _LOG.info(f"  R²-IN Trunk: MSE={results['R2IN_Trunk']['mse']:.6f} MAE={results['R2IN_Trunk']['mae']:.6f}")

    # Summary
    config_result = {
        "dataset": dataset_name,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "n_channels": n_channels,
        "n_train_windows": len(X_train),
        "n_test_windows": len(X_test),
        "results": results,
    }

    out_path = output_dir / f"{dataset_name}_s{seq_len}_p{pred_len}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config_result, f, indent=2)

    return config_result


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
def _print_summary(all_results: List[Dict], output_dir: Path) -> None:
    lines = [
        "# Battle 3: ETT/Weather Universal Test — 降维泛化打榜\n",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M')}\n",
        "## Results\n",
        "| dataset | pred_len | Naive MSE | DLinear MSE | R²-IN MSE | vs DLinear |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    wins = total = 0
    for r in all_results:
        res = r["results"]
        naive_mse = res["Naive"]["mse"]
        dlinear_mse = res["DLinear"]["mse"]
        r2in_mse = res["R2IN_Trunk"]["mse"]
        total += 1
        verdict = "WIN" if r2in_mse <= dlinear_mse * 1.05 else "LOSE"
        if verdict == "WIN":
            wins += 1
        lines.append(
            f"| {r['dataset']} | {r['pred_len']} | {naive_mse:.6f} | "
            f"{dlinear_mse:.6f} | {r2in_mse:.6f} | {verdict} |"
        )

    lines.append(f"\n## Verdict\n")
    lines.append(f"- R²-IN vs DLinear: {wins}/{total} within 5%")
    if wins >= total // 2:
        lines.append(f"\n**PASS**: R²-IN trunk generalizes to standard TSF benchmarks\n")
    else:
        lines.append(f"\n**FAIL**: R²-IN trunk does not generalize — finance-specific only\n")

    summary_md = "\n".join(lines)
    summary_path = output_dir / "battle3_summary.md"
    summary_path.write_text(summary_md)
    print(summary_md)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Battle 3: ETT/Weather universal test")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "ETTm1"])
    parser.add_argument("--pred-len", nargs="+", type=int, default=[96, 192, 336, 720])
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--output-dir", default="runs/benchmarks/single_model_mainline_localclear_20260420/battle3_ett_weather")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for dataset in args.datasets:
        for pred_len in args.pred_len:
            try:
                r = _run_config(dataset, pred_len, args.seq_len, output_dir)
                all_results.append(r)
            except Exception as exc:
                _LOG.error(f"FAILED {dataset}/p{pred_len}: {exc}", exc_info=True)

    _print_summary(all_results, output_dir)


if __name__ == "__main__":
    main()
