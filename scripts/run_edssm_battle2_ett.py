#!/usr/bin/env python3
"""Battle 2: Universal TSF Benchmark — ETT Generalization Test.

Verifies ED-SSM's adaptive degradation on purely continuous data.
ETT has NO discrete events, so the JumpGate must auto-disable (pure SSM).

Comparison:
  1. Naive (last-value repeat)
  2. Linear (sklearn Ridge on flat window)
  3. ED-SSM (pure Continuous Drift, no jump track)

Pass: ED-SSM MSE < Naive AND ED-SSM is competitive with Linear.

Usage:
    python scripts/run_edssm_battle2_ett.py \\
        --dataset ETTh1 \\
        --pred-lens 96 192 336 \\
        --seq-len 96 \\
        --output-dir runs/edssm_battle2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.narrative.block3.models.single_model_mainline.event_driven_ssm import EventDrivenSSM

_LOG = logging.getLogger("battle2_ett")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
ETT_DIR = Path("data/public/ett/raw")


# ─────────────────────────────────────────────────────────────────────
#  Data utilities
# ─────────────────────────────────────────────────────────────────────

def download_ett(name: str = "ETTh1") -> Path:
    """Download ETT CSV if not present."""
    ETT_DIR.mkdir(parents=True, exist_ok=True)
    fpath = ETT_DIR / f"{name}.csv"
    if fpath.exists():
        _LOG.info(f"ETT data exists: {fpath}")
        return fpath
    import urllib.request
    url = ETT_URL.format(name=name)
    _LOG.info(f"Downloading {url} → {fpath}")
    urllib.request.urlretrieve(url, fpath)
    return fpath


def load_ett(path: Path) -> np.ndarray:
    """Load ETT CSV, return (T, 7) float32 array (all features incl. OT)."""
    import pandas as pd
    df = pd.read_csv(path)
    # Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    cols = [c for c in df.columns if c != "date"]
    data = df[cols].to_numpy(dtype=np.float32)
    _LOG.info(f"Loaded {path.name}: shape={data.shape}")
    return data


class ETTWindowDataset(Dataset):
    """Sliding window dataset for ETT time-series."""

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_windows = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return max(0, self.n_windows)

    def __getitem__(self, idx):
        s = idx
        x = self.data[s : s + self.seq_len]           # (seq_len, 7)
        y = self.data[s + self.seq_len : s + self.seq_len + self.pred_len, -1]  # (pred_len,) OT
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


# ─────────────────────────────────────────────────────────────────────
#  ED-SSM forecaster
# ─────────────────────────────────────────────────────────────────────

class SSMForecaster(nn.Module):
    """ED-SSM + prediction head for time-series forecasting."""

    def __init__(self, d_input: int, pred_len: int, d_model: int = 64,
                 d_state: int = 16, n_layers: int = 2):
        super().__init__()
        self.ssm = EventDrivenSSM(
            d_cont=d_input, d_event=0, d_model=d_model,
            d_state=d_state, d_output=d_model, n_layers=n_layers,
        )
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.ssm(x)        # (B, d_model)
        return self.pred_head(emb)  # (B, pred_len)


# ─────────────────────────────────────────────────────────────────────
#  Baselines
# ─────────────────────────────────────────────────────────────────────

def naive_forecast(test_ds: ETTWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    """Last-value repeat baseline."""
    all_pred, all_true = [], []
    for i in range(len(test_ds)):
        x, y = test_ds[i]
        pred = x[-1, -1].item() * np.ones(len(y))  # repeat last OT
        all_pred.append(pred)
        all_true.append(y.numpy())
    return np.stack(all_pred), np.stack(all_true)


def linear_forecast(
    train_ds: ETTWindowDataset,
    test_ds: ETTWindowDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge regression on flattened input windows."""
    from sklearn.linear_model import Ridge

    def _collect(ds, max_n=50000):
        Xs, Ys = [], []
        for i in range(min(len(ds), max_n)):
            x, y = ds[i]
            Xs.append(x.numpy().ravel())
            Ys.append(y.numpy())
        return np.stack(Xs), np.stack(Ys)

    X_tr, Y_tr = _collect(train_ds)
    X_te, Y_te = _collect(test_ds)

    pred_len = Y_tr.shape[1]
    preds = np.zeros_like(Y_te)
    for h in range(pred_len):
        model = Ridge(alpha=1.0)
        model.fit(X_tr, Y_tr[:, h])
        preds[:, h] = model.predict(X_te)

    return preds, Y_te


def edssm_forecast(
    train_ds: ETTWindowDataset,
    test_ds: ETTWindowDataset,
    pred_len: int,
    device: torch.device,
    n_epochs: int = 30,
    batch_size: int = 256,
    d_model: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Train ED-SSM forecaster on train, evaluate on test."""
    d_input = train_ds.data.shape[1]  # 7

    model = SSMForecaster(d_input, pred_len, d_model=d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_b = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            n_b += 1

        if (epoch + 1) % 10 == 0:
            _LOG.info(f"  Epoch {epoch+1}/{n_epochs}: MSE={epoch_loss/max(n_b,1):.6f}")

    # Evaluate
    model.eval()
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0,
    )
    all_pred, all_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            pred = model(x_batch.to(device)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y_batch.numpy())

    return np.concatenate(all_pred), np.concatenate(all_true)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Battle 2: ETT Generalization")
    parser.add_argument("--dataset", default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("--pred-lens", nargs="+", type=int, default=[96, 192, 336])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--output-dir", default="runs/edssm_battle2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download & load data
    csv_path = download_ett(args.dataset)
    data = load_ett(csv_path)
    T = len(data)

    # Standard 60/20/20 split
    train_end = int(T * 0.6)
    val_end = int(T * 0.8)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Normalize using train statistics
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0) + 1e-8
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (test_data - mean) / std

    all_results = []
    t_start = time.time()

    for pred_len in args.pred_lens:
        _LOG.info(f"\n{'='*50}")
        _LOG.info(f"pred_len={pred_len}, seq_len={args.seq_len}")
        _LOG.info(f"{'='*50}")

        train_ds = ETTWindowDataset(train_data, args.seq_len, pred_len)
        test_ds = ETTWindowDataset(test_data, args.seq_len, pred_len)

        if len(train_ds) < 10 or len(test_ds) < 10:
            _LOG.warning(f"  Skipping pred_len={pred_len}: insufficient windows")
            continue

        cell = {"pred_len": pred_len, "seq_len": args.seq_len,
                "n_train": len(train_ds), "n_test": len(test_ds)}

        # Naive
        _LOG.info("  [Naive] Last-value repeat ...")
        naive_pred, naive_true = naive_forecast(test_ds)
        naive_mse = float(np.mean((naive_pred - naive_true) ** 2))
        naive_mae = float(np.mean(np.abs(naive_pred - naive_true)))
        cell["naive"] = {"mse": naive_mse, "mae": naive_mae}
        _LOG.info(f"    Naive: MSE={naive_mse:.6f} MAE={naive_mae:.6f}")

        # Linear
        _LOG.info("  [Linear] Ridge regression ...")
        lin_pred, lin_true = linear_forecast(train_ds, test_ds)
        lin_mse = float(np.mean((lin_pred - lin_true) ** 2))
        lin_mae = float(np.mean(np.abs(lin_pred - lin_true)))
        cell["linear"] = {"mse": lin_mse, "mae": lin_mae}
        _LOG.info(f"    Linear: MSE={lin_mse:.6f} MAE={lin_mae:.6f}")

        # ED-SSM
        _LOG.info("  [ED-SSM] Pure SSM mode ...")
        t0 = time.time()
        ssm_pred, ssm_true = edssm_forecast(
            train_ds, test_ds, pred_len, device,
            n_epochs=args.n_epochs,
        )
        ssm_time = time.time() - t0
        ssm_mse = float(np.mean((ssm_pred - ssm_true) ** 2))
        ssm_mae = float(np.mean(np.abs(ssm_pred - ssm_true)))
        cell["edssm"] = {"mse": ssm_mse, "mae": ssm_mae, "train_seconds": ssm_time}
        _LOG.info(f"    ED-SSM: MSE={ssm_mse:.6f} MAE={ssm_mae:.6f} ({ssm_time:.1f}s)")

        # Verdict
        beats_naive = ssm_mse < naive_mse
        competitive_linear = ssm_mse < lin_mse * 1.2  # within 20% of linear
        cell["verdict"] = {
            "beats_naive": beats_naive,
            "competitive_with_linear": competitive_linear,
            "PASS": beats_naive and competitive_linear,
            "ssm_vs_naive_pct": float((ssm_mse - naive_mse) / naive_mse * 100),
            "ssm_vs_linear_pct": float((ssm_mse - lin_mse) / lin_mse * 100),
        }
        v = "✓ PASS" if cell["verdict"]["PASS"] else "✗ FAIL"
        _LOG.info(f"  → {v}: SSM vs Naive {cell['verdict']['ssm_vs_naive_pct']:+.1f}%, "
                  f"vs Linear {cell['verdict']['ssm_vs_linear_pct']:+.1f}%")

        all_results.append(cell)

    total_time = time.time() - t_start
    n_pass = sum(1 for r in all_results if r.get("verdict", {}).get("PASS"))
    n_fail = len(all_results) - n_pass

    summary = {
        "battle": "2_ett_generalization",
        "dataset": args.dataset,
        "total": len(all_results),
        "pass": n_pass,
        "fail": n_fail,
        "total_seconds": total_time,
        "cells": all_results,
    }

    out_file = output_dir / "battle2_ett_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    _LOG.info(f"\n{'='*60}")
    _LOG.info(f"Battle 2 — Universal TSF Benchmark ({args.dataset})")
    _LOG.info(f"  {n_pass} PASS / {n_fail} FAIL")
    _LOG.info(f"  Total time: {total_time:.1f}s")
    _LOG.info(f"  Results: {out_file}")
    _LOG.info(f"{'='*60}")


if __name__ == "__main__":
    main()
