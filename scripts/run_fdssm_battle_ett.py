#!/usr/bin/env python3
"""Battle: FD-SSM vs DLinear on ETT benchmarks.

This script evaluates the Frequency-Decoupled SSM (FD-SSM) against
DLinear (Zeng et al. 2023) and naive baselines on ETT time-series.

Success criterion: FD-SSM MSE <= DLinear MSE on at least 2/3 pred_lens.

Architecture used:
    FDSSMForecaster — DLinear decomposition floor + gated SSM residual.
    The DLinear component alone guarantees competitive trend extrapolation.
    The SSM captures additional seasonal structure above the DLinear floor.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.narrative.block3.models.single_model_mainline.event_driven_ssm import (
    FDSSMForecaster,
    SeriesDecomp,
)

_LOG = logging.getLogger("fdssm_battle_ett")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

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
        return fpath
    import urllib.request
    url = ETT_URL.format(name=name)
    _LOG.info(f"Downloading {url}")
    urllib.request.urlretrieve(url, fpath)
    return fpath


def load_ett(path: Path) -> np.ndarray:
    """Load ETT CSV → (T, 7) float32."""
    import pandas as pd
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c != "date"]
    return df[cols].to_numpy(dtype=np.float32)


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
        x = self.data[s : s + self.seq_len]
        y = self.data[s + self.seq_len : s + self.seq_len + self.pred_len, -1]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


# ─────────────────────────────────────────────────────────────────────
#  DLinear baseline (reference implementation)
# ─────────────────────────────────────────────────────────────────────

class DLinear(nn.Module):
    """DLinear (Zeng et al. 2023) — decomposition + per-channel linear.

    This is the exact architecture from the paper: moving average
    decomposition into trend and seasonal, then independent linear
    projections from seq_len → pred_len for each channel.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_input: int,
        kernel_size: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_input = d_input
        self.individual = individual

        self.decomp = SeriesDecomp(kernel_size=kernel_size)

        if individual:
            self.trend_linear = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(d_input)
            ])
            self.seasonal_linear = nn.ModuleList([
                nn.Linear(seq_len, pred_len) for _ in range(d_input)
            ])
        else:
            self.trend_linear = nn.Linear(seq_len, pred_len)
            self.seasonal_linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, pred_len) for target channel (last)."""
        x_trend, x_seasonal = self.decomp(x)

        if self.individual:
            trend_parts = []
            seasonal_parts = []
            for i in range(self.d_input):
                trend_parts.append(self.trend_linear[i](x_trend[:, :, i]))
                seasonal_parts.append(self.seasonal_linear[i](x_seasonal[:, :, i]))
            trend_pred = torch.stack(trend_parts, dim=-1)
            seasonal_pred = torch.stack(seasonal_parts, dim=-1)
        else:
            trend_pred = self.trend_linear(
                x_trend.permute(0, 2, 1)
            ).permute(0, 2, 1)
            seasonal_pred = self.seasonal_linear(
                x_seasonal.permute(0, 2, 1)
            ).permute(0, 2, 1)

        return (trend_pred + seasonal_pred)[:, :, -1]


# ─────────────────────────────────────────────────────────────────────
#  Baselines
# ─────────────────────────────────────────────────────────────────────

def naive_forecast(test_ds: ETTWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    """Last-value repeat baseline."""
    preds, trues = [], []
    for i in range(len(test_ds)):
        x, y = test_ds[i]
        preds.append(x[-1, -1].item() * np.ones(len(y)))
        trues.append(y.numpy())
    return np.stack(preds), np.stack(trues)


# ─────────────────────────────────────────────────────────────────────
#  Training harness (shared for DLinear and FD-SSM)
# ─────────────────────────────────────────────────────────────────────

def train_and_eval(
    model: nn.Module,
    train_ds: ETTWindowDataset,
    test_ds: ETTWindowDataset,
    device: torch.device,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    wd: float = 1e-4,
    label: str = "model",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train model and return (preds, trues, train_seconds)."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    t0 = time.time()
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

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / max(n_b, 1)
            _LOG.info(f"  [{label}] Epoch {epoch+1}/{n_epochs}: MSE={avg:.6f}")

    train_time = time.time() - t0

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

    return np.concatenate(all_pred), np.concatenate(all_true), train_time


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FD-SSM vs DLinear Battle on ETT"
    )
    parser.add_argument(
        "--dataset", default="ETTh1",
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
    )
    parser.add_argument("--pred-lens", nargs="+", type=int, default=[96, 192, 336])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--decomp-kernel", type=int, default=25)
    parser.add_argument("--output-dir", default="runs/fdssm_battle_ett")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOG.info(f"Device: {device}")
    _LOG.info(f"Dataset: {args.dataset}, seq_len={args.seq_len}")
    _LOG.info(f"d_model={args.d_model}, decomp_kernel={args.decomp_kernel}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    csv_path = download_ett(args.dataset)
    data = load_ett(csv_path)
    _LOG.info(f"Data shape: {data.shape}")
    T = len(data)

    # Standard 60/20/20 split
    train_end = int(T * 0.6)
    val_end = int(T * 0.8)
    train_data = data[:train_end]
    test_data = data[val_end:]

    # z-score normalize
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0) + 1e-8
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    d_input = data.shape[1]  # 7 for ETT

    all_results = []
    wins = 0
    total = 0
    t_start = time.time()

    for pred_len in args.pred_lens:
        _LOG.info(f"\n{'='*60}")
        _LOG.info(f"  pred_len={pred_len}")
        _LOG.info(f"{'='*60}")

        train_ds = ETTWindowDataset(train_data, args.seq_len, pred_len)
        test_ds = ETTWindowDataset(test_data, args.seq_len, pred_len)

        if len(train_ds) < 10 or len(test_ds) < 10:
            _LOG.warning(f"  Skipping pred_len={pred_len}: insufficient windows")
            continue

        cell = {
            "dataset": args.dataset,
            "pred_len": pred_len,
            "seq_len": args.seq_len,
            "n_train": len(train_ds),
            "n_test": len(test_ds),
        }

        # ── Naive ──
        _LOG.info("  [Naive] Last-value repeat ...")
        naive_pred, naive_true = naive_forecast(test_ds)
        naive_mse = float(np.mean((naive_pred - naive_true) ** 2))
        naive_mae = float(np.mean(np.abs(naive_pred - naive_true)))
        cell["naive"] = {"mse": naive_mse, "mae": naive_mae}
        _LOG.info(f"    MSE={naive_mse:.6f}  MAE={naive_mae:.6f}")

        # ── DLinear ──
        _LOG.info("  [DLinear] Training ...")
        torch.manual_seed(args.seed)
        dlinear = DLinear(
            seq_len=args.seq_len,
            pred_len=pred_len,
            d_input=d_input,
            kernel_size=args.decomp_kernel,
            individual=False,
        )
        dl_pred, dl_true, dl_time = train_and_eval(
            dlinear, train_ds, test_ds, device,
            n_epochs=args.n_epochs, label="DLinear",
        )
        dl_mse = float(np.mean((dl_pred - dl_true) ** 2))
        dl_mae = float(np.mean(np.abs(dl_pred - dl_true)))
        cell["dlinear"] = {"mse": dl_mse, "mae": dl_mae, "train_seconds": dl_time}
        _LOG.info(f"    MSE={dl_mse:.6f}  MAE={dl_mae:.6f}  ({dl_time:.1f}s)")

        # ── FD-SSM ──
        _LOG.info("  [FD-SSM] Training ...")
        torch.manual_seed(args.seed)
        fdssm = FDSSMForecaster(
            d_input=d_input,
            seq_len=args.seq_len,
            pred_len=pred_len,
            d_model=args.d_model,
            d_state=16,
            n_layers=2,
            decomp_kernel=args.decomp_kernel,
            individual=False,
        )
        fd_pred, fd_true, fd_time = train_and_eval(
            fdssm, train_ds, test_ds, device,
            n_epochs=args.n_epochs, label="FD-SSM",
            wd=1e-3,
        )
        fd_mse = float(np.mean((fd_pred - fd_true) ** 2))
        fd_mae = float(np.mean(np.abs(fd_pred - fd_true)))
        cell["fdssm"] = {"mse": fd_mse, "mae": fd_mae, "train_seconds": fd_time}
        _LOG.info(f"    MSE={fd_mse:.6f}  MAE={fd_mae:.6f}  ({fd_time:.1f}s)")

        # ── Verdict ──
        beats_dlinear = fd_mse <= dl_mse
        beats_naive = fd_mse < naive_mse
        delta_pct = (fd_mse - dl_mse) / dl_mse * 100
        cell["verdict"] = {
            "beats_dlinear": beats_dlinear,
            "beats_naive": beats_naive,
            "fdssm_vs_dlinear_pct": round(delta_pct, 2),
            "PASS": beats_dlinear,
        }

        status = "PASS" if beats_dlinear else "FAIL"
        _LOG.info(
            f"  >>> {status}: FD-SSM vs DLinear = {delta_pct:+.2f}%"
        )

        if beats_dlinear:
            wins += 1
        total += 1

        all_results.append(cell)

    elapsed = time.time() - t_start

    # Summary
    summary = {
        "dataset": args.dataset,
        "wins": wins,
        "total": total,
        "overall_pass": wins >= max(1, total * 2 // 3),
        "elapsed_seconds": round(elapsed, 1),
        "cells": all_results,
    }

    out_path = output_dir / f"fdssm_battle_{args.dataset}_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    _LOG.info(f"\nResults saved to {out_path}")
    _LOG.info(f"Overall: {wins}/{total} wins — {'PASS' if summary['overall_pass'] else 'FAIL'}")

    print(json.dumps(summary, indent=2))
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
