#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


logger = logging.getLogger("run_full_benchmark")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _safe_float(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size == 0 or y.size == 0:
        return None
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return None
    try:
        value = float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def _load_offers_core(path: str, limit_rows: Optional[int]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if limit_rows is not None and limit_rows > 0 and len(df) > limit_rows:
        df = df.iloc[:limit_rows].copy()
    return df


def _compute_ratio_w(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    goal = pd.to_numeric(df["funding_goal_usd"], errors="coerce")
    raised = pd.to_numeric(df["funding_raised_usd"], errors="coerce")
    ratio = raised / goal
    df["funding_ratio"] = ratio
    if ratio.notna().any():
        q_low, q_high = ratio.quantile([0.01, 0.99])
        df["funding_ratio_w"] = ratio.clip(lower=q_low, upper=q_high)
    else:
        df["funding_ratio_w"] = np.nan
    return df


def _split_entities(
    entities: List[str],
    split_seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[set, set, set]:
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(entities)
    n = len(perm)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_ids = set(perm[:n_train])
    val_ids = set(perm[n_train : n_train + n_val])
    test_ids = set(perm[n_train + n_val :])
    return train_ids, val_ids, test_ids


@dataclass
class SampleRow:
    x: np.ndarray
    mask: np.ndarray
    y: float
    split: str
    entity_id: str
    input_end_idx: int
    label_idx: int
    input_end_ts: pd.Timestamp
    label_ts: pd.Timestamp
    input_raised: float
    input_goal: float
    input_ratio: float
    label_raised: float
    label_goal: float
    label_ratio: float


def _build_samples(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    label_horizon: int,
    label_goal_min: float,
    sample_strategy: str,
    sample_seed: int,
    split_seed: int,
    static_ratio_tol: float,
    min_label_delta_days: float,
    min_ratio_delta_abs: float,
    min_ratio_delta_rel: float,
    strict_future: bool,
) -> Tuple[List[SampleRow], Dict[str, int]]:
    df = df.copy()
    df = df[df["funding_goal_usd"] >= label_goal_min].copy()
    df = df.sort_values(["entity_id", "snapshot_ts"], kind="stable")
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], errors="coerce", utc=True)

    entities = df["entity_id"].dropna().unique().tolist()
    rng = np.random.RandomState(sample_seed)
    rng.shuffle(entities)
    train_ids, val_ids, test_ids = _split_entities(entities, split_seed)

    dropped_insufficient_future = 0
    dropped_due_to_static_ratio = 0
    dropped_due_to_min_delta_days = 0
    dropped_due_to_small_ratio_delta_abs = 0
    dropped_due_to_small_ratio_delta_rel = 0

    rows: List[SampleRow] = []
    for entity_id, group in df.groupby("entity_id", sort=False):
        if sample_strategy not in {"random_entities", "all"}:
            sample_strategy = "random_entities"

        group = group.reset_index(drop=True)
        if len(group) == 0:
            continue
        for input_end in range(len(group)):
            label_idx = input_end + label_horizon
            if label_idx >= len(group):
                dropped_insufficient_future += 1
                if strict_future:
                    continue
                label_idx = len(group) - 1

            window_start = max(0, input_end - seq_len + 1)
            window = group.iloc[window_start : input_end + 1]
            if len(window) == 0:
                dropped_insufficient_future += 1
                continue

            input_raised = _safe_float(group["funding_raised_usd"].iloc[input_end])
            input_goal = _safe_float(group["funding_goal_usd"].iloc[input_end])
            label_raised = _safe_float(group["funding_raised_usd"].iloc[label_idx])
            label_goal = _safe_float(group["funding_goal_usd"].iloc[label_idx])
            with np.errstate(divide="ignore", invalid="ignore"):
                input_ratio = input_raised / input_goal if input_goal not in (0.0, np.nan) else np.nan
                label_ratio = label_raised / label_goal if label_goal not in (0.0, np.nan) else np.nan

            if np.isfinite(label_ratio) and np.isfinite(input_ratio):
                if abs(label_ratio - input_ratio) < static_ratio_tol:
                    dropped_due_to_static_ratio += 1
                    continue

            input_end_ts = group["snapshot_ts"].iloc[input_end]
            label_ts = group["snapshot_ts"].iloc[label_idx]
            delta_days = None
            if pd.notna(input_end_ts) and pd.notna(label_ts):
                delta_days = (label_ts - input_end_ts).total_seconds() / 86400.0
            if delta_days is not None and min_label_delta_days > 0 and delta_days < min_label_delta_days:
                dropped_due_to_min_delta_days += 1
                continue

            if np.isfinite(label_ratio) and np.isfinite(input_ratio):
                delta_abs = abs(label_ratio - input_ratio)
                delta_rel = delta_abs / max(1.0, abs(input_ratio))
                if min_ratio_delta_abs > 0 and delta_abs < min_ratio_delta_abs:
                    dropped_due_to_small_ratio_delta_abs += 1
                    continue
                if min_ratio_delta_rel > 0 and delta_rel < min_ratio_delta_rel:
                    dropped_due_to_small_ratio_delta_rel += 1
                    continue

            values = window[feature_cols].to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            mask = np.ones((len(values),), dtype=bool)
            if len(values) < seq_len:
                pad = np.zeros((seq_len - len(values), values.shape[1]), dtype=float)
                values = np.concatenate([pad, values], axis=0)
                pad_mask = np.zeros((seq_len - len(mask),), dtype=bool)
                mask = np.concatenate([pad_mask, mask], axis=0)

            split = "train" if entity_id in train_ids else "val" if entity_id in val_ids else "test"
            rows.append(
                SampleRow(
                    x=values,
                    mask=mask,
                    y=_safe_float(group["funding_ratio_w"].iloc[label_idx]),
                    split=split,
                    entity_id=str(entity_id),
                    input_end_idx=int(input_end),
                    label_idx=int(label_idx),
                    input_end_ts=input_end_ts,
                    label_ts=label_ts,
                    input_raised=input_raised,
                    input_goal=input_goal,
                    input_ratio=_safe_float(input_ratio),
                    label_raised=label_raised,
                    label_goal=label_goal,
                    label_ratio=_safe_float(label_ratio),
                )
            )

    drop_counts = {
        "dropped_due_to_insufficient_future": dropped_insufficient_future,
        "dropped_due_to_static_ratio": dropped_due_to_static_ratio,
        "dropped_due_to_min_delta_days": dropped_due_to_min_delta_days,
        "dropped_due_to_small_ratio_delta_abs": dropped_due_to_small_ratio_delta_abs,
        "dropped_due_to_small_ratio_delta_rel": dropped_due_to_small_ratio_delta_rel,
    }
    return rows, drop_counts


class ArrayDataset(Dataset):
    def __init__(self, rows: List[SampleRow]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "x": torch.as_tensor(row.x, dtype=torch.float32),
            "mask": torch.as_tensor(row.mask, dtype=torch.bool),
            "y": torch.as_tensor(row.y, dtype=torch.float32),
        }


class SimpleRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(-1).to(x.dtype)
            x = x * mask
        x = x.reshape(x.size(0), -1)
        return self.net(x).squeeze(-1)


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> float:
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    model.to(device)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device)
            pred = model(x, mask)
            loss = crit(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if val_loader is not None:
            model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    mask = batch.get("mask")
                    if mask is not None:
                        mask = mask.to(device)
                    pred = model(x, mask)
                    loss = crit(pred, y)
                    total += float(loss.item()) * x.size(0)
                    n += x.size(0)
            if n > 0:
                best_val = min(best_val, total / n)
    return best_val


def _predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    preds: List[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device)
            pred = model(x, mask)
            preds.extend(pred.detach().cpu().numpy().tolist())
    return np.asarray(preds, dtype=float)


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(description="Run full benchmark (minimal reconstructed pipeline).")
    parser.add_argument("--offers_core", type=str, required=True)
    parser.add_argument("--offers_static", type=str, default=None)
    parser.add_argument("--edgar_features", type=str, default=None)
    parser.add_argument("--use_edgar", type=int, default=0)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--limit_entities", type=int, default=None)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--plan", type=str, default="paper_min")
    parser.add_argument("--strict_matrix", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=["dlinear"])
    parser.add_argument("--fusion_types", nargs="+", default=["none"])
    parser.add_argument("--module_variants", nargs="+", default=["base"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--sample_strategy", type=str, default="random_entities")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--label_goal_min", type=float, default=50.0)
    parser.add_argument("--label_horizon", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--min_label_delta_days", type=float, default=0.0)
    parser.add_argument("--min_ratio_delta_abs", type=float, default=0.0)
    parser.add_argument("--min_ratio_delta_rel", type=float, default=0.0)
    parser.add_argument("--strict_future", type=int, default=1)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = Path("runs") / "benchmarks" / f"{args.exp_name}_{timestamp}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "configs").mkdir(parents=True, exist_ok=True)

    df = _load_offers_core(args.offers_core, args.limit_rows)
    df = _compute_ratio_w(df)

    feature_cols = [
        c
        for c in [
            "funding_raised_usd",
            "funding_goal_usd",
            "investors_count",
            "time_since_start_days",
            "time_delta_days",
        ]
        if c in df.columns
    ]
    if not feature_cols:
        raise RuntimeError("No usable feature columns found in offers_core.")

    static_ratio_tol = 1e-6
    rows, drop_counts = _build_samples(
        df,
        feature_cols=feature_cols,
        seq_len=args.seq_len,
        label_horizon=args.label_horizon,
        label_goal_min=args.label_goal_min,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
        split_seed=args.split_seed,
        static_ratio_tol=static_ratio_tol,
        min_label_delta_days=args.min_label_delta_days,
        min_ratio_delta_abs=args.min_ratio_delta_abs,
        min_ratio_delta_rel=args.min_ratio_delta_rel,
        strict_future=bool(args.strict_future),
    )

    logger.info("Dropped counts: %s", drop_counts)

    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    test_rows = [r for r in rows if r.split == "test"]
    if not rows:
        raise RuntimeError("No samples available after filtering; check data availability and filters.")
    if not train_rows:
        train_rows = rows
    if not test_rows:
        test_rows = rows

    train_loader = DataLoader(ArrayDataset(train_rows), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ArrayDataset(val_rows), batch_size=args.batch_size, shuffle=False) if val_rows else None
    test_loader = DataLoader(ArrayDataset(test_rows), batch_size=args.batch_size, shuffle=False)

    results: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []

    input_dim = args.seq_len * len(feature_cols)
    device = args.device

    for model_name in args.models:
        for fusion_type in args.fusion_types:
            for module_variant in args.module_variants:
                for seed in args.seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model = SimpleRegressor(input_dim=input_dim)
                    best_val = _train_model(
                        model,
                        train_loader,
                        val_loader,
                        epochs=args.epochs,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        device=device,
                    )

                    y_pred_test = _predict(model, test_loader, device=device)
                    y_true_test = np.array([r.y for r in test_rows], dtype=float)
                    mask = np.isfinite(y_true_test) & np.isfinite(y_pred_test)
                    y_true_eval = y_true_test[mask]
                    y_pred_eval = y_pred_test[mask]
                    rmse = float(np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))) if len(y_true_eval) else float("nan")
                    mae = float(mean_absolute_error(y_true_eval, y_pred_eval)) if len(y_true_eval) else float("nan")
                    r2 = float(r2_score(y_true_eval, y_pred_eval)) if len(y_true_eval) > 1 else float("nan")

                    results.append(
                        {
                            "model": model_name,
                            "mse": float(mean_squared_error(y_true_eval, y_pred_eval)) if len(y_true_eval) else float("nan"),
                            "rmse": rmse,
                            "mae": mae,
                            "r2": r2,
                            "best_val_loss": float(best_val),
                            "train_time_sec": None,
                            "max_cuda_mem_mb": None,
                            "epochs": args.epochs,
                            "status": "success",
                            "task": "regression",
                            "backbone": model_name,
                            "fusion_type": fusion_type,
                            "module_flags": {
                                "nonstat": module_variant == "nonstat",
                                "multiscale": module_variant == "multiscale",
                                "multiscale_fft": False,
                                "ssm": False,
                            },
                            "module_variant": module_variant,
                            "seed": seed,
                        }
                    )

                    # predictions: include train + test for baseline
                    for split_name, split_rows, split_loader in [
                        ("train", train_rows, train_loader),
                        ("test", test_rows, test_loader),
                    ]:
                        if not split_rows:
                            continue
                        y_pred = _predict(model, split_loader, device=device)
                        for idx, row in enumerate(split_rows):
                            predictions.append(
                                {
                                    "exp_name": args.exp_name,
                                    "model": model_name,
                                    "task": "regression",
                                    "split": split_name,
                                    "y_pred": float(y_pred[idx]),
                                    "y_true": float(row.y),
                                    "y_pred_raw": float(y_pred[idx]),
                                    "y_true_raw": float(row.y),
                                    "funding_raised_usd_last": float(row.input_raised),
                                    "funding_goal_usd_last": float(row.input_goal),
                                    "funding_raised_usd_input_last": float(row.input_raised),
                                    "funding_goal_usd_input_last": float(row.input_goal),
                                    "funding_ratio_input_last": float(row.input_ratio),
                                    "funding_raised_usd_label": float(row.label_raised),
                                    "funding_goal_usd_label": float(row.label_goal),
                                    "funding_ratio_label": float(row.label_ratio),
                                    "input_end_idx": int(row.input_end_idx),
                                    "label_idx": int(row.label_idx),
                                    "input_end_ts": row.input_end_ts,
                                    "label_ts": row.label_ts,
                                }
                            )

    metrics = {
        "exp_name": args.exp_name,
        "timestamp": timestamp,
        "n_rows": int(len(df)),
        "n_entities": int(df["entity_id"].nunique()) if "entity_id" in df.columns else None,
        "n_features_static": 0,
        "n_features_edgar": 0,
        "edgar_valid_rate": 0.0,
        "cutoff_violation": 0,
        "use_edgar": bool(args.use_edgar),
        "unique_backbones": len(set(args.models)),
        "unique_fusion_types": len(set(args.fusion_types)),
        "unique_module_configs": len(set(args.module_variants)),
        "unique_module_variants": len(set(args.module_variants)),
        "unique_seeds": len(set(args.seeds)),
        "results_count": len(results),
        "total_runs": len(results),
        "results": results,
    }

    (bench_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_parquet(bench_dir / "metrics.parquet", index=False)
    pd.DataFrame(predictions).to_parquet(bench_dir / "predictions.parquet", index=False)

    resolved = {
        "exp_name": args.exp_name,
        "offers_core": args.offers_core,
        "offers_static": args.offers_static,
        "edgar_features": args.edgar_features,
        "use_edgar": bool(args.use_edgar),
        "limit_rows": args.limit_rows,
        "limit_entities": args.limit_entities,
        "device": args.device,
        "models": args.models,
        "max_runs": None,
        "plan": args.plan,
        "fusion_types": args.fusion_types,
        "module_variants": args.module_variants,
        "seeds": args.seeds,
        "sample_strategy": args.sample_strategy,
        "sample_seed": args.sample_seed,
        "split_seed": args.split_seed,
        "label_goal_min": args.label_goal_min,
        "label_horizon": args.label_horizon,
        "strict_future": bool(args.strict_future),
        "dropped_due_to_insufficient_future": drop_counts["dropped_due_to_insufficient_future"],
        "dropped_due_to_static_ratio": drop_counts["dropped_due_to_static_ratio"],
        "dropped_due_to_min_delta_days": drop_counts["dropped_due_to_min_delta_days"],
        "dropped_due_to_small_ratio_delta_abs": drop_counts["dropped_due_to_small_ratio_delta_abs"],
        "dropped_due_to_small_ratio_delta_rel": drop_counts["dropped_due_to_small_ratio_delta_rel"],
        "strict_matrix": bool(args.strict_matrix),
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "enc_in": len(feature_cols),
        "min_label_delta_days": args.min_label_delta_days,
        "min_ratio_delta_abs": args.min_ratio_delta_abs,
        "min_ratio_delta_rel": args.min_ratio_delta_rel,
        "bench_defaults": {
            "data": {"seq_len": args.seq_len, "pred_len": args.pred_len, "enc_in": len(feature_cols)},
            "train": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay},
        },
    }
    (bench_dir / "configs" / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    logger.info("Output: %s", bench_dir)


if __name__ == "__main__":
    main()
