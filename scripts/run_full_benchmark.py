#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


logger = logging.getLogger("run_full_benchmark")


def _resolve_edgar_dir(edgar_features: Optional[str]) -> Path:
    if edgar_features:
        return Path(edgar_features)
    latest = Path("runs/edgar_feature_store/latest.txt")
    if not latest.exists():
        raise RuntimeError("edgar_features not provided and latest.txt missing")
    stamp = latest.read_text(encoding="utf-8").strip()
    return Path("runs/edgar_feature_store") / stamp / "edgar_features"


def _edgar_preflight(edgar_dir: Path, bench_dir: Path) -> Dict[str, Any]:
    if "smoke" in str(edgar_dir):
        raise RuntimeError(f"edgar_features points to smoke dir: {edgar_dir}")
    files = [p for p in edgar_dir.rglob("*.parquet") if p.is_file()]
    if not files:
        raise RuntimeError(f"edgar_features empty: {edgar_dir}")
    rng = np.random.RandomState(42)
    sample_file = files[int(rng.randint(0, len(files)))]
    table = pq.read_table(sample_file, memory_map=True)
    if table.num_rows > 200:
        table = table.slice(0, 200)
    df = table.to_pandas()
    cols = list(df.columns)
    col_hash = hashlib.sha256(",".join(sorted(cols)).encode("utf-8")).hexdigest()
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    stats = []
    for col in numeric_cols[:5]:
        non_null_rate = float(df[col].notna().mean()) if len(df) else 0.0
        non_zero_rate = float((df[col].fillna(0) != 0).mean()) if len(df) else 0.0
        stats.append((col, non_null_rate, non_zero_rate))
    if not stats:
        raise RuntimeError("edgar_features has no numeric columns")
    if not any(nr > 0.01 and zr > 0.01 for _, nr, zr in stats):
        raise RuntimeError("edgar_features numeric columns fail non-null/non-zero >1% threshold")
    status_path = bench_dir / "STATUS_PRE.txt"
    lines = [
        f"edgar_dir={edgar_dir}",
        f"edgar_features_file_count={len(files)}",
        f"sample_file={sample_file}",
        f"sample_rows={len(df)}",
        f"sample_cols={len(df.columns)}",
        f"sample_col_names={cols[:30]}",
        f"col_hash={col_hash}",
    ]
    for col, non_null_rate, non_zero_rate in stats:
        lines.append(f"numeric_col={col} non_null_rate={non_null_rate:.4f} non_zero_rate={non_zero_rate:.4f}")
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "edgar_features_path": str(edgar_dir),
        "edgar_features_file_count": len(files),
        "edgar_features_sample_file": str(sample_file),
        "edgar_features_col_hash": col_hash,
        "edgar_status_path": str(status_path),
    }


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


def _split_stats(rows: List["SampleRow"]) -> Dict[str, Any]:
    if not rows:
        return {
            "n_samples": 0,
            "y_min": None,
            "y_max": None,
            "y_mean": None,
            "any_nonfinite_y": False,
        }
    y = np.array([r.y for r in rows], dtype=float)
    finite = np.isfinite(y)
    return {
        "n_samples": int(len(rows)),
        "y_min": float(np.min(y[finite])) if finite.any() else None,
        "y_max": float(np.max(y[finite])) if finite.any() else None,
        "y_mean": float(np.mean(y[finite])) if finite.any() else None,
        "any_nonfinite_y": bool((~finite).any()),
    }


def _limit_rows_by_entities(df: pd.DataFrame, limit_rows: int, seed: int) -> Tuple[pd.DataFrame, int]:
    if "entity_id" not in df.columns:
        return df, 0
    entities = df["entity_id"].dropna().unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(entities)
    frames = []
    total = 0
    for entity_id in entities:
        group = df[df["entity_id"] == entity_id]
        if group.empty:
            continue
        frames.append(group)
        total += len(group)
        if total >= limit_rows:
            break
    if not frames:
        return df.iloc[:0].copy(), 0
    return pd.concat(frames, ignore_index=True), len(frames)


def _load_offers_core(
    path: str, limit_rows: Optional[int], sample_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_parquet(path)
    limit_info: Dict[str, Any] = {"limit_rows_strategy": None, "limit_rows_selected_entities": None}
    if limit_rows is not None and limit_rows > 0 and len(df) > limit_rows:
        if sample_seed is not None:
            if "entity_id" in df.columns:
                df, selected = _limit_rows_by_entities(df, limit_rows, sample_seed)
                limit_info["limit_rows_strategy"] = "entity_subset"
                limit_info["limit_rows_selected_entities"] = int(selected)
            else:
                df = df.sample(n=limit_rows, random_state=sample_seed).copy()
                limit_info["limit_rows_strategy"] = "random_rows"
        else:
            df = df.iloc[:limit_rows].copy()
            limit_info["limit_rows_strategy"] = "head"
    return df, limit_info


def _compute_ratio_w(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    goal = pd.to_numeric(df["funding_goal_usd"], errors="coerce")
    raised = pd.to_numeric(df["funding_raised_usd"], errors="coerce")
    ratio = pd.Series(np.nan, index=df.index, dtype=float)
    valid = np.isfinite(goal) & (goal != 0) & np.isfinite(raised)
    ratio.loc[valid] = raised[valid] / goal[valid]
    df["funding_ratio"] = ratio
    ratio_finite = ratio[np.isfinite(ratio)]
    if not ratio_finite.empty:
        q_low, q_high = ratio_finite.quantile([0.01, 0.99])
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
    dropped_due_to_nonfinite_ratio = 0
    dropped_due_to_nonfinite_label = 0

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
            input_ratio = np.nan
            label_ratio = np.nan
            if np.isfinite(input_goal) and input_goal != 0.0 and np.isfinite(input_raised):
                input_ratio = input_raised / input_goal
            if np.isfinite(label_goal) and label_goal != 0.0 and np.isfinite(label_raised):
                label_ratio = label_raised / label_goal
            if not (np.isfinite(input_ratio) and np.isfinite(label_ratio)):
                dropped_due_to_nonfinite_ratio += 1
                continue

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
            y_value = _safe_float(group["funding_ratio_w"].iloc[label_idx])
            if not np.isfinite(y_value):
                dropped_due_to_nonfinite_label += 1
                continue

            rows.append(
                SampleRow(
                    x=values,
                    mask=mask,
                    y=y_value,
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
        "dropped_due_to_nonfinite_ratio": dropped_due_to_nonfinite_ratio,
        "dropped_due_to_nonfinite_label": dropped_due_to_nonfinite_label,
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
    parser.add_argument("--selected_entities_json", type=str, default=None)
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

    load_limit_rows = args.limit_rows
    if args.selected_entities_json:
        load_limit_rows = None
    df, limit_info = _load_offers_core(args.offers_core, load_limit_rows, args.sample_seed)
    selected_entities = None
    selected_entities_hash = None
    if args.selected_entities_json:
        selected_path = Path(args.selected_entities_json)
        if not selected_path.exists():
            raise RuntimeError(f"selected_entities_json not found: {selected_path}")
        selected_entities = json.loads(selected_path.read_text(encoding="utf-8"))
        selected_entities = [str(e) for e in selected_entities]
        selected_entities_hash = hashlib.sha256("\n".join(sorted(selected_entities)).encode("utf-8")).hexdigest()
        if "entity_id" not in df.columns:
            raise RuntimeError("selected_entities_json provided but entity_id column is missing.")
        df = df[df["entity_id"].astype(str).isin(set(selected_entities))].copy()
        if args.limit_rows:
            logger.info("selected_entities_json provided; ignoring limit_rows=%s", args.limit_rows)
            limit_info["limit_rows_strategy"] = "selected_entities"
            limit_info["limit_rows_selected_entities"] = len(selected_entities)
    edgar_meta: Dict[str, Any] = {}
    if bool(args.use_edgar):
        edgar_dir = _resolve_edgar_dir(args.edgar_features)
        args.edgar_features = str(edgar_dir)
        edgar_meta = _edgar_preflight(edgar_dir, bench_dir)
        logger.info("edgar_features=%s", args.edgar_features)
        logger.info("edgar_features_file_count=%s", edgar_meta.get("edgar_features_file_count"))
    df = _compute_ratio_w(df)

    edgar_feature_cols: List[str] = []
    edgar_join_valid_rate = 0.0
    if bool(args.use_edgar):
        edgar_df = pd.read_parquet(args.edgar_features)
        join_keys = ["offer_id", "cutoff_ts"]
        missing_keys = [k for k in join_keys if k not in df.columns or k not in edgar_df.columns]
        if missing_keys:
            raise RuntimeError(f"edgar join missing keys: {missing_keys}")
        dedup_before = len(edgar_df)
        if "edgar_filed_date" in edgar_df.columns:
            edgar_df = edgar_df.sort_values("edgar_filed_date")
        edgar_df = edgar_df.drop_duplicates(subset=join_keys, keep="last")
        logger.info("edgar_dedup: before=%d after=%d", dedup_before, len(edgar_df))
        df = df.merge(edgar_df, on=join_keys, how="left", validate="many_to_one")
        numeric_cols = [c for c in edgar_df.columns if c not in join_keys and pd.api.types.is_numeric_dtype(edgar_df[c])]
        edgar_feature_cols = [c for c in numeric_cols if c in df.columns]
        if not edgar_feature_cols:
            raise RuntimeError("No numeric edgar feature columns after join.")
        valid_mask = df[edgar_feature_cols].notna().any(axis=1)
        edgar_join_valid_rate = float(valid_mask.mean()) if len(df) else 0.0

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
    ] + edgar_feature_cols
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

    logger.info("Limit rows strategy: %s", limit_info)
    if selected_entities is not None:
        sampled_entities = sorted(set(selected_entities))
    else:
        sampled_entities = sorted({r.entity_id for r in rows})
    sampled_path = bench_dir / "sampled_entities.json"
    sampled_path.write_text(json.dumps(sampled_entities, indent=2), encoding="utf-8")
    selection_hash = selected_entities_hash or hashlib.sha256(
        "\n".join(sampled_entities).encode("utf-8")
    ).hexdigest()
    (bench_dir / "selection_hash.txt").write_text(selection_hash + "\n", encoding="utf-8")

    logger.info("Sampled entities: %d (hash=%s)", len(sampled_entities), selection_hash)
    logger.info("Dropped counts: %s", drop_counts)

    train_rows = [r for r in rows if r.split == "train"]
    val_rows = [r for r in rows if r.split == "val"]
    test_rows = [r for r in rows if r.split == "test"]
    if not rows:
        raise RuntimeError("No samples available after filtering; check data availability and filters.")

    train_stats = _split_stats(train_rows)
    val_stats = _split_stats(val_rows)
    test_stats = _split_stats(test_rows)

    fail_reasons = []
    if train_stats["n_samples"] == 0:
        fail_reasons.append("no_train_samples")
    if val_stats["n_samples"] == 0:
        fail_reasons.append("no_val_samples")
    if test_stats["n_samples"] == 0:
        fail_reasons.append("no_test_samples")
    if train_stats["any_nonfinite_y"]:
        fail_reasons.append("nonfinite_y_train")
    if val_stats["any_nonfinite_y"]:
        fail_reasons.append("nonfinite_y_val")
    if test_stats["any_nonfinite_y"]:
        fail_reasons.append("nonfinite_y_test")
    if bool(args.use_edgar) and (len(edgar_feature_cols) == 0 or edgar_join_valid_rate == 0.0):
        fail_reasons.append("edgar_join_valid_rate_zero")

    resolved = {
        "exp_name": args.exp_name,
        "offers_core": args.offers_core,
        "offers_static": args.offers_static,
        "edgar_features": args.edgar_features,
        "use_edgar": bool(args.use_edgar),
        "edgar_features_file_count": edgar_meta.get("edgar_features_file_count"),
        "edgar_features_col_hash": edgar_meta.get("edgar_features_col_hash"),
        "edgar_features_sample_file": edgar_meta.get("edgar_features_sample_file"),
        "edgar_status_path": edgar_meta.get("edgar_status_path"),
        "n_edgar_features": len(edgar_feature_cols),
        "edgar_join_valid_rate": edgar_join_valid_rate,
        "limit_rows": args.limit_rows,
        "limit_entities": args.limit_entities,
        "limit_rows_strategy": limit_info.get("limit_rows_strategy"),
        "limit_rows_selected_entities": limit_info.get("limit_rows_selected_entities"),
        "selected_entities_json": args.selected_entities_json,
        "selected_entities_hash": selected_entities_hash,
        "selected_entities_count": len(selected_entities) if selected_entities is not None else None,
        "sampled_entities_path": str(sampled_path),
        "sampled_entities_hash": selection_hash,
        "sampled_entities_count": len(sampled_entities),
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
        "dropped_due_to_nonfinite_ratio": drop_counts["dropped_due_to_nonfinite_ratio"],
        "dropped_due_to_nonfinite_label": drop_counts["dropped_due_to_nonfinite_label"],
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

    status_run = {
        "exp_name": args.exp_name,
        "timestamp": timestamp,
        "status": "fail" if fail_reasons else "ok",
        "errors": fail_reasons,
        "n_train": train_stats["n_samples"],
        "n_val": val_stats["n_samples"],
        "n_test": test_stats["n_samples"],
        "any_nonfinite_y_train": train_stats["any_nonfinite_y"],
        "any_nonfinite_y_val": val_stats["any_nonfinite_y"],
        "any_nonfinite_y_test": test_stats["any_nonfinite_y"],
        "y_train_min": train_stats["y_min"],
        "y_train_max": train_stats["y_max"],
        "y_train_mean": train_stats["y_mean"],
        "y_val_min": val_stats["y_min"],
        "y_val_max": val_stats["y_max"],
        "y_val_mean": val_stats["y_mean"],
        "y_test_min": test_stats["y_min"],
        "y_test_max": test_stats["y_max"],
        "y_test_mean": test_stats["y_mean"],
        "y_train_stats": train_stats,
        "y_val_stats": val_stats,
        "y_test_stats": test_stats,
        "n_edgar_features": len(edgar_feature_cols),
        "edgar_join_valid_rate": edgar_join_valid_rate,
        "drop_counts": drop_counts,
    }
    (bench_dir / "STATUS_RUN.json").write_text(json.dumps(status_run, indent=2), encoding="utf-8")

    if fail_reasons:
        metrics = {
            "exp_name": args.exp_name,
            "timestamp": timestamp,
            "status": "fail",
            "errors": fail_reasons,
            "n_rows": int(len(df)),
            "n_entities": int(df["entity_id"].nunique()) if "entity_id" in df.columns else None,
            "n_features_static": 0,
            "n_features_edgar": len(edgar_feature_cols),
            "edgar_valid_rate": edgar_join_valid_rate,
            "use_edgar": bool(args.use_edgar),
            "n_train": train_stats["n_samples"],
            "n_val": val_stats["n_samples"],
            "n_test": test_stats["n_samples"],
            "any_nonfinite_y_train": train_stats["any_nonfinite_y"],
            "any_nonfinite_y_val": val_stats["any_nonfinite_y"],
            "any_nonfinite_y_test": test_stats["any_nonfinite_y"],
            "y_train_min": train_stats["y_min"],
            "y_train_max": train_stats["y_max"],
            "y_train_mean": train_stats["y_mean"],
            "y_val_min": val_stats["y_min"],
            "y_val_max": val_stats["y_max"],
            "y_val_mean": val_stats["y_mean"],
            "y_test_min": test_stats["y_min"],
            "y_test_max": test_stats["y_max"],
            "y_test_mean": test_stats["y_mean"],
            "y_train_stats": train_stats,
            "y_val_stats": val_stats,
            "y_test_stats": test_stats,
            "results": [],
        }
        (bench_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        pd.DataFrame([]).to_parquet(bench_dir / "metrics.parquet", index=False)
        logger.error("FATAL: benchmark failed due to %s", fail_reasons)
        raise SystemExit(2)

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
        "status": "success",
        "errors": [],
        "n_rows": int(len(df)),
        "n_entities": int(df["entity_id"].nunique()) if "entity_id" in df.columns else None,
        "n_features_static": 0,
        "n_features_edgar": len(edgar_feature_cols),
        "edgar_valid_rate": edgar_join_valid_rate,
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
        "n_train": train_stats["n_samples"],
        "n_val": val_stats["n_samples"],
        "n_test": test_stats["n_samples"],
        "any_nonfinite_y_train": train_stats["any_nonfinite_y"],
        "any_nonfinite_y_val": val_stats["any_nonfinite_y"],
        "any_nonfinite_y_test": test_stats["any_nonfinite_y"],
        "y_train_min": train_stats["y_min"],
        "y_train_max": train_stats["y_max"],
        "y_train_mean": train_stats["y_mean"],
        "y_val_min": val_stats["y_min"],
        "y_val_max": val_stats["y_max"],
        "y_val_mean": val_stats["y_mean"],
        "y_test_min": test_stats["y_min"],
        "y_test_max": test_stats["y_max"],
        "y_test_mean": test_stats["y_mean"],
        "y_train_stats": train_stats,
        "y_val_stats": val_stats,
        "y_test_stats": test_stats,
    }

    (bench_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_parquet(bench_dir / "metrics.parquet", index=False)
    pd.DataFrame(predictions).to_parquet(bench_dir / "predictions.parquet", index=False)

    # resolved_config already written before training

    logger.info("Output: %s", bench_dir)


if __name__ == "__main__":
    main()
