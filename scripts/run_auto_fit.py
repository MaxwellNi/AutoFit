from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Add repo root
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def build_sequences(ts_df: pd.DataFrame, seq_len: int):
    value_cols = ["funding_raised_usd", "funding_goal_usd", "investors_count", "number_of_days_left"]
    grouped = ts_df.groupby(["platform_name", "offer_id"])

    X_list = []
    y_list = []
    for _, group in grouped:
        group = group.sort_values("crawled_date")
        values = group[value_cols].values.astype(np.float32)
        if len(values) == 0:
            continue
        if len(values) >= seq_len:
            X_list.append(values[:seq_len])
        else:
            pad_len = seq_len - len(values)
            pad_vals = np.repeat(values[-1:], pad_len, axis=0)
            X_list.append(np.concatenate([values, pad_vals], axis=0))
        y_val = group["funding_ratio_w"].iloc[-1] if "funding_ratio_w" in group.columns else 0.0
        y_list.append(float(y_val))

    if not X_list:
        raise ValueError("No valid sequences found for auto-fit")

    X = np.stack(X_list)
    y = np.array(y_list)

    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean) / std
    y = (y - y.mean()) / (y.std() + 1e-8)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-fit multi-agent composer")
    parser.add_argument("--offers_path", type=Path, default=Path("data/raw/offers"))
    parser.add_argument("--limit_rows", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--budget_epochs", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--final_epochs", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    args = parser.parse_args()

    from narrative.auto_fit.compose_candidates import compose_candidates
    from narrative.auto_fit.budget_search import successive_halving, train_candidate
    from narrative.auto_fit.leaderboard import write_leaderboard
    from narrative.data_preprocessing.load_data import normalize_offers_schema
    from narrative.data_preprocessing.parquet_catalog import scan_snapshots
    from narrative.models.local_registry import list_local_models
    from narrative.data_preprocessing.build_datasets import (
        add_outcomes,
        add_time_and_peer_group,
        build_offers_core,
        filter_modelling_sample,
    )

    if args.device.startswith("cuda") and torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            arch = f"sm_{major}{minor}"
            arch_list = torch.cuda.get_arch_list()
            if arch not in arch_list:
                print(f"[WARN] CUDA arch {arch} not in build; falling back to CPU.")
                args.device = "cpu"
        except Exception:
            print("[WARN] CUDA capability check failed; using CPU.")
            args.device = "cpu"
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    args = parser.parse_args()

    if args.device.startswith("cuda") and torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
            arch = f"sm_{major}{minor}"
            arch_list = torch.cuda.get_arch_list()
            if arch not in arch_list:
                print(f"[WARN] CUDA arch {arch} not in build; falling back to CPU.")
                args.device = "cpu"
        except Exception:
            print("[WARN] CUDA capability check failed; using CPU.")
            args.device = "cpu"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path("runs/auto_fit") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_dir / "configs_used"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "run_auto_fit.json").write_text(
        json.dumps(
            {
                "offers_path": str(args.offers_path),
                "limit_rows": args.limit_rows,
                "seq_len": args.seq_len,
                "device": args.device,
                "budget_epochs": args.budget_epochs,
                "final_epochs": args.final_epochs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cfg_dir = out_dir / "configs_used"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "run_auto_fit.json").write_text(
        json.dumps(
            {
                "offers_path": str(args.offers_path),
                "limit_rows": args.limit_rows,
                "seq_len": args.seq_len,
                "device": args.device,
                "budget_epochs": args.budget_epochs,
                "final_epochs": args.final_epochs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.offers_path.suffix.lower() == ".csv":
        raise ValueError("CSV input is disabled. Provide parquet snapshots path.")

    snapshot_cols = [
        "platform_name",
        "offer_id",
        "hash_id",
        "link",
        "platform_country",
        "market_tags",
        "crawled_date",
        "funding_raised_usd",
        "funding_goal_usd",
        "investors_count",
        "number_of_days_left",
    ]
    snapshots = scan_snapshots(
        [],
        base_dir=args.offers_path,
        columns=snapshot_cols,
        allow_all=True,
        limit_rows=args.limit_rows,
    )
    snapshots = normalize_offers_schema(snapshots, schema={})
    snapshots["crawled_date"] = pd.to_datetime(snapshots["crawled_date"], errors="coerce", utc=True)
    snapshots = snapshots.dropna(subset=["platform_name", "offer_id", "crawled_date"])

    latest = snapshots.sort_values(["platform_name", "offer_id", "crawled_date"], kind="mergesort")
    latest = latest.drop_duplicates(subset=["platform_name", "offer_id"], keep="last")
    offers_core = build_offers_core(
        latest,
        {
            "id": ["platform_name", "offer_id", "hash_id", "link"],
            "context": ["platform_name", "platform_country", "market_tags"],
            "outcomes": ["funding_raised_usd", "funding_goal_usd", "investors_count"],
            "firm_controls": [],
        },
    )
    offers_core = add_outcomes(offers_core)
    offers_core = add_time_and_peer_group(offers_core)
    offers_core = filter_modelling_sample(offers_core)
    if len(offers_core) == 0 and args.limit_rows is not None:
        offers_core = add_outcomes(latest.copy())
        offers_core = add_time_and_peer_group(offers_core)
        if "funding_ratio_w" in offers_core.columns:
            offers_core["funding_ratio_w"] = offers_core["funding_ratio_w"].fillna(0.0)
        else:
            offers_core["funding_ratio_w"] = 0.0

    ts_df = snapshots.merge(
        offers_core[["platform_name", "offer_id", "funding_ratio_w"]],
        on=["platform_name", "offer_id"],
        how="inner",
    )
    meta = diagnose_dataset(
        ts_df,
        key_col="offer_id",
        time_col="crawled_date",
        target_col="funding_ratio_w",
        output_dir=out_dir,
    )

    candidates = compose_candidates(meta, available_backbones=list_local_models())

    X, y = build_sequences(ts_df, seq_len=args.seq_len)
    results, best_candidate = successive_halving(
        candidates,
        X,
        y,
        seq_len=args.seq_len,
        enc_in=X.shape[-1],
        device=args.device,
        budgets=tuple(args.budget_epochs),
        output_dir=out_dir,
        early_stopping_patience=args.early_stopping_patience,
    )
    best_path = write_leaderboard(results, out_dir)
    print(f"✓ Best config saved to: {best_path}")

    final_loss = train_candidate(
        best_candidate,
        X,
        y,
        seq_len=args.seq_len,
        enc_in=X.shape[-1],
        epochs=args.final_epochs,
        device=args.device,
    )
    (out_dir / "final_training.json").write_text(
        json.dumps({"val_loss": final_loss, "epochs": args.final_epochs}, indent=2),
        encoding="utf-8",
    )
    print(f"✓ Final training complete (val_loss={final_loss:.4f})")
    best_path = write_leaderboard(results, out_dir)
    print(f"✓ Best config saved to: {best_path}")

    final_loss = train_candidate(
        best_candidate,
        X,
        y,
        seq_len=args.seq_len,
        enc_in=X.shape[-1],
        epochs=args.final_epochs,
        device=args.device,
    )
    (out_dir / "final_training.json").write_text(
        json.dumps({"val_loss": final_loss, "epochs": args.final_epochs}, indent=2),
        encoding="utf-8",
    )
    print(f"✓ Final training complete (val_loss={final_loss:.4f})")


if __name__ == "__main__":
    main()
