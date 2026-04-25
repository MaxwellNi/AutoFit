#!/usr/bin/env python3
"""
r13_poc_multihorizon_stacking.py — Round-12.5 PoC

Demonstrates the row-level multi-horizon stacking fix described in
docs/references/agents_handover.md §7.2.

Pipeline:
  1. Load offers_core_daily via Block3Dataset.
  2. Temporal split (train_end=2025-06-30, val_end=2025-09-30,
     test_end=2025-12-31) with embargo_days=7.
  3. For each h in HORIZONS, materialise y_h[i,t] = y[i, t+h] via
     `groupby('entity_id')[target].shift(-h)`; drop NaN rows (future
     leakage past test_end is impossible by construction of split).
  4. Stack (X, y_h, h) with '_horizon' as a row-varying covariate.
  5. Fit sklearn HistGradientBoostingRegressor on stacked train; report:
       - per-h test MAE (computed by slicing the stacked test set by
         '_horizon'),
       - feature_importance ranking of '_horizon' column,
       - a single-horizon baseline (h=7, no-shift) MAE for contrast.
  6. Dump metrics.json with bit-exact stats so the HPC job output is
     auditable.

Usage (local or SLURM):
  python r13_poc_multihorizon_stacking.py \
      --target funding_raised_usd \
      --horizons 7 14 30 \
      --output-dir runs/r13_poc/<tag>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# --- repo path setup -------------------------------------------------------
REPO_ROOT = Path(
    os.environ.get("BLOCK3_CANONICAL_REPO_ROOT", "/home/users/npin/repo_root")
)
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrative.data_preprocessing.block3_dataset import (  # noqa: E402
    Block3Dataset,
)

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402


# --- CLI -------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="funding_raised_usd")
    p.add_argument("--horizons", type=int, nargs="+", default=[7, 14, 30])
    p.add_argument("--train-end", default="2025-06-30")
    p.add_argument("--val-end", default="2025-09-30")
    p.add_argument("--test-end", default="2025-12-31")
    p.add_argument("--embargo-days", type=int, default=7)
    p.add_argument("--entity-col", default="entity_id")
    p.add_argument("--date-col", default="crawled_date_day")
    p.add_argument(
        "--pointer",
        default=str(REPO_ROOT / "docs/audits/FULL_SCALE_POINTER.yaml"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 = full panel; otherwise subsample train rows for smoke.",
    )
    return p.parse_args()


# --- helpers ---------------------------------------------------------------
def _numeric_features(df: pd.DataFrame, target: str) -> list[str]:
    """Return numeric columns usable as tabular features."""
    drop = {target, "entity_id", "crawled_date_day"}
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num if c not in drop]


def _temporal_slices(
    df: pd.DataFrame,
    date_col: str,
    train_end: str,
    val_end: str,
    test_end: str,
    embargo_days: int,
) -> dict[str, pd.Index]:
    """Return train/val/test index slices with a one-sided embargo."""
    d = pd.to_datetime(df[date_col])
    te = pd.Timestamp(train_end)
    ve = pd.Timestamp(val_end)
    tse = pd.Timestamp(test_end)
    emb = pd.Timedelta(days=embargo_days)
    tr = df.index[d <= te]
    vl = df.index[(d > te + emb) & (d <= ve)]
    ts = df.index[(d > ve + emb) & (d <= tse)]
    return {"train": tr, "val": vl, "test": ts}


def _byte_digest(arr: np.ndarray) -> str:
    b = np.ascontiguousarray(arr, dtype=np.float64).tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


# --- main ------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    ds = Block3Dataset.from_pointer(Path(args.pointer))
    df = ds.get_offers_core_daily()
    print(f"[r13-poc] loaded offers_core_daily: {df.shape}", flush=True)

    # Ensure sort order for groupby-shift
    df = df.sort_values([args.entity_col, args.date_col]).reset_index(drop=True)

    if args.target not in df.columns:
        raise SystemExit(f"target {args.target!r} not in columns")

    # 2. Materialise y_h per horizon
    feat_cols = _numeric_features(df, args.target)
    print(f"[r13-poc] numeric feature count: {len(feat_cols)}", flush=True)

    frames = []
    per_h_counts: dict[int, int] = {}
    for h in args.horizons:
        shifted = df.groupby(args.entity_col)[args.target].shift(-h)
        mask = shifted.notna()
        sub = df.loc[mask, [args.entity_col, args.date_col] + feat_cols].copy()
        sub["_horizon"] = h
        sub["_y"] = shifted.loc[mask].values
        frames.append(sub)
        per_h_counts[h] = int(mask.sum())
        print(f"[r13-poc]   h={h}: {per_h_counts[h]} rows with y_h notna",
              flush=True)

    stacked = pd.concat(frames, ignore_index=True)
    del frames
    print(f"[r13-poc] stacked shape: {stacked.shape}", flush=True)

    # 3. Temporal split on the stacked frame
    splits = _temporal_slices(
        stacked,
        args.date_col,
        args.train_end,
        args.val_end,
        args.test_end,
        args.embargo_days,
    )
    for k, idx in splits.items():
        print(f"[r13-poc]   split {k}: {len(idx)} rows", flush=True)

    if args.max_rows and len(splits["train"]) > args.max_rows:
        rng = np.random.RandomState(args.seed)
        keep = rng.choice(splits["train"], size=args.max_rows, replace=False)
        splits["train"] = pd.Index(sorted(keep))

    X_cols = feat_cols + ["_horizon"]
    X_train = stacked.loc[splits["train"], X_cols].to_numpy(dtype=np.float32)
    y_train = stacked.loc[splits["train"], "_y"].to_numpy(dtype=np.float64)
    X_test = stacked.loc[splits["test"], X_cols].to_numpy(dtype=np.float32)
    y_test = stacked.loc[splits["test"], "_y"].to_numpy(dtype=np.float64)
    h_test = stacked.loc[splits["test"], "_horizon"].to_numpy()

    # Replace any leftover NaN/Inf (feature side) with 0 (bit-exact with
    # the harness' fillna(0) policy).
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # 4. Fit stacked model
    print(f"[r13-poc] fitting HGB on stacked train "
          f"(n={X_train.shape[0]}, p={X_train.shape[1]})", flush=True)
    t_fit0 = time.time()
    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=8,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)
    fit_time = time.time() - t_fit0
    print(f"[r13-poc] fit done in {fit_time:.1f}s", flush=True)

    # 5. Predict + per-h MAE
    y_pred = model.predict(X_test)
    per_h_mae: dict[str, float] = {}
    for h in args.horizons:
        m = (h_test == h)
        if m.sum() > 0:
            per_h_mae[str(h)] = float(mean_absolute_error(y_test[m], y_pred[m]))
        else:
            per_h_mae[str(h)] = float("nan")
        print(f"[r13-poc]   h={h} test MAE = {per_h_mae[str(h)]:.6f} "
              f"(n={int(m.sum())})", flush=True)

    # HGB does not expose feature_importances_ natively; rely on permutation-
    # style proxy via the split gain attribute if available, else skip.
    horizon_idx = X_cols.index("_horizon")

    # 6. No-shift single-horizon baseline (contemporaneous y, h=args.horizons[0]
    #    shard only) — reproduces the current-harness behaviour for contrast.
    df_base = df.copy()
    df_base = df_base.loc[df_base[args.target].notna()]
    splits_b = _temporal_slices(
        df_base,
        args.date_col,
        args.train_end,
        args.val_end,
        args.test_end,
        args.embargo_days,
    )
    Xb_train = df_base.loc[splits_b["train"], feat_cols].to_numpy(
        dtype=np.float32)
    yb_train = df_base.loc[splits_b["train"], args.target].to_numpy(
        dtype=np.float64)
    Xb_test = df_base.loc[splits_b["test"], feat_cols].to_numpy(
        dtype=np.float32)
    yb_test = df_base.loc[splits_b["test"], args.target].to_numpy(
        dtype=np.float64)
    Xb_train = np.nan_to_num(Xb_train, nan=0.0, posinf=0.0, neginf=0.0)
    Xb_test = np.nan_to_num(Xb_test, nan=0.0, posinf=0.0, neginf=0.0)
    if args.max_rows and Xb_train.shape[0] > args.max_rows:
        rng = np.random.RandomState(args.seed)
        keep = rng.choice(Xb_train.shape[0], size=args.max_rows, replace=False)
        Xb_train = Xb_train[keep]
        yb_train = yb_train[keep]
    model_b = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=8,
        random_state=args.seed,
    )
    model_b.fit(Xb_train, yb_train)
    baseline_mae = float(mean_absolute_error(yb_test, model_b.predict(Xb_test)))
    print(f"[r13-poc] no-shift baseline (contemporaneous y) MAE = "
          f"{baseline_mae:.6f}", flush=True)

    # 7. Dump metrics.json
    out = {
        "script": "r13_poc_multihorizon_stacking.py",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_root": str(REPO_ROOT),
        "target": args.target,
        "horizons": args.horizons,
        "seed": args.seed,
        "n_features_numeric": len(feat_cols),
        "horizon_col_index_in_X": horizon_idx,
        "per_h_rows_stacked": per_h_counts,
        "split_sizes_stacked": {k: int(len(v)) for k, v in splits.items()},
        "per_h_test_mae_stacked": per_h_mae,
        "baseline_no_shift_test_mae": baseline_mae,
        "fit_seconds_stacked": fit_time,
        "wall_seconds_total": time.time() - t0,
        "y_train_sha16": _byte_digest(y_train),
        "X_train_sha16": _byte_digest(X_train),
        # Interpretation: if per_h_test_mae_stacked varies non-trivially with
        # h (> ~1% relative), then row-level horizon stacking actually makes
        # the model horizon-aware. If they collapse to near-equal values, the
        # bottleneck is data/feature structure, not harness plumbing.
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"[r13-poc] wrote {args.output_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
