#!/usr/bin/env python3
"""Apply the CV-based degeneracy admissibility gate (paper §3.5 Def 3.8) to
mainline-variant metrics.json artefacts and surface constant-collapsed cells.

Gate, following the paper:

    CV_M(\\hat y) = sqrt(Var_M(\\hat y)) / s_ref

with s_ref = median absolute deviation of strictly positive targets on the
evaluation cell. A cell fails the gate if CV_M < tau (default 0.05). A
(model, ablation, target) triple fails the gate if > rho_fail (default 0.5)
of its horizon cells fail individually.

This audit also checks for the pathological "identical across horizons"
signature (std of per-h MAE <= 1e-6) that flags the `h=7/14/30 MAE
bit-identical` regression we observed in mn_v2_full_* and
mn_v2_naked_t1_* on 2026-04-21.

Usage:
    python3 scripts/audit_mainline_degeneracy_gate.py \\
        runs/benchmarks/mn_v2_naked_t1_l40s_20260421 \\
        runs/benchmarks/mn_v2_full_t2_gpu_20260421 \\
        --tau 0.05 --rho-fail 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _iter_metrics(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_file() and p.name == "metrics.json":
            yield p
        elif p.is_dir():
            for q in p.rglob("metrics.json"):
                yield q


def _load_predictions(metrics_path: Path) -> pd.DataFrame | None:
    pred_path = metrics_path.parent / "predictions.parquet"
    if not pred_path.exists():
        return None
    try:
        return pd.read_parquet(pred_path)
    except Exception as exc:  # pragma: no cover - file-format drift
        print(f"[warn] cannot read {pred_path}: {exc}", file=sys.stderr)
        return None


def _cv_for_cell(preds: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if preds.size == 0:
        return 0.0, 0.0
    pred_std = float(np.nanstd(preds))
    positive = targets[(np.isfinite(targets)) & (targets > 0)]
    if positive.size == 0:
        s_ref = float(np.nanstd(targets)) or 1.0
    else:
        med = float(np.nanmedian(positive))
        s_ref = float(np.nanmedian(np.abs(positive - med))) or 1.0
    return pred_std / s_ref if s_ref > 0 else 0.0, s_ref


def audit(roots: list[Path], tau: float, rho_fail: float) -> pd.DataFrame:
    rows = []
    for metrics_path in _iter_metrics(roots):
        try:
            records = json.load(open(metrics_path))
        except PermissionError:
            continue
        except Exception as exc:  # pragma: no cover
            print(f"[warn] cannot parse {metrics_path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(records, list):
            continue
        preds_df = _load_predictions(metrics_path)
        for rec in records:
            model = rec.get("model_name")
            task = rec.get("task")
            ablation = rec.get("ablation")
            target = rec.get("target")
            horizon = rec.get("horizon")
            mae = rec.get("mae")
            if preds_df is not None:
                mask = (
                    (preds_df.get("target") == target)
                    & (preds_df.get("horizon") == horizon)
                )
                cell = preds_df[mask] if mask.any() else preds_df
                cv, s_ref = _cv_for_cell(
                    cell.get("prediction", pd.Series(dtype=float)).to_numpy(),
                    cell.get("target_value", pd.Series(dtype=float)).to_numpy(),
                )
            else:
                cv, s_ref = float("nan"), float("nan")
            rows.append({
                "metrics_path": str(metrics_path),
                "model_name": model,
                "task": task,
                "ablation": ablation,
                "target": target,
                "horizon": horizon,
                "mae": mae,
                "cv": cv,
                "s_ref": s_ref,
                "cell_fail": bool(np.isfinite(cv) and cv < tau),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Identical-across-horizons regression signature.
    group = df.groupby(
        ["metrics_path", "model_name", "task", "ablation", "target"], dropna=False
    )
    df["mae_std_across_h"] = group["mae"].transform(lambda s: np.nanstd(s.values))
    df["identical_mae_across_h"] = df["mae_std_across_h"] < 1e-6

    # Aggregate gate decision per (model, task, ablation, target).
    agg = (
        group.agg(
            n_cells=("cell_fail", "size"),
            n_cell_fail=("cell_fail", "sum"),
            any_identical_h=("identical_mae_across_h", "any"),
        )
        .reset_index()
    )
    agg["cell_fail_rate"] = agg["n_cell_fail"] / agg["n_cells"].clip(lower=1)
    agg["triple_fail"] = (
        (agg["cell_fail_rate"] > rho_fail) | agg["any_identical_h"].fillna(False)
    )
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--rho-fail", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    agg = audit(args.roots, tau=args.tau, rho_fail=args.rho_fail)
    if agg.empty:
        print("[info] no metrics.json found under provided roots")
        return 0

    pd.set_option("display.max_colwidth", 80)
    pd.set_option("display.width", 160)
    cols = [
        "model_name", "task", "ablation", "target",
        "n_cells", "n_cell_fail", "cell_fail_rate",
        "any_identical_h", "triple_fail",
    ]
    print(agg[cols].to_string(index=False))
    print()
    failed = agg[agg["triple_fail"]]
    if not failed.empty:
        print(f"[FAIL] {len(failed)} triples fail degeneracy gate "
              f"(tau={args.tau}, rho_fail={args.rho_fail}):")
        for _, row in failed.iterrows():
            print(f"  - {row['model_name']}/{row['task']}/{row['ablation']}/"
                  f"{row['target']}: fail_rate={row['cell_fail_rate']:.2f}, "
                  f"identical_h={row['any_identical_h']}")
    else:
        print("[OK] all triples pass degeneracy gate.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(args.out, index=False)
        print(f"[info] wrote {args.out}")

    return 0 if failed.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
