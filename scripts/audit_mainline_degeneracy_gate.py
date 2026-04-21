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


def _hurdle_baseline_mae(targets: np.ndarray) -> float:
    """Per-cell hurdle baseline: y_hat = pi_hat * m_hat (cell-wise)."""
    targets = np.asarray(targets, dtype=float)
    finite = targets[np.isfinite(targets)]
    if finite.size == 0:
        return float("nan")
    positive = finite[finite > 0]
    pi_hat = positive.size / max(finite.size, 1)
    m_hat = float(np.mean(positive)) if positive.size else 0.0
    return float(np.mean(np.abs(finite - pi_hat * m_hat)))


def _brier_vs_baseline(preds: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Return (candidate_brier, baseline_brier) for the occurrence component.

    Both predictor and baseline are reduced to their implied pi_hat on the
    cell. Strictly proper for binary events. Paper §5.2 (Gneiting-Raftery).
    """
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    finite = np.isfinite(targets)
    if not finite.any():
        return float("nan"), float("nan")
    y_bin = (targets[finite] > 0).astype(float)
    p_pred = np.clip(preds[finite], 0.0, np.inf)
    # Candidate occurrence proxy: sign of positive prediction.
    # (A well-specified probabilistic predictor would expose a pi head;
    # we approximate from the point predictor as a lower bound.)
    p_cand = (p_pred > 0).astype(float)
    p_base = float(y_bin.mean())
    brier_cand = float(np.mean((p_cand - y_bin) ** 2))
    brier_base = float(np.mean((p_base - y_bin) ** 2))
    return brier_cand, brier_base


def _pinball_positive(preds: np.ndarray, targets: np.ndarray,
                      quantiles: tuple[float, ...] = (0.5, 0.9, 0.95)) -> float:
    """Mean pinball loss of point predictions on {y>0} subset, averaged over
    the requested quantiles. Strictly proper per-quantile; conservative for
    point predictors (we treat the predictor as its own quantile estimate at
    each q, which is a pessimistic upper bound used only to produce a
    single monotonic skill number)."""
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    mask = np.isfinite(targets) & (targets > 0)
    if not mask.any():
        return float("nan")
    y = targets[mask]
    p = preds[mask]
    loss = 0.0
    for q in quantiles:
        diff = y - p
        loss += float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
    return loss / len(quantiles)


def _skill_margin(model_mae: float, baseline_mae: float) -> float:
    if not np.isfinite(model_mae) or not np.isfinite(baseline_mae) or baseline_mae <= 0:
        return float("nan")
    return (baseline_mae - model_mae) / baseline_mae


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
                target_vec = cell.get("target_value", pd.Series(dtype=float)).to_numpy()
                pred_vec = cell.get("prediction", pd.Series(dtype=float)).to_numpy()
                cv, s_ref = _cv_for_cell(pred_vec, target_vec)
                hurdle_mae = _hurdle_baseline_mae(target_vec)
                skill = _skill_margin(mae, hurdle_mae)
                # Strictly proper scoring components (paper §5.2).
                brier_cand, brier_base = _brier_vs_baseline(pred_vec, target_vec)
                brier_skill = _skill_margin(brier_cand, brier_base)
                pinball = _pinball_positive(pred_vec, target_vec)
            else:
                cv, s_ref, hurdle_mae, skill = (float("nan"),) * 4
                brier_cand = brier_base = brier_skill = pinball = float("nan")
            rows.append({
                "metrics_path": str(metrics_path),
                "model_name": model,
                "task": task,
                "ablation": ablation,
                "target": target,
                "horizon": horizon,
                "mae": mae,
                "hurdle_baseline_mae": hurdle_mae,
                "skill_margin": skill,
                "brier_candidate": brier_cand,
                "brier_baseline": brier_base,
                "brier_skill": brier_skill,
                "pinball_positive": pinball,
                "cv": cv,
                "s_ref": s_ref,
                "cell_gate_fail": bool(np.isfinite(cv) and cv < tau),
                # Admissibility now requires the strictly-proper Brier score to
                # beat baseline as well (paper §5.2); we relax to the weaker
                # MAE skill margin only when predictions.parquet is absent.
                "cell_skill_fail": bool(
                    (np.isfinite(brier_skill) and brier_skill <= 0.0)
                    or (not np.isfinite(brier_skill) and np.isfinite(skill) and skill <= 0.0)
                ),
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

    # Aggregate admissibility per (model, task, ablation, target).
    # A triple is admissible iff it passes BOTH the CV gate (necessary,
    # Definition 3.8) AND the skill check (sufficient filter against
    # high-variance random predictors, Remark on gate-not-sufficient).
    agg = (
        group.agg(
            n_cells=("cell_gate_fail", "size"),
            n_cell_gate_fail=("cell_gate_fail", "sum"),
            n_cell_skill_fail=("cell_skill_fail", "sum"),
            any_identical_h=("identical_mae_across_h", "any"),
            median_skill=("skill_margin", "median"),
        )
        .reset_index()
    )
    agg["cell_gate_fail_rate"] = agg["n_cell_gate_fail"] / agg["n_cells"].clip(lower=1)
    agg["cell_skill_fail_rate"] = agg["n_cell_skill_fail"] / agg["n_cells"].clip(lower=1)
    agg["triple_fail"] = (
        (agg["cell_gate_fail_rate"] > rho_fail)
        | (agg["cell_skill_fail_rate"] > rho_fail)
        | agg["any_identical_h"].fillna(False)
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
        "n_cells", "cell_gate_fail_rate", "cell_skill_fail_rate",
        "median_skill", "any_identical_h", "triple_fail",
    ]
    print(agg[cols].to_string(index=False))
    print()
    failed = agg[agg["triple_fail"]]
    if not failed.empty:
        print(f"[FAIL] {len(failed)} triples fail admissibility "
              f"(tau={args.tau}, rho_fail={args.rho_fail}, "
              f"skill = MAE vs hurdle baseline):")
        for _, row in failed.iterrows():
            print(f"  - {row['model_name']}/{row['task']}/{row['ablation']}/"
                  f"{row['target']}: gate_fail={row['cell_gate_fail_rate']:.2f} "
                  f"skill_fail={row['cell_skill_fail_rate']:.2f} "
                  f"median_skill={row['median_skill']:.3f} "
                  f"identical_h={row['any_identical_h']}")
    else:
        print("[OK] all triples admissible (gate + skill).")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(args.out, index=False)
        print(f"[info] wrote {args.out}")

    return 0 if failed.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
