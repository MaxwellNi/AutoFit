#!/usr/bin/env python3
"""
P7 Economic Constraint Audit — Cross-Head Consistency Checker.

Fits both `single_model_mainline` and a benchmark champion (default: DeepNPTS)
on the same temporal split, predicts all 3 targets on the same test set, then
flags violations of basic economic logic:

  1. Ghost Funding:   investors == 0  BUT  funding > 0
  2. Logical Inversion: P(is_funded) >= 0.90  BUT  funding in bottom-10% tail

Output: per-model violation rates + summary table.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v740_alpha_minibenchmark import _build_case_frame, _make_temporal_config
from scripts.run_v740_alpha_smoke_slice import _prepare_features
from src.narrative.block3.models.registry import check_model_available, get_model
from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper

# ---------------------------------------------------------------------------
# Violation detectors
# ---------------------------------------------------------------------------

_GHOST_FUNDING_INVESTOR_THRESHOLD = 0.5   # predicted investors rounded to 0
_INVERSION_PROB_THRESHOLD = 0.90          # high confidence funded
_INVERSION_FUNDING_QUANTILE = 0.10        # bottom 10% of funding distribution


def detect_ghost_funding(
    pred_investors: np.ndarray,
    pred_funding: np.ndarray,
) -> np.ndarray:
    """Return boolean mask: True where investors ≈ 0 but funding > 0."""
    investors_zero = pred_investors < _GHOST_FUNDING_INVESTOR_THRESHOLD
    funding_positive = pred_funding > 0.0
    return investors_zero & funding_positive


def detect_logical_inversion(
    pred_binary: np.ndarray,
    pred_funding: np.ndarray,
    funding_q10: float,
) -> np.ndarray:
    """Return boolean mask: True where P(funded) >= 90% but funding in bottom 10%."""
    high_prob = pred_binary >= _INVERSION_PROB_THRESHOLD
    low_funding = pred_funding <= funding_q10
    return high_prob & low_funding


# ---------------------------------------------------------------------------
# Model fitting helpers
# ---------------------------------------------------------------------------

TARGETS = ["is_funded", "funding_raised_usd", "investors_count"]


def _fit_and_predict_mainline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    task: str,
    ablation: str,
    horizon: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Fit 3 separate mainline wrappers, return {target: y_pred}."""
    results = {}
    for target in TARGETS:
        X_train, y_train = _prepare_features(train, target)
        X_test, _ = _prepare_features(test, target)
        if len(X_train) < 10 or len(X_test) < 10:
            results[target] = np.full(len(test), np.nan)
            continue
        wrapper = SingleModelMainlineWrapper(seed=seed)
        wrapper.fit(
            X_train, y_train,
            train_raw=train, val_raw=val,
            target=target, task=task,
            ablation=ablation, horizon=horizon,
        )
        preds = wrapper.predict(
            X_test, test_raw=test,
            target=target, task=task,
            ablation=ablation, horizon=horizon,
        )
        results[target] = np.asarray(preds, dtype=np.float64)
    return results


def _fit_and_predict_benchmark(
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    task: str,
    ablation: str,
    horizon: int,
) -> Dict[str, np.ndarray]:
    """Fit 3 separate benchmark model instances, return {target: y_pred}."""
    results = {}
    for target in TARGETS:
        X_train, y_train = _prepare_features(train, target)
        X_test, _ = _prepare_features(test, target)
        if len(X_train) < 10 or len(X_test) < 10:
            results[target] = np.full(len(test), np.nan)
            continue
        model = get_model(model_name)
        model.fit(
            X_train, y_train,
            train_raw=train, val_raw=val,
            target=target, task=task,
            ablation=ablation, horizon=horizon,
        )
        preds = model.predict(
            X_test, test_raw=test,
            target=target, task=task,
            ablation=ablation, horizon=horizon,
        )
        results[target] = np.asarray(preds, dtype=np.float64)
    return results


# ---------------------------------------------------------------------------
# Audit logic
# ---------------------------------------------------------------------------

def audit_cell(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    task: str,
    ablation: str,
    horizon: int,
    benchmark_model: str,
) -> Dict[str, Any]:
    """Run full economic constraint audit on one cell."""
    cell_name = f"{task}__{ablation}__h{horizon}"
    print(f"\n{'='*60}")
    print(f"  Auditing: {cell_name}")
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"{'='*60}")

    # --- fit mainline ---
    t0 = time.time()
    mainline_preds = _fit_and_predict_mainline(
        train, val, test, task, ablation, horizon,
    )
    mainline_time = time.time() - t0
    print(f"  Mainline fit+predict: {mainline_time:.1f}s")

    # --- fit benchmark ---
    t1 = time.time()
    bench_preds = _fit_and_predict_benchmark(
        benchmark_model, train, val, test, task, ablation, horizon,
    )
    bench_time = time.time() - t1
    print(f"  {benchmark_model} fit+predict: {bench_time:.1f}s")

    # --- compute funding Q10 from training data ---
    _, y_fund_train = _prepare_features(train, "funding_raised_usd")
    funding_q10 = float(np.nanquantile(y_fund_train.to_numpy(), _INVERSION_FUNDING_QUANTILE))
    print(f"  Funding Q10 (train): {funding_q10:,.2f}")

    # --- detect violations ---
    cell_result = {"cell": cell_name, "test_rows": len(test), "funding_q10": funding_q10}
    for label, preds in [("mainline", mainline_preds), (benchmark_model, bench_preds)]:
        p_binary = preds["is_funded"]
        p_funding = preds["funding_raised_usd"]
        p_investors = preds["investors_count"]

        n_valid = int(np.isfinite(p_binary).sum())
        ghost = detect_ghost_funding(p_investors, p_funding)
        inversion = detect_logical_inversion(p_binary, p_funding, funding_q10)

        ghost_count = int(ghost.sum())
        inversion_count = int(inversion.sum())
        ghost_rate = ghost_count / max(n_valid, 1)
        inversion_rate = inversion_count / max(n_valid, 1)

        cell_result[f"{label}_ghost_count"] = ghost_count
        cell_result[f"{label}_ghost_rate"] = round(ghost_rate, 4)
        cell_result[f"{label}_inversion_count"] = inversion_count
        cell_result[f"{label}_inversion_rate"] = round(inversion_rate, 4)
        cell_result[f"{label}_pred_binary_mean"] = round(float(np.nanmean(p_binary)), 4)
        cell_result[f"{label}_pred_funding_mean"] = round(float(np.nanmean(p_funding)), 2)
        cell_result[f"{label}_pred_investors_mean"] = round(float(np.nanmean(p_investors)), 4)
        cell_result[f"{label}_pred_funding_std"] = round(float(np.nanstd(p_funding)), 2)
        cell_result[f"{label}_pred_investors_std"] = round(float(np.nanstd(p_investors)), 4)

        print(f"\n  [{label}]")
        print(f"    Ghost Funding:     {ghost_count}/{n_valid} = {ghost_rate:.2%}")
        print(f"    Logical Inversion: {inversion_count}/{n_valid} = {inversion_rate:.2%}")
        print(f"    P(funded) mean:    {np.nanmean(p_binary):.4f}")
        print(f"    Funding mean:      {np.nanmean(p_funding):,.2f}")
        print(f"    Investors mean:    {np.nanmean(p_investors):.4f}")

    return cell_result


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict[str, Any]], benchmark_model: str, output_path: Path | None = None) -> str:
    """Print and optionally save a markdown summary table."""
    lines = []
    lines.append("# P7 Economic Constraint Audit — Summary")
    lines.append("")
    lines.append(f"> Benchmark: `{benchmark_model}`")
    lines.append(f"> Date: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> Cells audited: {len(results)}")
    lines.append("")

    # Aggregate
    ml_ghost_total = sum(r["mainline_ghost_count"] for r in results)
    ml_inv_total = sum(r["mainline_inversion_count"] for r in results)
    bm_ghost_total = sum(r[f"{benchmark_model}_ghost_count"] for r in results)
    bm_inv_total = sum(r[f"{benchmark_model}_inversion_count"] for r in results)
    total_rows = sum(r["test_rows"] for r in results)

    lines.append("## Aggregate Violation Rates")
    lines.append("")
    lines.append("| Model | Ghost Funding | Rate | Logical Inversion | Rate | Total Test Rows |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| mainline | {ml_ghost_total} | {ml_ghost_total/max(total_rows,1):.2%} "
        f"| {ml_inv_total} | {ml_inv_total/max(total_rows,1):.2%} | {total_rows} |"
    )
    lines.append(
        f"| {benchmark_model} | {bm_ghost_total} | {bm_ghost_total/max(total_rows,1):.2%} "
        f"| {bm_inv_total} | {bm_inv_total/max(total_rows,1):.2%} | {total_rows} |"
    )
    lines.append("")

    # Per-cell detail
    lines.append("## Per-Cell Detail")
    lines.append("")
    lines.append("| Cell | Model | Ghost | Ghost% | Inversion | Inversion% | P(funded) | Funding μ | Investors μ |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        cell = r["cell"]
        for label in ["mainline", benchmark_model]:
            lines.append(
                f"| {cell} | {label} "
                f"| {r[f'{label}_ghost_count']} "
                f"| {r[f'{label}_ghost_rate']:.2%} "
                f"| {r[f'{label}_inversion_count']} "
                f"| {r[f'{label}_inversion_rate']:.2%} "
                f"| {r[f'{label}_pred_binary_mean']:.4f} "
                f"| {r[f'{label}_pred_funding_mean']:,.2f} "
                f"| {r[f'{label}_pred_investors_mean']:.4f} |"
            )
    lines.append("")

    # Verdict
    ml_total_violations = ml_ghost_total + ml_inv_total
    bm_total_violations = bm_ghost_total + bm_inv_total
    lines.append("## Verdict")
    lines.append("")
    if bm_total_violations > ml_total_violations:
        advantage = bm_total_violations - ml_total_violations
        lines.append(
            f"**Mainline wins on economic consistency.** "
            f"{benchmark_model} has {advantage} more violations "
            f"({bm_total_violations} vs {ml_total_violations})."
        )
    elif ml_total_violations > bm_total_violations:
        advantage = ml_total_violations - bm_total_violations
        lines.append(
            f"**{benchmark_model} wins on economic consistency.** "
            f"Mainline has {advantage} more violations "
            f"({ml_total_violations} vs {bm_total_violations})."
        )
    else:
        lines.append(f"**Tied** on total violations ({ml_total_violations} each).")

    text = "\n".join(lines)
    print(text)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"\nSaved to: {output_path}")
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="P7 Economic Constraint Audit")
    parser.add_argument("--benchmark", default="DeepNPTS", help="Benchmark model name")
    parser.add_argument("--task", default="task1_outcome", help="Task family")
    parser.add_argument(
        "--ablations", nargs="+", default=["core_only", "core_edgar"],
        help="Ablation regimes to audit",
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[1, 7, 14],
        help="Horizons to audit",
    )
    parser.add_argument(
        "--max-entities", type=int, default=None,
        help="Cap entities per cell for speed",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Cap training rows per cell for speed",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output files",
    )
    args = parser.parse_args()

    temporal_config = _make_temporal_config()
    all_results: List[Dict[str, Any]] = []

    for ablation in args.ablations:
        for horizon in args.horizons:
            case = {
                "task": args.task,
                "ablation": ablation,
                "target": "is_funded",  # placeholder — we override per-target
                "horizon": horizon,
                "max_entities": args.max_entities,
                "max_rows": args.max_rows,
                "name": f"{args.task}__{ablation}__h{horizon}",
            }
            train, val, test = _build_case_frame(case, temporal_config)
            result = audit_cell(
                train, val, test,
                task=args.task,
                ablation=ablation,
                horizon=horizon,
                benchmark_model=args.benchmark,
            )
            all_results.append(result)

    # --- Output ---
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"runs/benchmarks/single_model_mainline_localclear_20260420/p7_economic_audit"
    )
    summary_path = out_dir / "economic_audit_summary.md"
    json_path = out_dir / "economic_audit_results.json"

    print_summary(all_results, args.benchmark, output_path=summary_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"JSON results: {json_path}")


if __name__ == "__main__":
    main()
