#!/usr/bin/env python3
"""Strict 3-task sanity check for the VC battle pipeline.

Constructs fully synthetic panel data covering all 3 tasks × both
ablation surfaces × 2 horizons = 12 cells.  Every cell must complete
fit + predict without any error.

Exit code 0 = ALL PASS.  Non-zero = at least one failure.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.narrative.block3.models.single_model_mainline.wrapper import (
    SingleModelMainlineWrapper,
)

# ── Canonical task mapping (FIXED from battle script bug) ──────────────
_TASK_MAP = {
    "is_funded":           "task1_outcome",
    "funding_raised_usd":  "task2_forecast",
    "investors_count":     "task3_risk_adjust",
}

_N_ENTITIES  = 30
_N_ROWS      = 300   # ~10 rows per entity on average
_N_FEATURES  = 20
_RNG         = np.random.default_rng(42)


# ── Minimal synthetic data builder ────────────────────────────────────
def _build_mock_frames(target: str, ablation: str):
    """Return (train, val, test, X_train, y_train, X_test, y_test)."""
    n = _N_ROWS
    entity_ids = _RNG.integers(0, _N_ENTITIES, size=n)
    dates      = pd.date_range("2020-01-01", periods=n, freq="D")

    X = pd.DataFrame(
        _RNG.standard_normal((n, _N_FEATURES)).astype(np.float32),
        columns=[f"feat_{i}" for i in range(_N_FEATURES)],
    )

    if target == "is_funded":
        y = pd.Series((_RNG.uniform(size=n) > 0.6).astype(np.float64), name=target)
    elif target == "funding_raised_usd":
        y = pd.Series(np.abs(_RNG.exponential(1e6, size=n)), name=target)
    else:  # investors_count
        y = pd.Series(_RNG.integers(0, 15, size=n).astype(np.float64), name=target)

    # Build raw frame (includes entity_id, crawled_date_day, target col)
    raw = X.copy()
    raw["entity_id"]        = entity_ids
    raw["crawled_date_day"] = dates
    raw[target]             = y.values

    # Split 70/15/15
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_raw = raw.iloc[:n_train].reset_index(drop=True)
    val_raw   = raw.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_raw  = raw.iloc[n_train + n_val:].reset_index(drop=True)

    X_train = X.iloc[:n_train].reset_index(drop=True)
    y_train = y.iloc[:n_train].reset_index(drop=True)
    X_test  = X.iloc[n_train + n_val:].reset_index(drop=True)
    y_test  = y.iloc[n_train + n_val:].reset_index(drop=True)

    return train_raw, val_raw, test_raw, X_train, y_train, X_test, y_test


# ── Run one cell ──────────────────────────────────────────────────────
def _run_cell(
    target: str,
    ablation: str,
    horizon: int,
    use_sequential: bool = False,
) -> bool:
    """Returns True on success, False on any error."""
    task = _TASK_MAP[target]
    label = f"{task}/{ablation}/{target}/h{horizon}"
    print(f"  [{label}] ...", end=" ", flush=True)

    try:
        train, val, test, X_train, y_train, X_test, y_test = _build_mock_frames(
            target, ablation
        )

        kwargs_extra = {}
        if use_sequential:
            kwargs_extra = dict(
                enable_sequential_trunk=True,
                learnable_compact_dim=16,
                learnable_n_experts=3,
                learnable_expert_dim=8,
                learnable_top_k=1,
                seq_window_size=5,
                seq_d_model=16,
                seq_d_state=4,
                seq_n_ssm_layers=1,
                seq_n_epochs=2,
                seq_batch_size=64,
            )

        w = SingleModelMainlineWrapper(**kwargs_extra)
        w.fit(
            X_train, y_train,
            train_raw=train, val_raw=val,
            target=target,
            task=task,
            ablation=ablation,
            horizon=horizon,
        )
        preds = w.predict(
            X_test,
            test_raw=test,
            target=target,
            task=task,
            ablation=ablation,
            horizon=horizon,
        )

        assert preds is not None, "predict returned None"
        assert len(preds) == len(X_test), (
            f"pred length mismatch: {len(preds)} vs {len(X_test)}"
        )
        assert np.all(np.isfinite(preds)), (
            f"non-finite predictions: {np.sum(~np.isfinite(preds))} NaN/Inf"
        )

        mae = float(np.mean(np.abs(preds - y_test.to_numpy())))
        print(f"OK  MAE={mae:.6f}")
        return True

    except Exception:
        print("FAIL")
        traceback.print_exc()
        return False


def main():
    targets   = ["is_funded", "funding_raised_usd", "investors_count"]
    ablations = ["core_only", "core_edgar"]
    horizons  = [1, 7]

    total, passed, failed = 0, 0, 0
    failures = []

    # ── Arm 1: Original backbone (no trunk) ─────────────────────────
    print("\n=== Arm 1: Original backbone (no trunk) ===")
    for tgt in targets:
        for abl in ablations:
            for h in horizons:
                total += 1
                ok = _run_cell(tgt, abl, h, use_sequential=False)
                if ok:
                    passed += 1
                else:
                    failed += 1
                    failures.append(f"orig/{tgt}/{abl}/h{h}")

    # ── Arm 2: Sequential ED-SSM + MoE trunk ────────────────────────
    print("\n=== Arm 2: Sequential ED-SSM + MoE trunk ===")
    for tgt in targets:
        for abl in ablations:
            for h in horizons:
                total += 1
                ok = _run_cell(tgt, abl, h, use_sequential=True)
                if ok:
                    passed += 1
                else:
                    failed += 1
                    failures.append(f"edssm/{tgt}/{abl}/h{h}")

    print(f"\n{'='*60}")
    print(f"Sanity Check: {passed}/{total} PASS  |  {failed} FAIL")
    if failures:
        print("Failed cells:")
        for f in failures:
            print(f"  - {f}")
        print("SANITY CHECK FAILED — DO NOT SUBMIT JOB")
        sys.exit(1)
    else:
        print("ALL PASS — pipeline cleared for Slurm submission")
        sys.exit(0)


if __name__ == "__main__":
    main()
