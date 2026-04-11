#!/usr/bin/env python3
"""
Run a narrow real-data local smoke slice for an arbitrary Block 3 model.

This stays outside the canonical benchmark harness and does not write under
`runs/`.  It reuses the same freeze-backed local slice loader used by the V740
smoke utilities so that new model additions can be checked quickly and fairly
before any larger submission.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.narrative.block3.models.registry import get_model
from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper
from src.narrative.block3.unified_protocol import (
    TemporalSplitConfig,
    apply_temporal_split,
    load_block3_tasks_config,
)

from scripts.run_v740_alpha_smoke_slice import (
    _compute_metrics,
    _downsample_binary_preserve_time,
    _downsample_preserve_time,
    _load_smoke_frame,
    _prepare_features,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        required=True,
        help=(
            "Registered model name or local mainline alias. "
            "Supported local aliases: single_model_mainline, single_model_mainline_delegate"
        ),
    )
    ap.add_argument("--model-kwargs", default="{}", help="JSON kwargs passed to the model factory")
    ap.add_argument("--task", required=True)
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--max-entities", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=800)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument(
        "--skip-if-output-exists",
        action="store_true",
        help="Exit successfully without re-running if output-json already exists.",
    )
    return ap.parse_args()


def _make_temporal_config() -> TemporalSplitConfig:
    tasks_config = load_block3_tasks_config()
    split_cfg = tasks_config.get("split", {})
    return TemporalSplitConfig(
        train_end=split_cfg.get("train_end", "2025-06-30"),
        val_end=split_cfg.get("val_end", "2025-09-30"),
        test_end=split_cfg.get("test_end", "2025-12-31"),
        embargo_days=split_cfg.get("embargo_days", 7),
    )


def _build_model(model_name: str, model_kwargs: Dict[str, Any]):
    kwargs = dict(model_kwargs)
    if model_name == "single_model_mainline":
        variant = kwargs.pop("variant", "mainline_alpha")
        return SingleModelMainlineWrapper(variant=variant, **kwargs)
    if model_name == "single_model_mainline_delegate":
        variant = kwargs.pop("variant", "mainline_delegate_alpha")
        kwargs.setdefault("use_delegate", True)
        return SingleModelMainlineWrapper(variant=variant, **kwargs)
    return get_model(model_name, **kwargs)


def main() -> int:
    args = _parse_args()
    if args.output_json and args.skip_if_output_exists and args.output_json.exists():
        print(json.dumps({"skipped": True, "output_json": str(args.output_json)}, indent=2))
        return 0

    model_kwargs = json.loads(args.model_kwargs)
    temporal_config = _make_temporal_config()
    df = _load_smoke_frame(args, temporal_config)
    train, val, test, _ = apply_temporal_split(df, temporal_config)
    if args.max_rows and len(train) > args.max_rows:
        if args.target == "is_funded":
            train = _downsample_binary_preserve_time(train, args.target, args.max_rows)
        else:
            train = _downsample_preserve_time(train, args.max_rows)

    X_train, y_train = _prepare_features(train, args.target)
    X_test, y_test = _prepare_features(test, args.target)
    if len(X_train) < 10 or len(X_test) < 10:
        raise RuntimeError(
            f"Insufficient rows for local smoke: train={len(X_train)} test={len(X_test)}"
        )

    model = _build_model(args.model, model_kwargs)
    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        train_raw=train,
        val_raw=val,
        target=args.target,
        task=args.task,
        ablation=args.ablation,
        horizon=args.horizon,
    )
    fit_seconds = time.time() - t0

    t1 = time.time()
    preds = model.predict(
        X_test,
        test_raw=test,
        target=args.target,
        task=args.task,
        ablation=args.ablation,
        horizon=args.horizon,
    )
    pred_seconds = time.time() - t1
    preds = np.asarray(preds, dtype=np.float64)
    metrics = _compute_metrics(y_test.to_numpy(dtype=np.float64), preds)
    pred_std = float(np.nanstd(preds)) if len(preds) else 0.0
    constant_prediction = bool(len(preds) > 1 and pred_std < 1e-8)

    payload: Dict[str, Any] = {
        "model": args.model,
        "model_kwargs": model_kwargs,
        "task": args.task,
        "ablation": args.ablation,
        "target": args.target,
        "horizon": args.horizon,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_matrix_rows": int(len(X_train)),
        "test_matrix_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "fit_seconds": fit_seconds,
        "predict_seconds": pred_seconds,
        "prediction_std": pred_std,
        "constant_prediction": constant_prediction,
        "metrics": metrics,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
