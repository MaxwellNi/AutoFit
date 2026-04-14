#!/usr/bin/env python3
"""Compare the narrowed investors horizon contract against the baseline on official local slices."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_mainline_investors_source_ablation import SliceCase, _load_case_frames, _make_temporal_config
from scripts.run_v740_alpha_smoke_slice import _compute_metrics, _prepare_features
from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper


DEFAULT_CASES = (
    SliceCase(task="task1_outcome", ablation="core_only", horizon=1),
    SliceCase(task="task1_outcome", ablation="core_only", horizon=7),
    SliceCase(task="task1_outcome", ablation="core_only", horizon=14),
    SliceCase(task="task1_outcome", ablation="core_only", horizon=30),
    SliceCase(task="task2_forecast", ablation="core_edgar", horizon=1),
    SliceCase(task="task2_forecast", ablation="core_edgar", horizon=7),
    SliceCase(task="task2_forecast", ablation="core_edgar", horizon=14),
    SliceCase(task="task2_forecast", ablation="core_edgar", horizon=30),
    SliceCase(task="task2_forecast", ablation="full", horizon=1),
    SliceCase(task="task2_forecast", ablation="full", horizon=7),
    SliceCase(task="task2_forecast", ablation="full", horizon=14),
    SliceCase(task="task2_forecast", ablation="full", horizon=30),
)


VARIANTS: Dict[str, Dict[str, Any]] = {
    "legacy_baseline": {
        "enable_investors_horizon_contract": False,
        "enable_count_hurdle_head": False,
        "enable_count_jump": False,
        "enable_count_sparsity_gate": False,
    },
    "hurdle_only": {
        "enable_count_hurdle_head": True,
        "enable_count_jump": False,
        "enable_count_sparsity_gate": False,
    },
    "hurdle_plus_jump": {
        "enable_count_hurdle_head": True,
        "enable_count_jump": True,
        "count_jump_strength": 0.30,
        "enable_count_sparsity_gate": False,
    },
    "hurdle_plus_sparsity": {
        "enable_count_hurdle_head": True,
        "enable_count_jump": False,
        "enable_count_sparsity_gate": True,
        "count_sparsity_gate_strength": 0.75,
    },
    "narrow_contract": {
        "enable_count_hurdle_head": True,
        "enable_count_jump": True,
        "count_jump_strength": 0.30,
        "enable_count_sparsity_gate": True,
        "count_sparsity_gate_strength": 0.75,
    },
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-json", type=Path, default=None, help="Optional path to write the JSON report.")
    return ap.parse_args()


def _run_variant(case: SliceCase, variant_name: str, kwargs: Dict[str, Any], frames) -> Dict[str, Any]:
    train, val, test = frames
    X_train, y_train = _prepare_features(train, case.target)
    X_test, y_test = _prepare_features(test, case.target)

    model = SingleModelMainlineWrapper(seed=7, **kwargs)
    model.fit(
        X_train,
        y_train,
        train_raw=train,
        val_raw=val,
        target=case.target,
        task=case.task,
        ablation=case.ablation,
        horizon=case.horizon,
    )
    preds = np.asarray(
        model.predict(
            X_test,
            test_raw=test,
            target=case.target,
            task=case.task,
            ablation=case.ablation,
            horizon=case.horizon,
        ),
        dtype=np.float64,
    )
    y_true = y_test.to_numpy(dtype=np.float64)
    regime = model.get_regime_info()
    return {
        "variant": variant_name,
        "model_kwargs": kwargs,
        "metrics": _compute_metrics(y_true, preds),
        "pred_mean": float(np.mean(preds)),
        "target_mean": float(np.mean(y_true)),
        "investors_process_contract": regime.get("investors_process_contract", {}),
        "investors_source_activation": regime.get("investors_source_activation", {}),
    }


def _delta_vs_baseline(case_report: Dict[str, Any]) -> Dict[str, Any]:
    baseline = case_report["variants"]["legacy_baseline"]["metrics"]["mae"]
    out: Dict[str, Any] = {}
    for name, payload in case_report["variants"].items():
        mae = payload["metrics"]["mae"]
        delta = float(baseline - mae)
        delta_pct = float(100.0 * delta / baseline) if baseline else 0.0
        out[name] = {
            "mae_delta": delta,
            "mae_delta_pct": delta_pct,
        }
    return out


def main() -> int:
    args = _parse_args()
    temporal_config = _make_temporal_config()

    report: Dict[str, Any] = {
        "cases": {},
        "variants": list(VARIANTS.keys()),
    }
    for case in DEFAULT_CASES:
        frames = _load_case_frames(case, temporal_config)
        case_report: Dict[str, Any] = {
            "case": asdict(case),
            "variants": {},
        }
        for variant_name, kwargs in VARIANTS.items():
            case_report["variants"][variant_name] = _run_variant(case, variant_name, kwargs, frames)
        case_report["delta_vs_baseline"] = _delta_vs_baseline(case_report)
        report["cases"][case.name] = case_report

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())