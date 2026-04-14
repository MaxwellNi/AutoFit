#!/usr/bin/env python3
"""Research-only synthetic multi-profile surface for investors source-aware studies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper
from scripts.run_v740_alpha_smoke_slice import _compute_metrics


PROFILE_ORDER = ("none", "edgar_only", "text_only", "mixed")
PROFILE_TO_ID = {name: idx for idx, name in enumerate(PROFILE_ORDER)}
VARIANTS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "enable_count_hurdle_head": False,
        "enable_count_jump": False,
        "enable_count_sparsity_gate": False,
    },
    "forced_source_features": {
        "enable_investors_source_features": True,
        "enable_investors_selective_source_activation": False,
    },
    "forced_source_features_plus_guard": {
        "enable_investors_source_features": True,
        "enable_investors_source_guard": True,
        "enable_investors_selective_source_activation": False,
    },
    "selective_source_read_policy": {
        "enable_investors_source_read_policy": True,
    },
    "selective_source_read_policy_plus_transition": {
        "enable_investors_source_read_policy": True,
        "enable_investors_transition_correction": True,
    },
    "selective_source_read_policy_plus_transition_plus_guard": {
        "enable_investors_source_read_policy": True,
        "enable_investors_transition_correction": True,
        "enable_investors_source_guard": True,
    },
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-json", type=Path, default=None, help="Optional path to save the JSON report.")
    return ap.parse_args()


def _profile_params(profile: str) -> Dict[str, float]:
    params = {
        "none": {"anchor_coef": 0.15, "signal_coef": 7.5, "level": 16.0, "seasonal_coef": 2.6},
        "edgar_only": {"anchor_coef": 0.94, "signal_coef": 0.8, "level": 20.0, "seasonal_coef": 0.4},
        "text_only": {"anchor_coef": 0.22, "signal_coef": 7.0, "level": 14.0, "seasonal_coef": 3.2},
        "mixed": {"anchor_coef": 0.72, "signal_coef": 3.5, "level": 18.0, "seasonal_coef": 1.2},
    }
    return params[profile]


def build_synthetic_surface(
    entities_per_profile: int = 6,
    steps_per_entity: int = 26,
    train_steps: int = 18,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    start_day = pd.Timestamp("2024-01-01")
    for profile in PROFILE_ORDER:
        params = _profile_params(profile)
        for entity_idx in range(entities_per_profile):
            entity_id = f"{profile}_entity_{entity_idx}"
            prev_target = params["level"] + 0.7 * entity_idx
            for step in range(steps_per_entity):
                day = start_day + pd.Timedelta(days=step)
                signal = np.sin((step + entity_idx) / 2.7) + 0.15 * entity_idx
                trend = float(step) / max(steps_per_entity - 1, 1)
                seasonal = np.cos((step + 2 * entity_idx) / 3.3)
                target = (
                    params["level"]
                    + params["anchor_coef"] * (prev_target - params["level"])
                    + params["signal_coef"] * signal
                    + params["seasonal_coef"] * seasonal
                )
                target = float(max(target, 0.5))
                prev_target = target

                row = {
                    "entity_id": entity_id,
                    "crawled_date_day": day,
                    "core_signal": float(signal),
                    "core_trend": float(trend),
                    "core_interaction": float(signal * (1.0 + trend)),
                    "funding_raised_usd": float(50000.0 + 2500.0 * target),
                    "is_funded": 1.0,
                    "investors_count": target,
                    "source_profile": profile,
                    "last_total_amount_sold": np.nan,
                    "edgar_has_filing": np.nan,
                    "text_emb_0": np.nan,
                    "text_emb_1": np.nan,
                }
                if profile in {"edgar_only", "mixed"}:
                    row["last_total_amount_sold"] = float(5000.0 + 90.0 * step + 30.0 * entity_idx)
                    row["edgar_has_filing"] = 1.0
                if profile in {"text_only", "mixed"}:
                    row["text_emb_0"] = float(0.4 * signal + 0.1 * entity_idx)
                    row["text_emb_1"] = float(0.3 * seasonal + 0.05 * step)
                rows.append(row)

    frame = pd.DataFrame(rows)
    train_mask = frame.groupby("entity_id", sort=False).cumcount() < train_steps
    train = frame.loc[train_mask].reset_index(drop=True)
    test = frame.loc[~train_mask].reset_index(drop=True)
    return train, test


def _prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = frame[["core_signal", "core_trend", "core_interaction"]].copy()
    y = frame["investors_count"].copy()
    y.name = "investors_count"
    return X, y


def _profile_breakdown(frame: pd.DataFrame, y_true: np.ndarray, preds: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for profile in PROFILE_ORDER:
        mask = frame["source_profile"].to_numpy() == profile
        if not mask.any():
            continue
        out[profile] = {
            "rows": int(mask.sum()),
            "metrics": _compute_metrics(y_true[mask], preds[mask]),
            "pred_mean": float(np.mean(preds[mask])),
            "target_mean": float(np.mean(y_true[mask])),
        }
    return out


def _run_variant(name: str, kwargs: Dict[str, Any], train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Any]:
    X_train, y_train = _prepare_xy(train)
    X_test, y_test = _prepare_xy(test)

    model = SingleModelMainlineWrapper(seed=7, **kwargs)
    model.fit(
        X_train,
        y_train,
        train_raw=train,
        target="investors_count",
        task="task2_forecast",
        ablation="full",
        horizon=1,
    )
    preds = np.asarray(
        model.predict(
            X_test,
            test_raw=test,
            target="investors_count",
            task="task2_forecast",
            ablation="full",
            horizon=1,
        ),
        dtype=np.float64,
    )
    y_true = y_test.to_numpy(dtype=np.float64)
    regime = model.get_regime_info()["investors_source_activation"]
    return {
        "variant": name,
        "model_kwargs": kwargs,
        "overall_metrics": _compute_metrics(y_true, preds),
        "overall_pred_mean": float(np.mean(preds)),
        "overall_target_mean": float(np.mean(y_true)),
        "profile_metrics": _profile_breakdown(test, y_true, preds),
        "investors_source_activation": regime,
    }


def main() -> int:
    args = _parse_args()
    train, test = build_synthetic_surface()
    report = {
        "surface": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_profile_counts": train["source_profile"].value_counts().sort_index().to_dict(),
            "test_profile_counts": test["source_profile"].value_counts().sort_index().to_dict(),
        },
        "variants": {},
    }
    for name, kwargs in VARIANTS.items():
        report["variants"][name] = _run_variant(name, kwargs, train, test)

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())