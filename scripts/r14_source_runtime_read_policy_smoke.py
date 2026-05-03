#!/usr/bin/env python3
"""Smoke-audit runtime source read/no-read policy wiring."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper
from scripts.analyze_mainline_investors_multisource_surface import build_synthetic_surface


def _funding_frame(n_rows: int = 36) -> pd.DataFrame:
    day0 = pd.Timestamp("2024-01-01")
    rows = []
    for idx in range(n_rows):
        rows.append(
            {
                "entity_id": f"funding_entity_{idx // 6}",
                "crawled_date_day": day0 + pd.Timedelta(days=idx),
                "core_signal": float((idx % 7) - 3),
                "funding_raised_usd": float(1000.0 + 35.0 * idx),
                "is_funded": 1.0,
                "investors_count": float(3.0 + (idx % 5)),
                "last_total_amount_sold": float(2000.0 + 75.0 * idx),
                "edgar_has_filing": 1.0,
                "text_emb_0": float(0.1 + 0.02 * idx),
                "text_emb_1": float(0.2 + 0.01 * idx),
            }
        )
    return pd.DataFrame(rows)


def _fit_funding(ablation: str) -> dict[str, Any]:
    frame = _funding_frame()
    feature_cols = ["core_signal", "last_total_amount_sold", "edgar_has_filing", "text_emb_0", "text_emb_1"]
    model = SingleModelMainlineWrapper(seed=17, enable_funding_source_scaling=True)
    y = frame["funding_raised_usd"].copy()
    y.name = "funding_raised_usd"
    model.fit(
        frame[feature_cols].copy(),
        y,
        train_raw=frame,
        target="funding_raised_usd",
        task="task1_outcome",
        ablation=ablation,
        horizon=30,
    )
    return model.get_regime_info()["funding_process_contract"]


def _investors_read_policy_summary() -> dict[str, Any]:
    train, test = build_synthetic_surface()
    x_train = train[["core_signal", "core_trend", "core_interaction"]].copy()
    y_train = train["investors_count"].copy()
    y_train.name = "investors_count"
    x_test = test[["core_signal", "core_trend", "core_interaction"]].copy()
    y_test = test["investors_count"].to_numpy(dtype=np.float64)

    def _mae(**kwargs: Any) -> tuple[float, dict[str, Any]]:
        model = SingleModelMainlineWrapper(seed=7, **kwargs)
        model.fit(
            x_train,
            y_train,
            train_raw=train,
            target="investors_count",
            task="task2_forecast",
            ablation="full",
            horizon=1,
        )
        pred = model.predict(
            x_test,
            test_raw=test,
            target="investors_count",
            task="task2_forecast",
            ablation="full",
            horizon=1,
        )
        return float(np.mean(np.abs(pred - y_test))), model.get_regime_info()["investors_source_activation"]

    baseline_mae, _ = _mae()
    read_policy_mae, regime = _mae(enable_investors_source_read_policy=True)
    learned = {str(k): float(v) for k, v in regime.get("learned_profile_anchor_blends", {}).items()}
    return {
        "baseline_mae": baseline_mae,
        "read_policy_mae": read_policy_mae,
        "read_policy_not_worse_than_baseline": bool(read_policy_mae <= baseline_mae + 1e-9),
        "effective_source_read_policy": bool(regime.get("effective_source_read_policy")),
        "activation_reason": regime.get("activation_reason"),
        "learned_profile_anchor_blends": learned,
    }


def main() -> int:
    core_only = _fit_funding("core_only")
    full = _fit_funding("full")
    investors = _investors_read_policy_summary()
    checks = {
        "core_only_no_read_even_with_raw_source": bool(
            core_only.get("train_source_scale_max") == 0.0
            and core_only.get("train_funding_read_confidence_max") == 0.0
            and core_only.get("train_funding_no_read_rate") == 1.0
        ),
        "full_source_confidence_available": bool(
            float(full.get("train_funding_read_confidence_max") or 0.0) > 0.0
            and float(full.get("train_source_scale_max") or 0.0) > 0.0
        ),
        "investors_read_policy_runtime_effective": bool(
            investors.get("effective_source_read_policy")
            and investors.get("read_policy_not_worse_than_baseline")
        ),
    }
    report = {
        "timestamp_cest": datetime.now().isoformat(),
        "status": "passed" if all(checks.values()) else "not_passed",
        "scope": "runtime source read/no-read wiring smoke",
        "checks": checks,
        "funding_core_only": core_only,
        "funding_full": full,
        "investors_read_policy": investors,
        "interpretation": [
            "This smoke verifies runtime wiring only; it does not promote source-read point-forecast claims on landed benchmark artifacts.",
            "Formal promotion still requires temporal reruns and the strict row-key counterfactual read-gate audit to pass.",
        ],
    }
    out_json = ROOT / "runs" / "audits" / f"r14_source_runtime_read_policy_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    out_json.with_suffix(".md").write_text(
        "# R14 Source Runtime Read Policy Smoke\n\n```json\n"
        + json.dumps(report, indent=2, ensure_ascii=False, default=str)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "checks": checks, "out_json": str(out_json)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())