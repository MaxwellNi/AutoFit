from scripts.build_shared112_scorecard import (
    _binary_calibration_rows,
    _build_cases,
    _dispersion_rows,
    _summarize_cases,
)


def test_build_cases_marks_calibration_and_dispersion_issues():
    manifest = {
        "cells": [
            {
                "name": "task1_outcome__core_only__is_funded__h1",
                "task": "task1_outcome",
                "ablation": "core_only",
                "target": "is_funded",
                "horizon": 1,
                "incumbent_model": "DeepNPTS",
                "incumbent_benchmark_mae": 0.03,
            },
            {
                "name": "task2_forecast__core_only__funding_raised_usd__h7",
                "task": "task2_forecast",
                "ablation": "core_only",
                "target": "funding_raised_usd",
                "horizon": 7,
                "incumbent_model": "NBEATS",
                "incumbent_benchmark_mae": 100.0,
            },
        ]
    }
    rows = [
        {
            "case_name": "task1_outcome__core_only__is_funded__h1",
            "model_label": "candidate",
            "metrics": {"mae": 0.050, "brier": 0.200, "logloss": 0.700, "ece": 0.160},
            "binary_prob_std": 0.001,
            "constant_prediction": False,
        },
        {
            "case_name": "task1_outcome__core_only__is_funded__h1",
            "model_label": "incumbent__DeepNPTS",
            "metrics": {"mae": 0.040, "brier": 0.080, "logloss": 0.300, "ece": 0.040},
            "binary_prob_std": 0.180,
            "constant_prediction": False,
        },
        {
            "case_name": "task2_forecast__core_only__funding_raised_usd__h7",
            "model_label": "candidate",
            "metrics": {"mae": 120.0},
            "prediction_to_truth_std_ratio": 4.6,
            "residual_to_truth_std_ratio": 2.8,
            "constant_prediction": False,
        },
        {
            "case_name": "task2_forecast__core_only__funding_raised_usd__h7",
            "model_label": "incumbent__NBEATS",
            "metrics": {"mae": 100.0},
            "prediction_to_truth_std_ratio": 1.1,
            "residual_to_truth_std_ratio": 1.2,
            "constant_prediction": False,
        },
    ]

    cases = _build_cases(manifest, rows, "candidate", tie_tol_pct=0.5, catastrophic_ratio=10.0)
    binary_case = next(case for case in cases if case["target"] == "is_funded")
    funding_case = next(case for case in cases if case["target"] == "funding_raised_usd")

    assert binary_case["binary_calibration_issue"] is True
    assert funding_case["dispersion_drift"] is True

    total = _summarize_cases(cases, lambda case: True)
    assert total["catastrophic"] == 2


def test_scorecard_audit_rows_surface_new_fields():
    cases = [
        {
            "task": "task1_outcome",
            "ablation": "core_only",
            "target": "is_funded",
            "horizon": 1,
            "candidate": {"metrics": {"brier": 0.21, "logloss": 0.72, "ece": 0.17}, "binary_prob_std": 0.001},
            "incumbent": {"metrics": {"brier": 0.08, "logloss": 0.30, "ece": 0.04}, "binary_prob_std": 0.18},
            "binary_calibration_issue": True,
            "dispersion_drift": False,
            "blow_up": False,
            "constant_prediction": False,
            "error": None,
            "candidate_mae": 0.05,
            "incumbent_local_mae": 0.04,
            "gap_pct": 25.0,
            "name": "task1_outcome__core_only__is_funded__h1",
        },
        {
            "task": "task2_forecast",
            "ablation": "core_only",
            "target": "funding_raised_usd",
            "horizon": 7,
            "candidate": {"prediction_to_truth_std_ratio": 4.6, "residual_to_truth_std_ratio": 2.8},
            "incumbent": {"prediction_to_truth_std_ratio": 1.1, "residual_to_truth_std_ratio": 1.2},
            "binary_calibration_issue": False,
            "dispersion_drift": True,
            "blow_up": False,
            "constant_prediction": False,
            "error": None,
            "candidate_mae": 120.0,
            "incumbent_local_mae": 100.0,
            "gap_pct": 20.0,
            "name": "task2_forecast__core_only__funding_raised_usd__h7",
        },
    ]

    calibration_rows = _binary_calibration_rows(cases)
    dispersion_rows = _dispersion_rows(cases)

    core_only_calibration = next(row for row in calibration_rows if row["ablation"] == "core_only")
    core_only_funding = next(
        row for row in dispersion_rows if row["ablation"] == "core_only" and row["target"] == "funding_raised_usd"
    )

    assert core_only_calibration["verdict"] == "drift"
    assert "med_delta=" in core_only_calibration["brier"]
    assert core_only_funding["verdict"] == "drift"
    assert "cand_med=" in core_only_funding["prediction_dispersion"]