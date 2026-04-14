from types import SimpleNamespace

from scripts.analyze_mainline_binary_postfix_panel import _aggregate_group, _classify_against_baseline, _select_panel_cases


def test_select_panel_cases_hardest_family_filters_source_rich_long_horizon_binary_cells():
    args = SimpleNamespace(
        panel="hardest_family",
        task="",
        ablation="",
        horizon=0,
        case_substr="",
        max_cases=0,
    )
    cells = [
        {
            "name": "task1_outcome__core_edgar__is_funded__h14",
            "task": "task1_outcome",
            "ablation": "core_edgar",
            "target": "is_funded",
            "horizon": 14,
        },
        {
            "name": "task1_outcome__full__is_funded__h30",
            "task": "task1_outcome",
            "ablation": "full",
            "target": "is_funded",
            "horizon": 30,
        },
        {
            "name": "task1_outcome__core_text__is_funded__h30",
            "task": "task1_outcome",
            "ablation": "core_text",
            "target": "is_funded",
            "horizon": 30,
        },
        {
            "name": "task1_outcome__core_edgar__is_funded__h7",
            "task": "task1_outcome",
            "ablation": "core_edgar",
            "target": "is_funded",
            "horizon": 7,
        },
        {
            "name": "task1_outcome__full__funding_raised_usd__h30",
            "task": "task1_outcome",
            "ablation": "full",
            "target": "funding_raised_usd",
            "horizon": 30,
        },
    ]

    selected = _select_panel_cases(cells, args)

    assert [case["name"] for case in selected] == [
        "task1_outcome__core_edgar__is_funded__h14",
        "task1_outcome__full__is_funded__h30",
    ]


def test_classify_against_baseline_respects_metric_direction():
    assert _classify_against_baseline(0.08, 0.10, 0.5, lower_is_better=True) == "better"
    assert _classify_against_baseline(0.1004, 0.10, 0.5, lower_is_better=True) == "tie"
    assert _classify_against_baseline(0.78, 0.80, 0.5, lower_is_better=False) == "worse"
    assert _classify_against_baseline(0.804, 0.80, 0.5, lower_is_better=False) == "tie"


def test_aggregate_group_counts_collapse_and_calibration_issues():
    reports = [
        {
            "case": {"task": "task1_outcome", "ablation": "full", "horizon": 30},
            "mainline": {
                "runtime_contract_ok": True,
                "probability_collapse": True,
                "binary_process_contract": {
                    "uses_hazard_adapter": True,
                    "calibration_method": "platt",
                },
            },
            "calibration_issue": True,
            "comparisons": {
                "vs_incumbent": {
                    "mae": {"outcome": "worse", "gap_pct": 20.0},
                    "brier": {"outcome": "worse", "gap_pct": 50.0},
                    "logloss": {"outcome": "worse", "gap_pct": 40.0},
                    "ece": {"outcome": "worse", "gap_pct": 100.0},
                    "auc": {"outcome": "worse", "gap_pct": 10.0},
                    "prauc": {"outcome": "worse", "gap_pct": 25.0},
                }
            },
        },
        {
            "case": {"task": "task2_forecast", "ablation": "core_edgar", "horizon": 14},
            "mainline": {
                "runtime_contract_ok": True,
                "probability_collapse": False,
                "binary_process_contract": {
                    "uses_hazard_adapter": False,
                    "calibration_method": "identity",
                },
            },
            "calibration_issue": False,
            "comparisons": {
                "vs_incumbent": {
                    "mae": {"outcome": "better", "gap_pct": -5.0},
                    "brier": {"outcome": "better", "gap_pct": -10.0},
                    "logloss": {"outcome": "better", "gap_pct": -8.0},
                    "ece": {"outcome": "better", "gap_pct": -20.0},
                    "auc": {"outcome": "better", "gap_pct": -4.0},
                    "prauc": {"outcome": "better", "gap_pct": -6.0},
                }
            },
        },
    ]

    summary = _aggregate_group(reports)

    assert summary["cases"] == 2
    assert summary["no_leak_runtime_pass"] == 2
    assert summary["probability_collapse_cases"] == 1
    assert summary["severe_calibration_cases"] == 1
    assert summary["hazard_adapter_active_cases"] == 1
    assert summary["calibration_method_counts"] == {"platt": 1, "identity": 1}
    assert summary["metric_outcomes"]["brier"] == {"worse": 1, "better": 1}
    assert summary["median_gap_pct_by_metric"]["brier"] == 20.0
    assert summary["median_gap_pct_by_metric"]["auc"] == 3.0