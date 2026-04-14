from types import SimpleNamespace

from scripts.analyze_mainline_funding_postfix_panel import _aggregate_group, _classify_against_baseline, _select_panel_cases


def test_select_panel_cases_hardest_family_filters_source_rich_long_horizon():
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
            "name": "task1_outcome__core_edgar__funding_raised_usd__h14",
            "task": "task1_outcome",
            "ablation": "core_edgar",
            "target": "funding_raised_usd",
            "horizon": 14,
        },
        {
            "name": "task1_outcome__full__funding_raised_usd__h30",
            "task": "task1_outcome",
            "ablation": "full",
            "target": "funding_raised_usd",
            "horizon": 30,
        },
        {
            "name": "task1_outcome__core_text__funding_raised_usd__h30",
            "task": "task1_outcome",
            "ablation": "core_text",
            "target": "funding_raised_usd",
            "horizon": 30,
        },
        {
            "name": "task1_outcome__core_edgar__funding_raised_usd__h7",
            "task": "task1_outcome",
            "ablation": "core_edgar",
            "target": "funding_raised_usd",
            "horizon": 7,
        },
        {
            "name": "task1_outcome__full__investors_count__h30",
            "task": "task1_outcome",
            "ablation": "full",
            "target": "investors_count",
            "horizon": 30,
        },
    ]

    selected = _select_panel_cases(cells, args)

    assert [case["name"] for case in selected] == [
        "task1_outcome__core_edgar__funding_raised_usd__h14",
        "task1_outcome__full__funding_raised_usd__h30",
    ]


def test_classify_against_baseline_uses_relative_tolerance():
    assert _classify_against_baseline(100.4, 100.0, 0.5) == "tie"
    assert _classify_against_baseline(99.0, 100.0, 0.5) == "better"
    assert _classify_against_baseline(101.0, 100.0, 0.5) == "worse"
    assert _classify_against_baseline(None, 100.0, 0.5) == "incomplete"


def test_aggregate_group_counts_no_leak_and_anchor_fallback():
    reports = [
        {
            "case": {"task": "task1_outcome", "ablation": "full", "horizon": 30},
            "mainline": {
                "runtime_contract_ok": True,
                "funding_process_contract": {"lane_residual_blend": 0.0},
            },
            "comparisons": {
                "vs_anchor": {"outcome": "worse", "gap_pct": 12.0},
                "vs_incumbent": {"outcome": "worse", "gap_pct": 25.0},
            },
        },
        {
            "case": {"task": "task2_forecast", "ablation": "core_edgar", "horizon": 14},
            "mainline": {
                "runtime_contract_ok": True,
                "funding_process_contract": {"lane_residual_blend": 0.25},
            },
            "comparisons": {
                "vs_anchor": {"outcome": "better", "gap_pct": -3.0},
                "vs_incumbent": {"outcome": "tie", "gap_pct": 0.2},
            },
        },
    ]

    summary = _aggregate_group(reports)

    assert summary["cases"] == 2
    assert summary["no_leak_runtime_pass"] == 2
    assert summary["anchor_fallback_cases"] == 1
    assert summary["anchor_outcomes"] == {"worse": 1, "better": 1}
    assert summary["incumbent_outcomes"] == {"worse": 1, "tie": 1}
    assert summary["mean_gap_vs_anchor_pct"] == 4.5