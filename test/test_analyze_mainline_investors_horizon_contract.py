from scripts.analyze_mainline_investors_horizon_contract import _delta_vs_baseline


def test_delta_vs_baseline_reports_absolute_and_percent_improvement() -> None:
    case_report = {
        "variants": {
            "legacy_baseline": {"metrics": {"mae": 10.0}},
            "hurdle_only": {"metrics": {"mae": 9.5}},
            "hurdle_plus_jump": {"metrics": {"mae": 8.0}},
            "hurdle_plus_sparsity": {"metrics": {"mae": 10.5}},
            "narrow_contract": {"metrics": {"mae": 9.0}},
        }
    }

    report = _delta_vs_baseline(case_report)

    assert report["legacy_baseline"] == {"mae_delta": 0.0, "mae_delta_pct": 0.0}
    assert report["hurdle_only"]["mae_delta"] == 0.5
    assert report["hurdle_only"]["mae_delta_pct"] == 5.0
    assert report["hurdle_plus_jump"]["mae_delta"] == 2.0
    assert report["hurdle_plus_jump"]["mae_delta_pct"] == 20.0
    assert report["hurdle_plus_sparsity"]["mae_delta"] == -0.5
    assert report["hurdle_plus_sparsity"]["mae_delta_pct"] == -5.0