import pytest
import pandas as pd

from scripts.analyze_shared_business_evolution_geometry import _final_state_relations, _goal_alignment, _process_geometry


def test_final_state_relations_capture_funding_investor_asymmetry() -> None:
    final_rows = pd.DataFrame(
        {
            "funding_raised_usd": [100.0, 50.0, 0.0, 200.0],
            "funding_goal_usd": [100.0, 200.0, 100.0, 150.0],
            "investors_count": [3.0, 0.0, 0.0, 2.0],
            "is_funded": [1.0, 1.0, 0.0, 1.0],
        }
    )

    report = _final_state_relations(final_rows)

    assert report["conditionals"]["p_funding_pos_given_investors_pos"] == pytest.approx(1.0)
    assert report["conditionals"]["p_investors_pos_given_funding_pos"] == pytest.approx(2.0 / 3.0)
    assert report["conditionals"]["p_funded_given_funding_pos"] == pytest.approx(1.0)


def test_goal_alignment_summarizes_goal_ratio_conditionals() -> None:
    final_rows = pd.DataFrame(
        {
            "funding_raised_usd": [120.0, 40.0, 160.0],
            "funding_goal_usd": [100.0, 100.0, 200.0],
            "is_funded": [1.0, 0.0, 1.0],
        }
    )

    report = _goal_alignment(final_rows)

    assert report["rows"] == 3
    assert report["conditionals"]["p_funded_given_ratio_ge_1_0"] == pytest.approx(1.0)
    assert report["conditionals"]["p_ratio_ge_0_5_given_funded"] == pytest.approx(1.0)
    assert report["goal_ratio"]["median_funded"] == pytest.approx(1.0)


def test_process_geometry_reports_increment_coupling_and_absorbing_status() -> None:
    panel = pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "b", "b", "b"],
            "funding_raised_usd": [0.0, 100.0, 100.0, 0.0, 50.0, 70.0],
            "investors_count": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "is_funded": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        }
    )

    report = _process_geometry(panel)

    assert report["monotonicity"]["funding_monotone_share"] == pytest.approx(1.0)
    assert report["monotonicity"]["funded_absorbing_share"] == pytest.approx(1.0)
    assert report["increment_coupling"]["p_funding_jump_given_investor_jump"] == pytest.approx(1.0)
    assert report["increment_coupling"]["p_investor_jump_given_funding_jump"] == pytest.approx(1.0 / 3.0)
    assert report["increment_coupling"]["contingency"] == {
        "neither": 1,
        "funding_only": 2,
        "investor_only": 0,
        "both": 1,
    }