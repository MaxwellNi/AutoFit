import pytest
import pandas as pd

from scripts.analyze_investors_benchmark_geometry import _dominant_profile, _dynamic_entity_stats, _target_stats


def test_dominant_profile_returns_name_and_share() -> None:
    report = _dominant_profile({"none": 2, "mixed": 6, "text_only": 2})

    assert report == {"name": "mixed", "share": pytest.approx(0.6)}


def test_target_stats_handles_dense_numeric_target() -> None:
    frame = pd.DataFrame({"investors_count": [0.0, 1.0, 2.0, 8.0, 9.0]})

    stats = _target_stats(frame, "investors_count")

    assert stats["rows"] == 5
    assert stats["mean"] == pytest.approx(4.0)
    assert stats["p50"] == pytest.approx(2.0)
    assert stats["zero_rate"] == pytest.approx(0.2)


def test_dynamic_entity_stats_separates_static_and_dynamic_entities() -> None:
    frame = pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "b", "b", "c", "c"],
            "investors_count": [0.0, 1.0, 3.0, 2.0, 2.0, 5.0, 9.0],
        }
    )

    stats = _dynamic_entity_stats(frame, "investors_count")

    assert stats["entities"] == 3
    assert stats["dynamic_entities"] == 2
    assert stats["dynamic_entity_ratio"] == pytest.approx(2.0 / 3.0)
    assert stats["dynamic_rows"] == 5
    assert stats["range_max"] == pytest.approx(4.0)