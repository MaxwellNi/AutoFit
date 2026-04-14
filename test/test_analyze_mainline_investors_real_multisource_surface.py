from scripts.analyze_mainline_investors_real_multisource_surface import _rank_dynamic_entities


def test_rank_dynamic_entities_filters_static_and_requires_cik() -> None:
    stats = {
        "dynamic_cik_high": {
            "train_count": 12,
            "test_count": 6,
            "test_values": {1.0, 2.0},
            "has_cik": True,
        },
        "dynamic_no_cik": {
            "train_count": 30,
            "test_count": 9,
            "test_values": {1.0, 2.0, 3.0},
            "has_cik": False,
        },
        "static_cik": {
            "train_count": 40,
            "test_count": 10,
            "test_values": {7.0},
            "has_cik": True,
        },
        "missing_test": {
            "train_count": 8,
            "test_count": 0,
            "test_values": {1.0, 2.0},
            "has_cik": True,
        },
        "dynamic_cik_low": {
            "train_count": 5,
            "test_count": 5,
            "test_values": {3.0, 4.0, 5.0},
            "has_cik": True,
        },
    }

    ranked = _rank_dynamic_entities(stats, limit=10, require_cik=True)

    assert [item["entity_id"] for item in ranked] == ["dynamic_cik_high", "dynamic_cik_low"]
    assert ranked[0]["test_unique_values"] == 2
    assert ranked[1]["test_unique_values"] == 3


def test_rank_dynamic_entities_can_include_non_cik_entities() -> None:
    stats = {
        "dynamic_no_cik": {
            "train_count": 30,
            "test_count": 9,
            "test_values": {1.0, 2.0, 3.0},
            "has_cik": False,
        },
        "dynamic_cik": {
            "train_count": 12,
            "test_count": 6,
            "test_values": {1.0, 2.0},
            "has_cik": True,
        },
    }

    ranked = _rank_dynamic_entities(stats, limit=10, require_cik=False)

    assert [item["entity_id"] for item in ranked] == ["dynamic_no_cik", "dynamic_cik"]