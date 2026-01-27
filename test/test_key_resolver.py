import pytest
import pandas as pd

from narrative.data_preprocessing.key_resolver import (
    resolve_entity_id,
    build_crosswalk,
    validate_unique_keys,
)


def test_resolve_entity_id_prefers_cik_then_offer_key():
    df = pd.DataFrame(
        {
            "platform_name": ["p1", "p2"],
            "offer_id": ["o1", "o2"],
            "cik": ["123", ""],
        }
    )
    entity = resolve_entity_id(df)
    assert entity.tolist()[0] == "123"
    assert entity.tolist()[1] == "p2||o2"


def test_build_crosswalk_includes_cik():
    offers = pd.DataFrame(
        {
            "platform_name": ["p1", "p1"],
            "offer_id": ["o1", "o2"],
            "cik": ["123", "456"],
        }
    )
    edgar = pd.DataFrame({"cik": ["123", "789"]})
    cross = build_crosswalk(offers, edgar)
    assert "entity_id" in cross.columns
    assert cross["cik"].astype(str).tolist() == ["123", "456"]


def test_validate_unique_keys_counts_duplicates():
    df = pd.DataFrame({"platform_name": ["p", "p"], "offer_id": ["1", "1"]})
    chk = validate_unique_keys(df, ["platform_name", "offer_id"])
    assert chk.n_duplicates == 1
