import pandas as pd

from narrative.data_preprocessing.external_datasets import normalize_kickstarter, normalize_kiva


def test_normalize_kickstarter():
    df = pd.DataFrame(
        {
            "ID": [1],
            "launched": ["2020-01-01"],
            "deadline": ["2020-02-01"],
            "goal": [1000],
            "pledged": [1200],
            "backers": [10],
            "category": ["games"],
            "country": ["US"],
            "state": ["successful"],
            "name": ["test"],
        }
    )
    out = normalize_kickstarter(df)
    assert "project_id" in out.columns
    assert out["platform"].iloc[0] == "kickstarter"


def test_normalize_kiva():
    df = pd.DataFrame(
        {
            "id": [1],
            "posted_time": ["2020-01-01"],
            "loan_amount": [500],
            "funded_amount": [500],
            "sector": ["agriculture"],
            "country": ["KE"],
        }
    )
    out = normalize_kiva(df)
    assert out["platform"].iloc[0] == "kiva"
