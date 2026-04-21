import numpy as np
import pandas as pd

from narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper
from narrative.block3.models.single_model_mainline.investor_mark_encoder import InvestorMarkEncoder
from narrative.block3.models.single_model_mainline.lanes.investors_lane import InvestorsLaneRuntime


def test_investor_mark_encoder_separates_institutional_like_from_retail_like_rows():
    frame = pd.DataFrame(
        {
            "entity_id": ["e0", "e0"],
            "crawled_date_day": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "investors__json": [
                '["Acme Ventures", "Blue Capital"]',
                '["Retail Crowd", "Community Backers"]',
            ],
            "investors__len": [2, 2],
            "investor_website": ["https://acmeventures.com", ""],
            "investment_type": ["equity", "crowdfunding"],
            "non_national_investors": [1, 0],
            "last_number_non_accredited_investors": [0, 10],
            "last_total_number_already_invested": [5, 12],
            "last_minimum_investment_accepted": [25000, 100],
            "last_total_offering_amount": [500000, 100000],
            "last_total_amount_sold": [300000, 5000],
            "last_total_remaining": [200000, 95000],
        }
    )

    encoder = InvestorMarkEncoder()
    marks = encoder.build_mark_frame(frame)

    assert marks.loc[0, "mark_institutional_like_score"] > marks.loc[1, "mark_institutional_like_score"]
    assert marks.loc[1, "mark_retail_like_score"] > marks.loc[0, "mark_retail_like_score"]
    assert marks.loc[0, "mark_large_investor_event_score"] > marks.loc[1, "mark_large_investor_event_score"]
    assert marks.loc[1, "mark_list_changed_flag"] == 1.0


def test_investors_lane_can_use_mark_features_to_separate_two_regimes():
    runtime = InvestorsLaneRuntime(random_state=5)

    n_rows = 64
    lane_state = np.zeros((n_rows, 4), dtype=np.float32)
    aux = np.zeros((n_rows, 5), dtype=np.float32)
    anchor = np.full(n_rows, 2.0, dtype=np.float64)
    source = np.zeros((n_rows, 7), dtype=np.float32)
    marks = np.zeros((n_rows, 4), dtype=np.float32)
    target = np.full(n_rows, 2.0, dtype=np.float64)

    marks[:32, 0] = 1.0
    marks[:32, 1] = 0.9
    marks[:32, 2] = 0.8
    target[:32] = np.linspace(6.5, 7.5, 32)

    marks[32:, 0] = 0.1
    marks[32:, 1] = 0.2
    marks[32:, 2] = 0.1
    target[32:] = np.linspace(2.0, 3.0, 32)

    runtime.fit(
        lane_state,
        target,
        aux_features=aux,
        anchor=anchor,
        source_features=source,
        mark_features=marks,
        enable_mark_features=True,
        horizon=7,
        task_name="task2_forecast",
    )

    preds = runtime.predict(
        lane_state,
        aux_features=aux,
        anchor=anchor,
        source_features=source,
        mark_features=marks,
        enable_mark_features=True,
    )

    assert preds[:32].mean() > preds[32:].mean()


def test_wrapper_surfaces_mark_activation_regime_when_requested():
    frame = pd.DataFrame(
        {
            "entity_id": [f"entity_{idx // 4}" for idx in range(24)],
            "crawled_date_day": pd.date_range("2024-01-01", periods=24, freq="D"),
            "core_signal": np.linspace(-1.0, 1.0, 24),
            "funding_raised_usd": np.linspace(1000.0, 5000.0, 24),
            "is_funded": np.ones(24),
            "investors_count": np.concatenate([np.full(12, 7.0), np.full(12, 2.0)]),
            "investors__json": ['["Acme Ventures"]'] * 12 + ['["Retail Crowd"]'] * 12,
            "investors__len": [1] * 24,
            "investor_website": ["https://acmeventures.com"] * 12 + [""] * 12,
            "investment_type": ["equity"] * 12 + ["crowdfunding"] * 12,
            "last_minimum_investment_accepted": [25000.0] * 12 + [100.0] * 12,
            "last_total_amount_sold": [200000.0] * 12 + [5000.0] * 12,
            "last_total_offering_amount": [500000.0] * 24,
            "last_total_remaining": [300000.0] * 12 + [495000.0] * 12,
            "last_total_number_already_invested": [5.0] * 12 + [40.0] * 12,
            "last_number_non_accredited_investors": [0.0] * 12 + [35.0] * 12,
        }
    )

    wrapper = SingleModelMainlineWrapper(seed=17, enable_investors_mark_features=True)
    X = frame[["core_signal"]].copy()
    y = frame["investors_count"].copy()
    y.name = "investors_count"

    wrapper.fit(
        X,
        y,
        train_raw=frame,
        target="investors_count",
        task="task2_forecast",
        ablation="core_only",
        horizon=7,
    )

    mark_regime = wrapper.get_regime_info()["investor_mark_activation"]

    assert mark_regime["requested_mark_features"] is True
    assert mark_regime["effective_mark_features"] is True
    assert mark_regime["mark_mode"] == "hybrid"
    assert mark_regime["mark_feature_count"] > 0
    assert mark_regime["mark_coverage_share"] > 0.0
    assert mark_regime["raw_reference_mark_share"] > 0.0
    assert mark_regime["proxy_mark_share"] > 0.0


def test_wrapper_keeps_mark_features_active_on_proxy_only_investor_surface():
    frame = pd.DataFrame(
        {
            "entity_id": [f"entity_{idx // 4}" for idx in range(24)],
            "crawled_date_day": pd.date_range("2024-02-01", periods=24, freq="D"),
            "core_signal": np.linspace(-1.5, 1.5, 24),
            "funding_raised_usd": np.linspace(500.0, 4500.0, 24),
            "is_funded": np.ones(24),
            "investors_count": np.concatenate([np.full(12, 6.0), np.full(12, 2.0)]),
            "investors__json": [""] * 24,
            "investors__len": [0] * 24,
            "investor_website": [""] * 24,
            "investment_type": ["equity"] * 12 + ["crowdfunding"] * 12,
            "non_national_investors": [1.0] * 12 + [0.0] * 12,
            "last_minimum_investment_accepted": [20000.0] * 12 + [100.0] * 12,
            "last_total_amount_sold": [150000.0] * 12 + [3000.0] * 12,
            "last_total_offering_amount": [400000.0] * 24,
            "last_total_remaining": [250000.0] * 12 + [397000.0] * 12,
            "last_total_number_already_invested": [4.0] * 12 + [35.0] * 12,
            "last_number_non_accredited_investors": [0.0] * 12 + [30.0] * 12,
        }
    )

    wrapper = SingleModelMainlineWrapper(seed=23, enable_investors_mark_features=True)
    X = frame[["core_signal"]].copy()
    y = frame["investors_count"].copy()
    y.name = "investors_count"

    wrapper.fit(
        X,
        y,
        train_raw=frame,
        target="investors_count",
        task="task2_forecast",
        ablation="core_edgar",
        horizon=7,
    )

    mark_regime = wrapper.get_regime_info()["investor_mark_activation"]

    assert mark_regime["requested_mark_features"] is True
    assert mark_regime["effective_mark_features"] is True
    assert mark_regime["activation_reason"] == "mark_features_proxy_only"
    assert mark_regime["mark_mode"] == "proxy_only"
    assert mark_regime["mark_coverage_share"] > 0.0
    assert mark_regime["raw_reference_mark_share"] == 0.0
    assert mark_regime["proxy_mark_share"] > 0.0
    assert mark_regime["proxy_only_mark_share"] > 0.0
    assert mark_regime["mark_summary"]["mark_concentrated_capital_score"] > 0.0
    assert mark_regime["mark_summary"]["mark_institutional_like_score"] > 0.0