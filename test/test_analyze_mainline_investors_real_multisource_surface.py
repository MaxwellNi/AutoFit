import numpy as np
import pandas as pd

from scripts.analyze_mainline_investors_real_multisource_surface import (
    RealSurfaceCase,
    _fit_shared_encoder,
    _rank_dynamic_entities,
    _surface_aux_features,
)


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


def test_surface_aux_features_can_append_event_state_columns() -> None:
    surface = {
        "aux": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "event_state": np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
    }

    plain = _surface_aux_features(surface, enable_event_state_features=False)
    combined = _surface_aux_features(surface, enable_event_state_features=True)

    assert plain.shape == (2, 2)
    assert combined.shape == (2, 5)
    assert np.allclose(combined[:, :2], surface["aux"])
    assert np.allclose(combined[:, 2:], surface["event_state"])


def _make_dynamic_surface_frame(profile: str, n_rows: int = 18) -> pd.DataFrame:
    rows = []
    start_day = pd.Timestamp("2024-01-01")
    for idx in range(n_rows):
        row = {
            "entity_id": f"{profile}_entity_{idx // 3}",
            "crawled_date_day": start_day + pd.Timedelta(days=idx),
            "core_signal": float((idx % 5) - 2),
            "investors_count": float(12.0 + (idx % 4) + (idx // 6)),
            "funding_raised_usd": float(1200.0 + 40.0 * idx),
            "funding_goal_usd": float(2500.0 + 25.0 * idx),
            "is_funded": 1.0,
            "last_total_amount_sold": float(700.0 + 20.0 * idx),
            "last_total_offering_amount": float(3200.0 + 25.0 * idx),
            "last_total_remaining": float(max(0.0, 2500.0 - 12.0 * idx)),
            "last_minimum_investment_accepted": float(100.0 + 4.0 * idx),
            "last_total_number_already_invested": float(10.0 + (idx % 5)),
            "last_number_non_accredited_investors": float(idx % 3),
            "non_national_investors": float(idx % 2),
            "edgar_has_filing": np.nan,
            "text_emb_0": np.nan,
            "text_emb_1": np.nan,
        }
        if profile in {"core_edgar", "full"}:
            row["edgar_has_filing"] = 1.0
        if profile in {"core_text", "full"}:
            row["text_emb_0"] = float(0.05 * (idx + 1))
            row["text_emb_1"] = float(0.02 * ((idx % 3) + 1))
        rows.append(row)
    return pd.DataFrame(rows)


def _make_surface_payload(frame: pd.DataFrame) -> dict:
    split_idx = len(frame) // 2
    train = frame.iloc[:split_idx].copy()
    test = frame.iloc[split_idx:].copy()
    X_train = train[["core_signal"]].copy()
    X_test = test[["core_signal"]].copy()
    y_train = train["investors_count"].copy()
    y_train.name = "investors_count"
    y_test = test["investors_count"].copy()
    y_test.name = "investors_count"
    return {
        "train_raw": train,
        "test_raw": test,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def test_fit_shared_encoder_can_apply_process_state_feedback_backbone() -> None:
    case = RealSurfaceCase(horizon=14, ablations=("core_edgar", "full"))
    surfaces = {
        ablation: _make_surface_payload(_make_dynamic_surface_frame(ablation))
        for ablation in case.ablations
    }

    baseline_encoded, _ = _fit_shared_encoder(case, surfaces, backbone_kwargs=None)
    feedback_encoded, _ = _fit_shared_encoder(
        case,
        surfaces,
        backbone_kwargs={
            "enable_process_state_feedback": True,
            "process_state_feedback_strength": 0.12,
            "process_state_feedback_source_decay": 0.65,
            "process_state_feedback_min_horizon": 7,
            "process_state_feedback_state_weights": (0.30, 0.20, 0.0, 0.0, 0.50),
        },
    )

    baseline_trunk = baseline_encoded["full"]["event_state_trunk_train"]["shared_state_atoms"]
    feedback_trunk = feedback_encoded["full"]["event_state_trunk_train"]["shared_state_atoms"]
    feedback_process = feedback_encoded["full"]["event_state_trunk_train"]["process_state_atoms"]

    assert baseline_trunk["effective_process_state_feedback"] is False
    assert feedback_trunk["process_state_feedback_enabled"] is True
    assert feedback_trunk["effective_process_state_feedback"] is True
    assert feedback_trunk["shared_state_dim"] == baseline_trunk["shared_state_dim"]
    assert feedback_trunk["process_state_feedback_gate"] > 0.0
    assert feedback_trunk["process_feedback_closure_conversion_abs_mean"] > 0.0
    assert feedback_process["closure_conversion_score"] > 0.0
    assert feedback_process["closure_conversion_score"] != baseline_encoded["full"]["event_state_trunk_train"]["process_state_atoms"]["closure_conversion_score"]