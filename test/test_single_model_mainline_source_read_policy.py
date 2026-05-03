import numpy as np

from narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper
from scripts.analyze_mainline_investors_multisource_surface import build_synthetic_surface


def _evaluate_variant(**kwargs):
    train, test = build_synthetic_surface()
    X_train = train[["core_signal", "core_trend", "core_interaction"]].copy()
    y_train = train["investors_count"].copy()
    y_train.name = "investors_count"
    X_test = test[["core_signal", "core_trend", "core_interaction"]].copy()
    y_test = test["investors_count"].to_numpy(dtype=np.float64)

    wrapper = SingleModelMainlineWrapper(seed=7, **kwargs)
    wrapper.fit(
        X_train,
        y_train,
        train_raw=train,
        target="investors_count",
        task="task2_forecast",
        ablation="full",
        horizon=1,
    )
    preds = wrapper.predict(
        X_test,
        test_raw=test,
        target="investors_count",
        task="task2_forecast",
        ablation="full",
        horizon=1,
    )
    mae = float(np.mean(np.abs(preds - y_test)))
    regime = wrapper.get_regime_info()["investors_source_activation"]
    return mae, regime


def test_source_read_policy_improves_multi_profile_surface_and_learns_profile_ordering():
    baseline_mae, _ = _evaluate_variant()
    injected_mae, _ = _evaluate_variant(
        enable_investors_source_features=True,
        enable_investors_selective_source_activation=False,
    )
    read_policy_mae, regime = _evaluate_variant(enable_investors_source_read_policy=True)

    assert regime["requested_source_read_policy"] is True
    assert regime["effective_source_read_policy"] is True
    assert regime["activation_reason"] == "multi_profile_train_surface"

    learned = {int(profile_id): float(blend) for profile_id, blend in regime["learned_profile_anchor_blends"].items()}
    assert learned[1] > learned[3] > learned[0]
    assert learned[0] >= learned[2]

    assert read_policy_mae < baseline_mae