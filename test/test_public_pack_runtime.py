from pathlib import Path

import pandas as pd

from narrative.public_pack.registry import expand_public_pack_cells, load_public_pack_registry
from narrative.public_pack.runtime import (
    _compute_public_pack_metrics,
    infer_public_pack_schema,
    normalize_public_pack_frame,
    prepare_public_pack_supervision,
    resolve_public_pack_models,
    run_public_pack_model,
)


def _make_wide_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=18, freq="D"),
            "series_a": list(range(18)),
            "series_b": list(range(100, 118)),
        }
    )


def _get_test_cell():
    registry = load_public_pack_registry()
    cells = expand_public_pack_cells(registry, requested_families=["ett"])
    return cells[0]


def test_infer_public_pack_schema_from_wide_frame():
    schema = infer_public_pack_schema(_make_wide_frame())

    assert schema.layout == "wide"
    assert schema.time_col == "date"
    assert set(schema.value_cols) == {"series_a", "series_b"}


def test_prepare_public_pack_supervision_from_wide_frame():
    cell = _get_test_cell()
    frame = _make_wide_frame()
    schema = infer_public_pack_schema(frame)
    normalized = normalize_public_pack_frame(frame, schema)
    prepared = prepare_public_pack_supervision(
        normalized,
        cell=cell,
        dataset_path=Path("/tmp/test_public_pack.parquet"),
        schema=schema,
        context_length=4,
        prediction_length=2,
        max_train_samples_per_entity=8,
        max_covariates=4,
    )

    assert list(prepared.train_raw.columns[:3]) == ["entity_id", "crawled_date_day", "target"]
    assert len(prepared.X_train) > 0
    assert len(prepared.X_test) == 2
    assert len(prepared.test_raw) == 2
    assert len(prepared.y_test) == 2


def test_resolve_public_pack_models_uses_tabpfn_regressor_for_continuous_targets():
    models = resolve_public_pack_models(preset="first_wave_entrants", is_binary_target=False)

    assert "TabPFNRegressor" in models
    assert "TabPFNClassifier" not in models


def test_run_public_pack_model_with_mean_predictor():
    cell = _get_test_cell()
    frame = _make_wide_frame()
    schema = infer_public_pack_schema(frame)
    normalized = normalize_public_pack_frame(frame, schema)
    prepared = prepare_public_pack_supervision(
        normalized,
        cell=cell,
        dataset_path=Path("/tmp/test_public_pack.parquet"),
        schema=schema,
        context_length=4,
        prediction_length=2,
        max_train_samples_per_entity=8,
        max_covariates=4,
    )

    result, predictions = run_public_pack_model(prepared, "MeanPredictor")

    assert result["model_name"] == "MeanPredictor"
    assert result["category"] == "ml_tabular"
    assert len(predictions) == len(prepared.X_test)


def test_compute_public_pack_metrics_does_not_infer_binary_from_small_test_slice():
    metrics = _compute_public_pack_metrics(
        pd.Series([0.0, 1.0]).to_numpy(),
        pd.Series([0.2, 0.8]).to_numpy(),
        is_binary_target=False,
    )

    assert "accuracy" not in metrics
    assert "precision" not in metrics
    assert "recall" not in metrics
    assert "f1" not in metrics