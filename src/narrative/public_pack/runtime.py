from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .registry import PublicPackCell


REPO_ROOT = Path(__file__).resolve().parents[3]

_TIME_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "timestamp",
    "datetime",
    "date",
    "ds",
    "time",
    "crawled_date_day",
)
_ENTITY_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "entity_id",
    "unique_id",
    "series_id",
    "item_id",
    "id",
)
_TARGET_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "target",
    "y",
    "value",
    "values",
)
_RAW_PANEL_MODEL_CATEGORIES = {
    "deep_classical",
    "transformer_sota",
    "foundation",
    "statistical",
    "irregular",
    "tslib_sota",
    "autofit",
}


FIRST_WAVE_MODEL_PRESETS: Dict[str, Tuple[str, ...]] = {
    "first_wave_entrants": (
        "SAMformer",
        "LightGTS",
        "OLinear",
        "UniTS",
        "Prophet",
        "TabPFN",
    ),
    "first_wave_forecasting": (
        "SAMformer",
        "LightGTS",
        "OLinear",
        "UniTS",
        "Prophet",
    ),
}


@dataclass(frozen=True)
class PublicPackSchema:
    layout: str
    time_col: Optional[str]
    entity_col: Optional[str]
    target_col: Optional[str]
    value_cols: Tuple[str, ...] = ()
    covariate_cols: Tuple[str, ...] = ()


@dataclass
class PublicPackPreparedData:
    cell: PublicPackCell
    dataset_path: Path
    schema: PublicPackSchema
    normalized_frame: pd.DataFrame
    train_raw: pd.DataFrame
    val_raw: pd.DataFrame
    test_raw: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    target_name: str
    context_length: int
    prediction_length: int
    covariate_cols: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_binary_target(self) -> bool:
        if self.y_train.empty:
            return False
        values = self.y_train.to_numpy(dtype=np.float64, copy=False)
        unique = np.unique(values[np.isfinite(values)])
        return bool(len(unique) and np.all(np.isin(unique, [0.0, 1.0])))


def _normalize_token(text: str) -> str:
    return (
        text.strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _resolve_input_path(path_like: Path | str) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _candidate_variant_stems(cell: PublicPackCell) -> Tuple[str, ...]:
    seeds = {
        cell.variant,
        cell.display_name,
        cell.family,
        _normalize_token(cell.variant),
        _normalize_token(cell.display_name),
        _normalize_token(cell.family),
    }
    ordered = []
    seen = set()
    for seed in seeds:
        if not seed:
            continue
        normalized = _normalize_token(seed)
        for value in (seed, normalized):
            if value and value not in seen:
                seen.add(value)
                ordered.append(value)
    return tuple(ordered)


def _iter_dataset_file_candidates(root: Path, stems: Sequence[str]) -> Iterable[Path]:
    suffixes = (".parquet", ".csv")
    if root.is_file():
        yield root
        return
    if not root.exists():
        return

    for stem in stems:
        for suffix in suffixes:
            direct = root / f"{stem}{suffix}"
            if direct.exists():
                yield direct

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in suffixes:
            continue
        normalized_stem = _normalize_token(path.stem)
        if normalized_stem in stems or path.stem in stems:
            yield path


def resolve_public_pack_dataset_path(
    cell: PublicPackCell,
    dataset_path: Optional[Path | str] = None,
) -> Path:
    if dataset_path is not None:
        path = _resolve_input_path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"public-pack dataset path not found: {path}")
        return path

    roots = []
    for root_text in (cell.processed_root, cell.raw_root):
        if root_text:
            roots.append(_resolve_input_path(root_text))

    stems = _candidate_variant_stems(cell)
    looked = []
    for root in roots:
        looked.append(str(root))
        candidates = list(dict.fromkeys(_iter_dataset_file_candidates(root, stems)))
        if candidates:
            candidates.sort(key=lambda path: (len(path.parts), len(str(path))))
            return candidates[0]

    raise FileNotFoundError(
        "could not resolve public-pack dataset file for "
        f"family={cell.family}, variant={cell.variant}; looked under {looked}"
    )


def load_public_pack_frame(path: Path | str) -> pd.DataFrame:
    data_path = _resolve_input_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"public-pack dataset file not found: {data_path}")
    if data_path.is_dir():
        candidates = sorted(
            child for child in data_path.rglob("*") if child.suffix.lower() in {".parquet", ".csv"}
        )
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one dataset file under {data_path}, found {len(candidates)}"
            )
        data_path = candidates[0]

    if data_path.suffix.lower() == ".parquet":
        return pd.read_parquet(data_path)
    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path)
    raise ValueError(f"unsupported public-pack dataset suffix: {data_path.suffix}")


def _first_matching_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized_map = {_normalize_token(col): col for col in df.columns}
    for candidate in candidates:
        hit = normalized_map.get(_normalize_token(candidate))
        if hit is not None:
            return hit
    return None


def infer_public_pack_schema(
    df: pd.DataFrame,
    target: Optional[str] = None,
) -> PublicPackSchema:
    if df.empty:
        raise ValueError("cannot infer schema from an empty public-pack frame")

    time_col = _first_matching_column(df, _TIME_COLUMN_CANDIDATES)
    entity_col = _first_matching_column(df, _ENTITY_COLUMN_CANDIDATES)

    target_col = None
    if target and target in df.columns:
        target_col = target
    elif target:
        normalized_target = _normalize_token(target)
        for column in df.columns:
            if _normalize_token(column) == normalized_target:
                target_col = column
                break
    if target_col is None:
        target_col = _first_matching_column(df, _TARGET_COLUMN_CANDIDATES)

    numeric_cols = [
        column for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column])
    ]
    reserved = {value for value in (time_col, entity_col) if value is not None}

    if target_col is not None:
        covariate_cols = tuple(
            column for column in numeric_cols if column not in reserved and column != target_col
        )
        return PublicPackSchema(
            layout="long",
            time_col=time_col,
            entity_col=entity_col,
            target_col=target_col,
            covariate_cols=covariate_cols,
        )

    value_cols = tuple(column for column in numeric_cols if column not in reserved)
    if value_cols:
        return PublicPackSchema(
            layout="wide",
            time_col=time_col,
            entity_col=None,
            target_col=None,
            value_cols=value_cols,
        )

    raise ValueError(
        "could not infer public-pack schema: no usable target column and no numeric value columns"
    )


def _coerce_datetime_column(values: pd.Series, length: int) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.notna().any():
        parsed = parsed.dt.tz_convert(None)
        fallback = pd.Series(pd.date_range("2000-01-01", periods=length, freq="D"), index=values.index)
        return parsed.where(parsed.notna(), fallback)
    return pd.Series(pd.date_range("2000-01-01", periods=length, freq="D"), index=values.index)


def _add_time_covariates(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["crawled_date_day"] = pd.to_datetime(enriched["crawled_date_day"], errors="coerce")
    if enriched["crawled_date_day"].isna().all():
        enriched["crawled_date_day"] = pd.date_range("2000-01-01", periods=len(enriched), freq="D")

    enriched = enriched.sort_values(["entity_id", "crawled_date_day"], kind="mergesort").reset_index(drop=True)
    enriched["time_idx"] = enriched.groupby("entity_id").cumcount().astype(np.float64)
    enriched["month"] = enriched["crawled_date_day"].dt.month.fillna(0).astype(np.float64)
    enriched["day"] = enriched["crawled_date_day"].dt.day.fillna(0).astype(np.float64)
    enriched["dayofweek"] = enriched["crawled_date_day"].dt.dayofweek.fillna(0).astype(np.float64)
    enriched["hour"] = enriched["crawled_date_day"].dt.hour.fillna(0).astype(np.float64)
    enriched["is_month_start"] = enriched["crawled_date_day"].dt.is_month_start.fillna(False).astype(np.float64)
    enriched["is_month_end"] = enriched["crawled_date_day"].dt.is_month_end.fillna(False).astype(np.float64)
    ordered_columns = [
        column
        for column in ("entity_id", "crawled_date_day", "target")
        if column in enriched.columns
    ]
    ordered_columns.extend(
        column for column in enriched.columns if column not in ordered_columns
    )
    return enriched[ordered_columns]


def normalize_public_pack_frame(
    df: pd.DataFrame,
    schema: PublicPackSchema,
) -> pd.DataFrame:
    if schema.layout == "wide":
        time_values = (
            _coerce_datetime_column(df[schema.time_col], len(df))
            if schema.time_col is not None
            else pd.Series(pd.date_range("2000-01-01", periods=len(df), freq="D"), index=df.index)
        )
        wide = df.copy()
        wide["__public_pack_time"] = time_values
        long_df = wide[["__public_pack_time", *schema.value_cols]].melt(
            id_vars=["__public_pack_time"],
            value_vars=list(schema.value_cols),
            var_name="entity_id",
            value_name="target",
        )
        out = long_df.rename(columns={"__public_pack_time": "crawled_date_day"})
        out["entity_id"] = out["entity_id"].astype(str)
        out["target"] = pd.to_numeric(out["target"], errors="coerce")
        return _add_time_covariates(out)

    entity_values = (
        df[schema.entity_col].astype(str)
        if schema.entity_col is not None
        else pd.Series(["series_0"] * len(df), index=df.index)
    )
    time_values = (
        _coerce_datetime_column(df[schema.time_col], len(df))
        if schema.time_col is not None
        else pd.Series(pd.date_range("2000-01-01", periods=len(df), freq="D"), index=df.index)
    )

    out = pd.DataFrame(
        {
            "entity_id": entity_values,
            "crawled_date_day": time_values,
            "target": pd.to_numeric(df[schema.target_col], errors="coerce"),
        }
    )
    for column in schema.covariate_cols:
        out[column] = pd.to_numeric(df[column], errors="coerce")

    return _add_time_covariates(out)


def _top_variance_covariates(df: pd.DataFrame, max_covariates: int) -> Tuple[str, ...]:
    numeric_cols = [
        column for column in df.select_dtypes(include=[np.number]).columns
        if column != "target"
    ]
    stats: List[Tuple[str, float]] = []
    for column in numeric_cols:
        values = pd.to_numeric(df[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) < 8:
            continue
        variance = float(finite.var())
        if variance <= 1e-12:
            continue
        stats.append((column, variance))
    stats.sort(key=lambda item: item[1], reverse=True)
    return tuple(column for column, _ in stats[:max_covariates])


def _sanitize_numeric(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    return series.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)


def _context_feature_row(
    context: pd.DataFrame,
    covariate_cols: Sequence[str],
    prediction_length: int,
) -> Dict[str, float]:
    target_values = _sanitize_numeric(context["target"])
    row: Dict[str, float] = {
        "context_observations": float(len(context)),
        "prediction_length": float(prediction_length),
        "target_last": float(target_values[-1]),
        "target_mean": float(target_values.mean()),
        "target_std": float(target_values.std()),
        "target_min": float(target_values.min()),
        "target_max": float(target_values.max()),
        "target_delta": float(target_values[-1] - target_values[0]),
    }

    lag_count = min(8, len(target_values))
    for lag in range(1, lag_count + 1):
        row[f"target_lag_{lag}"] = float(target_values[-lag])

    for column in covariate_cols:
        cov_values = _sanitize_numeric(context[column])
        row[f"{column}__last"] = float(cov_values[-1])
        row[f"{column}__mean"] = float(cov_values.mean())
        row[f"{column}__std"] = float(cov_values.std())

    return row


def prepare_public_pack_supervision(
    frame: pd.DataFrame,
    *,
    cell: PublicPackCell,
    dataset_path: Path,
    schema: PublicPackSchema,
    context_length: int,
    prediction_length: int,
    max_entities: Optional[int] = None,
    max_train_samples_per_entity: int = 512,
    max_covariates: int = 8,
) -> PublicPackPreparedData:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive")

    clean = frame.dropna(subset=["target"]).copy()
    clean = clean.sort_values(["entity_id", "crawled_date_day"], kind="mergesort")

    entity_sizes = clean.groupby("entity_id").size().sort_values(ascending=False)
    min_required = context_length + 2 * prediction_length
    selected_entities = [entity for entity, size in entity_sizes.items() if int(size) >= min_required]
    if max_entities is not None:
        selected_entities = selected_entities[:max_entities]
    if not selected_entities:
        raise ValueError(
            f"no entities have enough history for context={context_length}, horizon={prediction_length}"
        )

    clean = clean[clean["entity_id"].isin(selected_entities)].copy()
    covariate_cols = _top_variance_covariates(clean, max_covariates=max_covariates)

    train_feature_rows: List[Dict[str, float]] = []
    train_targets: List[float] = []
    val_feature_rows: List[Dict[str, float]] = []
    val_targets: List[float] = []
    test_feature_rows: List[Dict[str, float]] = []
    test_targets: List[float] = []
    train_raw_parts: List[pd.DataFrame] = []
    val_raw_rows: List[pd.DataFrame] = []
    test_raw_rows: List[pd.DataFrame] = []

    for entity_id, group in clean.groupby("entity_id", sort=False):
        group = group.sort_values("crawled_date_day", kind="mergesort").reset_index(drop=True)
        history = group.iloc[:-prediction_length].copy()
        if len(history) < context_length + prediction_length:
            continue

        train_raw_parts.append(history)
        test_raw_rows.append(group.iloc[[-1]].copy())

        min_target_idx = context_length + prediction_length - 1
        candidate_target_indices = np.arange(min_target_idx, len(history), dtype=int)
        if len(candidate_target_indices) == 0:
            continue
        if len(candidate_target_indices) > max_train_samples_per_entity:
            take = np.linspace(
                0,
                len(candidate_target_indices) - 1,
                num=max_train_samples_per_entity,
                dtype=int,
            )
            candidate_target_indices = candidate_target_indices[take]

        reserve_for_val = len(candidate_target_indices) >= 3
        if reserve_for_val:
            val_target_idx = int(candidate_target_indices[-1])
            train_target_indices = candidate_target_indices[:-1]
            val_context_end = val_target_idx - prediction_length
            val_context = history.iloc[val_context_end - context_length + 1 : val_context_end + 1]
            val_feature_rows.append(
                _context_feature_row(val_context, covariate_cols, prediction_length)
            )
            val_targets.append(float(history["target"].iloc[val_target_idx]))
            val_raw_rows.append(history.iloc[[val_target_idx]].copy())
        else:
            train_target_indices = candidate_target_indices

        for target_idx in train_target_indices:
            context_end = int(target_idx) - prediction_length
            context = history.iloc[context_end - context_length + 1 : context_end + 1]
            train_feature_rows.append(
                _context_feature_row(context, covariate_cols, prediction_length)
            )
            train_targets.append(float(history["target"].iloc[int(target_idx)]))

        test_context = history.iloc[-context_length:]
        test_feature_rows.append(_context_feature_row(test_context, covariate_cols, prediction_length))
        test_targets.append(float(group["target"].iloc[-1]))

    if not train_feature_rows or not test_feature_rows:
        raise ValueError("public-pack supervision preparation produced no train/test samples")

    empty_raw = clean.head(0).copy()
    train_raw = pd.concat(train_raw_parts, ignore_index=True) if train_raw_parts else empty_raw.copy()
    val_raw = pd.concat(val_raw_rows, ignore_index=True) if val_raw_rows else empty_raw.copy()
    test_raw = pd.concat(test_raw_rows, ignore_index=True) if test_raw_rows else empty_raw.copy()

    return PublicPackPreparedData(
        cell=cell,
        dataset_path=dataset_path,
        schema=schema,
        normalized_frame=clean.reset_index(drop=True),
        train_raw=train_raw.reset_index(drop=True),
        val_raw=val_raw.reset_index(drop=True),
        test_raw=test_raw.reset_index(drop=True),
        X_train=pd.DataFrame(train_feature_rows).fillna(0.0),
        y_train=pd.Series(train_targets, name="target", dtype=np.float64),
        X_val=pd.DataFrame(val_feature_rows).fillna(0.0),
        y_val=pd.Series(val_targets, name="target", dtype=np.float64),
        X_test=pd.DataFrame(test_feature_rows).fillna(0.0),
        y_test=pd.Series(test_targets, name="target", dtype=np.float64),
        target_name="target",
        context_length=context_length,
        prediction_length=prediction_length,
        covariate_cols=covariate_cols,
        metadata={
            "n_entities": len(selected_entities),
            "n_train_rows": len(train_raw),
            "n_train_samples": len(train_feature_rows),
            "n_val_samples": len(val_feature_rows),
            "n_test_samples": len(test_feature_rows),
        },
    )


def prepare_public_pack_data(
    cell: PublicPackCell,
    *,
    dataset_path: Optional[Path | str] = None,
    target: Optional[str] = None,
    context_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    max_entities: Optional[int] = None,
    max_train_samples_per_entity: int = 512,
    max_covariates: int = 8,
) -> PublicPackPreparedData:
    resolved_context = context_length if context_length is not None else cell.context_length
    resolved_prediction = prediction_length if prediction_length is not None else cell.prediction_length
    if resolved_context is None or resolved_prediction is None:
        raise ValueError(
            "public-pack cell is missing context_length or prediction_length; provide overrides"
        )

    resolved_path = resolve_public_pack_dataset_path(cell, dataset_path=dataset_path)
    raw_frame = load_public_pack_frame(resolved_path)
    schema = infer_public_pack_schema(raw_frame, target=target)
    normalized_frame = normalize_public_pack_frame(raw_frame, schema)
    return prepare_public_pack_supervision(
        normalized_frame,
        cell=cell,
        dataset_path=resolved_path,
        schema=schema,
        context_length=int(resolved_context),
        prediction_length=int(resolved_prediction),
        max_entities=max_entities,
        max_train_samples_per_entity=max_train_samples_per_entity,
        max_covariates=max_covariates,
    )


def resolve_public_pack_models(
    requested_models: Optional[Sequence[str]] = None,
    *,
    preset: Optional[str] = None,
    is_binary_target: bool = False,
) -> List[str]:
    if requested_models:
        seeds = list(requested_models)
    else:
        preset_name = preset or "first_wave_entrants"
        if preset_name not in FIRST_WAVE_MODEL_PRESETS:
            raise ValueError(
                f"unknown public-pack model preset: {preset_name}; "
                f"available={sorted(FIRST_WAVE_MODEL_PRESETS)}"
            )
        seeds = list(FIRST_WAVE_MODEL_PRESETS[preset_name])

    resolved: List[str] = []
    for name in seeds:
        if name == "TabPFN":
            resolved.append("TabPFNClassifier" if is_binary_target else "TabPFNRegressor")
        else:
            resolved.append(name)

    deduped: List[str] = []
    seen = set()
    for name in resolved:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _compute_public_pack_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    is_binary_target: bool = False,
    n_bootstrap: int = 0,
) -> Dict[str, float]:
    from narrative.block3.metrics import (
        accuracy,
        auroc,
        bootstrap_ci,
        f1_score,
        mae,
        mape,
        precision,
        recall,
        rmse,
        smape,
    )

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {}

    y_true = y_true[mask].astype(np.float64, copy=False)
    y_pred = y_pred[mask].astype(np.float64, copy=False)
    metrics: Dict[str, float] = {
        "mae": float(mae(y_true, y_pred)),
        "rmse": float(rmse(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
    }
    nonzero = np.abs(y_true) > 1e-12
    if nonzero.any():
        metrics["mape"] = float(mape(y_true[nonzero], y_pred[nonzero]))

    if is_binary_target:
        y_label = (y_pred >= 0.5).astype(int)
        metrics["accuracy"] = float(accuracy(y_true.astype(int), y_label))
        metrics["precision"] = float(precision(y_true.astype(int), y_label))
        metrics["recall"] = float(recall(y_true.astype(int), y_label))
        metrics["f1"] = float(f1_score(y_true.astype(int), y_label))
        if len(np.unique(y_true.astype(int))) > 1:
            metrics["auc"] = float(auroc(y_true.astype(int), np.clip(y_pred, 0.0, 1.0)))

    if n_bootstrap > 0 and len(y_true) >= 10:
        try:
            ci = bootstrap_ci(y_true, y_pred, mae, n_bootstrap=n_bootstrap, seed=42)
            metrics["mae_ci_lower"] = float(ci.ci_lower)
            metrics["mae_ci_upper"] = float(ci.ci_upper)
        except Exception:
            pass

    return metrics


def run_public_pack_model(
    prepared: PublicPackPreparedData,
    model_name: str,
    *,
    model_kwargs: Optional[Dict[str, Any]] = None,
    n_bootstrap: int = 0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    from narrative.block3.models.registry import MODEL_CATEGORIES, get_model

    category = next(
        (candidate for candidate, models in MODEL_CATEGORIES.items() if model_name in models),
        None,
    )
    if category is None:
        raise ValueError(f"unknown public-pack model: {model_name}")

    fit_kwargs: Dict[str, Any] = {}
    predict_kwargs: Dict[str, Any] = {}
    if category in _RAW_PANEL_MODEL_CATEGORIES:
        fit_kwargs.update(
            {
                "train_raw": prepared.train_raw,
                "target": prepared.target_name,
                "horizon": prepared.prediction_length,
            }
        )
        predict_kwargs.update(
            {
                "test_raw": prepared.test_raw,
                "target": prepared.target_name,
                "horizon": prepared.prediction_length,
            }
        )

    model = get_model(model_name, **(model_kwargs or {}))
    model.fit(prepared.X_train, prepared.y_train, **fit_kwargs)
    if predict_kwargs:
        y_pred = np.asarray(model.predict(prepared.X_test, **predict_kwargs), dtype=np.float64)
    else:
        y_pred = np.asarray(model.predict(prepared.X_test), dtype=np.float64)

    y_true = prepared.y_test.to_numpy(dtype=np.float64, copy=False)
    pred_finite = np.isfinite(y_pred)
    prediction_coverage_ratio = float(pred_finite.mean()) if len(y_pred) else 0.0
    rounded = np.round(y_pred[pred_finite], 12) if pred_finite.any() else np.array([])
    constant_prediction = bool(len(rounded) and np.unique(rounded).size <= 1)
    metrics = _compute_public_pack_metrics(
        y_true,
        y_pred,
        is_binary_target=prepared.is_binary_target,
        n_bootstrap=n_bootstrap,
    )

    result = {
        "pack": prepared.cell.pack,
        "family": prepared.cell.family,
        "variant": prepared.cell.variant,
        "task_type": prepared.cell.task_type,
        "dataset_path": str(prepared.dataset_path),
        "model_name": model_name,
        "category": category,
        "context_length": prepared.context_length,
        "prediction_length": prepared.prediction_length,
        "n_entities": int(prepared.metadata.get("n_entities", 0)),
        "n_train_samples": int(len(prepared.X_train)),
        "n_test_samples": int(len(prepared.X_test)),
        "prediction_coverage_ratio": prediction_coverage_ratio,
        "constant_prediction": constant_prediction,
        "fairness_pass": bool(prediction_coverage_ratio >= 0.98 and not constant_prediction),
        "metrics": metrics,
    }

    prediction_frame = prepared.test_raw[["entity_id", "crawled_date_day", prepared.target_name]].copy()
    prediction_frame = prediction_frame.rename(columns={prepared.target_name: "y_true"})
    prediction_frame["y_pred"] = y_pred
    prediction_frame["family"] = prepared.cell.family
    prediction_frame["variant"] = prepared.cell.variant
    prediction_frame["model_name"] = model_name
    prediction_frame["prediction_length"] = prepared.prediction_length
    prediction_frame["context_length"] = prepared.context_length
    return result, prediction_frame.reset_index(drop=True)


def run_public_pack_models(
    prepared: PublicPackPreparedData,
    model_names: Sequence[str],
    *,
    model_kwargs_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
    n_bootstrap: int = 0,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    results: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    overrides = model_kwargs_by_name or {}

    for model_name in model_names:
        try:
            result, prediction_frame = run_public_pack_model(
                prepared,
                model_name,
                model_kwargs=overrides.get(model_name),
                n_bootstrap=n_bootstrap,
            )
        except Exception as exc:
            result = {
                "pack": prepared.cell.pack,
                "family": prepared.cell.family,
                "variant": prepared.cell.variant,
                "task_type": prepared.cell.task_type,
                "dataset_path": str(prepared.dataset_path),
                "model_name": model_name,
                "context_length": prepared.context_length,
                "prediction_length": prepared.prediction_length,
                "n_entities": int(prepared.metadata.get("n_entities", 0)),
                "n_train_samples": int(len(prepared.X_train)),
                "n_test_samples": int(len(prepared.X_test)),
                "prediction_coverage_ratio": 0.0,
                "constant_prediction": False,
                "fairness_pass": False,
                "error": str(exc),
                "metrics": {},
            }
            prediction_frame = pd.DataFrame()
        results.append(result)
        if not prediction_frame.empty:
            prediction_frames.append(prediction_frame)

    merged_predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame(
            columns=[
                "entity_id",
                "crawled_date_day",
                "y_true",
                "y_pred",
                "family",
                "variant",
                "model_name",
                "prediction_length",
                "context_length",
            ]
        )
    )
    return results, merged_predictions


__all__ = [
    "FIRST_WAVE_MODEL_PRESETS",
    "PublicPackPreparedData",
    "PublicPackSchema",
    "infer_public_pack_schema",
    "load_public_pack_frame",
    "normalize_public_pack_frame",
    "prepare_public_pack_data",
    "prepare_public_pack_supervision",
    "resolve_public_pack_dataset_path",
    "resolve_public_pack_models",
    "run_public_pack_model",
    "run_public_pack_models",
]