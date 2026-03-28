#!/usr/bin/env python3
"""
Statistical Forecasting Models for Block 3 KDD'26 Benchmark.

Uses Nixtla StatsForecast plus selected standalone statistical baselines:
  AutoARIMA, AutoETS, AutoTheta, MSTL, SeasonalNaive, Prophet
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig
from .optional_runtime import ensure_optional_vendor_on_path

_logger = logging.getLogger(__name__)


class StatsForecastWrapper(ModelBase):
    """Wrapper for StatsForecast models with entity-panel support."""

    def __init__(self, config: ModelConfig, model_name: str, **kw):
        super().__init__(config)
        self.model_name = model_name
        self.model_kwargs = kw
        self._sf = None
        self._last_y = np.array([])
        self._use_fallback = False
        self._train_entity_set: set = set()

    def _get_model_instance(self):
        from statsforecast.models import (
            AutoARIMA, AutoETS, AutoTheta, MSTL, SeasonalNaive,
            Naive, HistoricAverage, WindowAverage,
        )
        # Additional models for Phase 9 saturation
        try:
            from statsforecast.models import (
                CrostonClassic, CrostonOptimized, CrostonSBA,
                DynamicOptimizedTheta, AutoCES, Holt, HoltWinters,
            )
            extra_registry = {
                "CrostonClassic": CrostonClassic,
                "CrostonOptimized": CrostonOptimized,
                "CrostonSBA": CrostonSBA,
                "DynamicOptimizedTheta": DynamicOptimizedTheta,
                "AutoCES": AutoCES,
                "Holt": Holt,
                "HoltWinters": HoltWinters,
            }
        except ImportError:
            extra_registry = {}

        registry = {
            "AutoARIMA": AutoARIMA,
            "AutoETS": AutoETS,
            "AutoTheta": AutoTheta,
            "MSTL": MSTL,
            "SF_SeasonalNaive": SeasonalNaive,
            "Naive": Naive,
            "HistoricAverage": HistoricAverage,
            "WindowAverage": WindowAverage,
            **extra_registry,
        }
        if self.model_name not in registry:
            raise ValueError(f"Unknown SF model: {self.model_name}")
        cls = registry[self.model_name]

        if self.model_name == "SF_SeasonalNaive":
            return cls(season_length=self.model_kwargs.get("season_length", 7))
        if self.model_name == "MSTL":
            return cls(season_length=self.model_kwargs.get("season_length", [7, 30]))
        if self.model_name == "WindowAverage":
            return cls(window_size=self.model_kwargs.get("window_size", 7))
        if self.model_name == "HoltWinters":
            return cls(
                season_length=self.model_kwargs.get("season_length", 7),
                error_type=self.model_kwargs.get("error_type", "A"),
            )
        if self.model_name == "Holt":
            return cls()
        return cls()

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "StatsForecastWrapper":
        from statsforecast import StatsForecast

        self._last_y = y.values
        self._use_fallback = False

        # Build entity panel if raw data available
        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target")
        # No entity cap: use ALL entities with enough history for maximum
        # test-time coverage (StatsForecast CPU models handle large panels).

        if (train_raw is not None and target is not None
                and "entity_id" in train_raw.columns):
            raw = train_raw[["entity_id", "crawled_date_day", target]].dropna(subset=[target])
            raw["crawled_date_day"] = pd.to_datetime(raw["crawled_date_day"])
            ec = raw.groupby("entity_id").size()
            valid = ec[ec >= 20].index
            if len(valid) == 0:
                self._use_fallback = True
                self._fitted = True
                return self
            self._train_entity_set = set(valid)
            raw = raw[raw["entity_id"].isin(valid)].sort_values(
                ["entity_id", "crawled_date_day"])
            df = pd.DataFrame({
                "unique_id": raw["entity_id"].values,
                "ds": raw["crawled_date_day"].values,
                "y": raw[target].values.astype(np.float32),
            })
        else:
            # Use a single series (subsample if too big)
            n = min(len(y), 5000)
            df = pd.DataFrame({
                "unique_id": ["s_0"] * n,
                "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
                "y": y.values[-n:].astype(np.float32),
            })

        _logger.info(f"  [{self.model_name}] StatsForecast panel: "
                      f"{df['unique_id'].nunique()} series, {len(df):,} rows")

        try:
            model = self._get_model_instance()
            self._sf = StatsForecast(models=[model], freq="D", n_jobs=1)
            self._sf.fit(df)
            self._fitted = True
        except Exception as e:
            _logger.warning(f"  [{self.model_name}] SF fit failed: {e}, fallback")
            self._use_fallback = True
            self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Not fitted")
        n_test = len(X)
        if self._use_fallback or self._sf is None:
            return np.full(n_test, float(np.mean(self._last_y)) if len(self._last_y) else 0)
        try:
            # Use the actual forecast horizon from kwargs, NOT len(X)
            forecast_h = kwargs.get("horizon", 30)
            forecast_h = max(1, min(forecast_h, 30))  # clamp [1, 30]

            # Try predict with requested horizon; retry with smaller h on shape errors
            fcs = None
            for try_h in [forecast_h, 14, 7, 1]:
                try:
                    fcs = self._sf.predict(h=try_h)
                    break
                except ValueError as ve:
                    _logger.warning(
                        f"  [{self.model_name}] SF predict(h={try_h}) failed: {ve}, "
                        f"retrying with smaller h"
                    )
                    continue

            if fcs is None:
                _logger.warning(
                    f"  [{self.model_name}] All SF predict attempts failed, "
                    f"returning training mean fallback"
                )
                self._use_fallback = True
                return np.full(n_test, float(np.mean(self._last_y)) if len(self._last_y) else 0)

            pred_col = fcs.columns[-1]

            # Build per-entity forecast: last step per entity
            if "unique_id" in fcs.columns or fcs.index.name == "unique_id":
                if fcs.index.name == "unique_id":
                    fcs = fcs.reset_index()
                entity_fcs = (
                    fcs.sort_values("ds")
                    .groupby("unique_id")[pred_col]
                    .last()
                    .to_dict()
                )
            else:
                entity_fcs = {}

            global_mean = float(fcs[pred_col].mean())

            # Map to test rows via entity_id
            test_raw = kwargs.get("test_raw")
            target = kwargs.get("target")
            if test_raw is not None and "entity_id" in test_raw.columns and entity_fcs:
                if target and target in test_raw.columns:
                    valid_mask = test_raw[target].notna()
                    test_entities = test_raw.loc[valid_mask, "entity_id"].values
                else:
                    test_entities = test_raw["entity_id"].values

                if len(test_entities) == n_test:
                    y_pred = np.array([
                        entity_fcs.get(eid, global_mean) for eid in test_entities
                    ])
                    _logger.info(
                        f"  [{self.model_name}] Per-entity predict: "
                        f"{len(entity_fcs)} entities, "
                        f"{len(set(test_entities) & set(entity_fcs.keys()))}/{len(set(test_entities))} matched, "
                        f"unique_preds={len(np.unique(np.round(y_pred, 4)))}"
                    )
                    return y_pred

            self._use_fallback = True
            return np.full(n_test, global_mean)
        except Exception as exc:
            _logger.warning(
                f"  [{self.model_name}] predict failed: {exc}, "
                f"returning training mean fallback"
            )
            self._use_fallback = True
            return np.full(n_test, float(np.mean(self._last_y)) if len(self._last_y) else 0)


# ============================================================================
# Prophet wrapper
# ============================================================================

class ProphetWrapper(ModelBase):
    """Per-entity Prophet wrapper for business-time-series sanity checks.

    This is intentionally conservative:
    - one Prophet model per entity
    - no hidden test-set routing
    - optional dependency guard
    - fallback to training mean when entity-level fitting fails

    It is not expected to be fast enough to become a first-wave all-cells
    benchmark workhorse.  Its value is as a recognizable business forecasting
    baseline and a local/bounded canonical comparator.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = "additive",
        yearly_seasonality: str | bool = "auto",
        weekly_seasonality: str | bool = "auto",
        daily_seasonality: bool = False,
        min_history: int = 20,
        max_entities_fit: Optional[int] = None,
        **kwargs,
    ):
        config = ModelConfig(
            name="Prophet",
            model_type="forecasting",
            params={
                "changepoint_prior_scale": changepoint_prior_scale,
                "seasonality_prior_scale": seasonality_prior_scale,
                "seasonality_mode": seasonality_mode,
                "yearly_seasonality": yearly_seasonality,
                "weekly_seasonality": weekly_seasonality,
                "daily_seasonality": daily_seasonality,
                "min_history": min_history,
                "max_entities_fit": max_entities_fit,
                **kwargs,
            },
            optional_dependency="prophet",
        )
        super().__init__(config)
        self._prophet_kwargs = {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "seasonality_mode": seasonality_mode,
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            **kwargs,
        }
        self._min_history = int(min_history)
        self._max_entities_fit = max_entities_fit
        self._models: Dict[str, Any] = {}
        self._fallback_value = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ProphetWrapper":
        ensure_optional_vendor_on_path()
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("prophet not installed. Run: pip install prophet")

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target")
        self._fallback_value = float(np.nanmean(y.values)) if len(y) else 0.0
        self._models = {}

        if train_raw is None or target is None or "entity_id" not in train_raw.columns:
            _logger.warning("  [Prophet] Missing train_raw/entity_id, using fallback-only mode")
            self._fitted = True
            return self

        raw = train_raw[["entity_id", "crawled_date_day", target]].dropna(subset=[target]).copy()
        raw["crawled_date_day"] = pd.to_datetime(raw["crawled_date_day"], errors="coerce")
        raw = raw.dropna(subset=["crawled_date_day"]).sort_values(["entity_id", "crawled_date_day"])
        counts = raw.groupby("entity_id").size()
        valid_entities = counts[counts >= self._min_history].index.tolist()
        if self._max_entities_fit:
            valid_entities = valid_entities[: self._max_entities_fit]

        for entity_id in valid_entities:
            sdf = raw.loc[raw["entity_id"] == entity_id, ["crawled_date_day", target]].copy()
            sdf.rename(columns={"crawled_date_day": "ds", target: "y"}, inplace=True)
            sdf["y"] = sdf["y"].astype(float)
            try:
                model = Prophet(**self._prophet_kwargs)
                model.fit(sdf)
                self._models[str(entity_id)] = {
                    "model": model,
                    "last_ds": pd.Timestamp(sdf["ds"].max()),
                }
            except Exception as exc:
                _logger.warning("  [Prophet] fit failed for entity %s: %s", entity_id, exc)
                continue

        _logger.info(
            "  [Prophet] fitted %d/%d eligible entities (min_history=%d)",
            len(self._models),
            len(valid_entities),
            self._min_history,
        )
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self._fitted:
            raise ValueError("ProphetWrapper is not fitted")

        n_test = len(X)
        if not self._models:
            return np.full(n_test, self._fallback_value, dtype=float)

        test_raw = kwargs.get("test_raw")
        horizon = int(kwargs.get("horizon", 1))
        if test_raw is None or "entity_id" not in test_raw.columns:
            return np.full(n_test, self._fallback_value, dtype=float)

        if "entity_id" not in X.columns and len(test_raw) != n_test:
            return np.full(n_test, self._fallback_value, dtype=float)

        if len(test_raw) == n_test:
            entity_series = test_raw["entity_id"].astype(str).to_numpy()
        else:
            entity_series = X["entity_id"].astype(str).to_numpy() if "entity_id" in X.columns else np.array([], dtype=str)
            if len(entity_series) != n_test:
                return np.full(n_test, self._fallback_value, dtype=float)

        entity_cache: Dict[str, float] = {}
        out = np.full(n_test, self._fallback_value, dtype=float)
        for idx, entity_id in enumerate(entity_series):
            if entity_id in entity_cache:
                out[idx] = entity_cache[entity_id]
                continue
            payload = self._models.get(str(entity_id))
            if payload is None:
                pred_value = self._fallback_value
            else:
                try:
                    model = payload["model"]
                    future = model.make_future_dataframe(periods=max(1, horizon), freq="D", include_history=False)
                    fcst = model.predict(future)
                    pred_value = float(fcst["yhat"].iloc[-1])
                except Exception as exc:
                    _logger.warning("  [Prophet] predict failed for entity %s: %s", entity_id, exc)
                    pred_value = self._fallback_value
            entity_cache[entity_id] = pred_value
            out[idx] = pred_value
        return out


# ============================================================================
# Factory functions
# ============================================================================

def _sf_factory(name: str):
    def create(**kw):
        cfg = ModelConfig(name=name, model_type="forecasting", params=kw,
                          optional_dependency="statsforecast")
        return StatsForecastWrapper(cfg, name, **kw)
    create.__doc__ = f"{name} model (StatsForecast)."
    return create


create_auto_arima = _sf_factory("AutoARIMA")
create_auto_ets = _sf_factory("AutoETS")
create_auto_theta = _sf_factory("AutoTheta")
create_mstl = _sf_factory("MSTL")
create_sf_seasonal_naive = _sf_factory("SF_SeasonalNaive")
# Phase 9 additions
create_croston_classic = _sf_factory("CrostonClassic")
create_croston_optimized = _sf_factory("CrostonOptimized")
create_croston_sba = _sf_factory("CrostonSBA")
create_dynamic_opt_theta = _sf_factory("DynamicOptimizedTheta")
create_auto_ces = _sf_factory("AutoCES")
create_holt = _sf_factory("Holt")
create_holt_winters = _sf_factory("HoltWinters")
create_naive = _sf_factory("Naive")
create_historic_average = _sf_factory("HistoricAverage")
create_window_average = _sf_factory("WindowAverage")


def create_prophet(**kwargs) -> ModelBase:
    return ProphetWrapper(**kwargs)


STATISTICAL_MODELS = {
    "AutoARIMA": create_auto_arima,
    "AutoETS": create_auto_ets,
    "AutoTheta": create_auto_theta,
    "MSTL": create_mstl,
    "SF_SeasonalNaive": create_sf_seasonal_naive,
    # Phase 9 additions
    "CrostonClassic": create_croston_classic,
    "CrostonOptimized": create_croston_optimized,
    "CrostonSBA": create_croston_sba,
    "DynamicOptimizedTheta": create_dynamic_opt_theta,
    "AutoCES": create_auto_ces,
    "Holt": create_holt,
    "HoltWinters": create_holt_winters,
    "Naive": create_naive,
    "HistoricAverage": create_historic_average,
    "WindowAverage": create_window_average,
    "Prophet": create_prophet,
}


def get_statistical_model(name: str, **kwargs) -> ModelBase:
    if name not in STATISTICAL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    return STATISTICAL_MODELS[name](**kwargs)


def list_statistical_models() -> list:
    return list(STATISTICAL_MODELS.keys())


def check_statsforecast_available() -> bool:
    try:
        import statsforecast
        return True
    except ImportError:
        return False
