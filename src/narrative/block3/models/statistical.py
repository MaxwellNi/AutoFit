#!/usr/bin/env python3
"""
Statistical Forecasting Models for Block 3 KDD'26 Benchmark.

Uses Nixtla StatsForecast:
  AutoARIMA, AutoETS, AutoTheta, MSTL, SeasonalNaive
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig

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
        registry = {
            "AutoARIMA": AutoARIMA,
            "AutoETS": AutoETS,
            "AutoTheta": AutoTheta,
            "MSTL": MSTL,
            "SF_SeasonalNaive": SeasonalNaive,
            "Naive": Naive,
            "HistoricAverage": HistoricAverage,
            "WindowAverage": WindowAverage,
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
        return cls()

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "StatsForecastWrapper":
        from statsforecast import StatsForecast

        self._last_y = y.values
        self._use_fallback = False

        # Build entity panel if raw data available
        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target")
        MAX_ENTITIES = 500  # increased for better test coverage

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
            rng = np.random.RandomState(42)
            sampled = rng.choice(valid, size=min(MAX_ENTITIES, len(valid)), replace=False)
            self._train_entity_set = set(sampled)
            raw = raw[raw["entity_id"].isin(sampled)].sort_values(
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

            return np.full(n_test, global_mean)
        except Exception as exc:
            _logger.warning(
                f"  [{self.model_name}] predict failed: {exc}, "
                f"returning training mean fallback"
            )
            return np.full(n_test, float(np.mean(self._last_y)) if len(self._last_y) else 0)


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


STATISTICAL_MODELS = {
    "AutoARIMA": create_auto_arima,
    "AutoETS": create_auto_ets,
    "AutoTheta": create_auto_theta,
    "MSTL": create_mstl,
    "SF_SeasonalNaive": create_sf_seasonal_naive,
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
