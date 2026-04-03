#!/usr/bin/env python3
"""
Model Registry for Block 3 KDD'26 Benchmark — Phase 15+H.

Unified interface for all 143+ registered models across 8 categories:
  - ml_tabular      (20): LogisticRegression, Ridge, Lasso, ElasticNet, SVR,
                           KNN, RandomForest, ExtraTrees, HistGradientBoosting,
                           LightGBM, XGBoost, CatBoost, XGBoostPoisson,
                           LightGBMTweedie, NegativeBinomialGLM, TabPFNRegressor,
                           TabPFNClassifier, QuantileRegressor, SeasonalNaive,
                           MeanPredictor
  - statistical     (16): AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive,
                           CrostonClassic, CrostonOptimized, CrostonSBA,
                           DynamicOptimizedTheta, AutoCES, Holt, HoltWinters,
                           Naive, HistoricAverage, WindowAverage, Prophet
  - deep_classical   (9): NBEATS, NHITS, TFT, DeepAR, GRU, LSTM, TCN, MLP,
                           DilatedRNN
  - transformer_sota(24): PatchTST, iTransformer, TimesNet, TSMixer, Informer,
                           Autoformer, FEDformer, VanillaTransformer, TiDE,
                           NBEATSx, BiTCN, KAN, RMoK, SOFTS, StemGNN, DLinear,
                           NLinear, TimeMixer, TimeXer, TSMixerx, xLSTM,
                           TimeLLM, DeepNPTS, SAMformer
  - foundation      (15): Chronos, ChronosBolt, Chronos2, Moirai, MoiraiLarge,
                           Moirai2, Timer, TimeMoE, MOMENT, LagLlama, TimesFM,
                           Sundial, TTM, TimerXL, TimesFM2
  - irregular        (4): GRU-D, SAITS, BRITS, CSDI
  - tslib_sota      (42): TimeFilter, WPMixer, MultiPatchFormer, TiRex,
                           MSGNet, PAttn, MambaSimple, Mamba, Koopa, FreTS,
                           Crossformer, MICN, SegRNN, ETSformer,
                           NonstationaryTransformer, FiLM, SCINet, LightTS,
                           Pyraformer, Reformer, KANAD, FITS, SparseTSF, CATS,
                           Fredformer, CycleNet, xPatch, FilterTS,
                           CFPT, DeformableTST, ModernTCN, PathFormer, SEMPO,
                           TimePerceiver, TimeBridge, TQNet, PIR, CARD, PDF,
                           TimeRecipe, DUET, SRSNet
    - autofit          (1): AutoFitV739 only in the active current registry

Historical AutoFit-family implementations remain in source for archival and
audit purposes, but are intentionally excluded from the active registry.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import ModelBase, ModelConfig
from .traditional_ml import TRADITIONAL_ML_MODELS, get_traditional_model
from .statistical import STATISTICAL_MODELS, get_statistical_model
from .deep_models import (
    DEEP_MODELS,
    TRANSFORMER_MODELS,
    FOUNDATION_MODELS,
    get_deep_model,
)
from .irregular_models import IRREGULAR_MODELS, get_irregular_model
from ..autofit_status import CURRENT_AUTOFIT_BASELINE, get_retired_autofit_reason, is_retired_autofit_model
from .autofit_wrapper import ACTIVE_AUTOFIT_MODELS
from .tslib_models import TSLIB_MODELS


# ============================================================================
# Unified Registry
# ============================================================================

MODEL_CATEGORIES: Dict[str, List[str]] = {
    "statistical":      list(STATISTICAL_MODELS.keys()),
    "ml_tabular":       list(TRADITIONAL_ML_MODELS.keys()),
    "deep_classical":   list(DEEP_MODELS.keys()),
    "transformer_sota": list(TRANSFORMER_MODELS.keys()),
    "foundation":       list(FOUNDATION_MODELS.keys()),
    "irregular":        list(IRREGULAR_MODELS.keys()),
    "tslib_sota":       list(TSLIB_MODELS.keys()),
    "autofit":          list(ACTIVE_AUTOFIT_MODELS.keys()),
}

# Flattened registry for direct lookup
_ALL_MODELS: Dict[str, Any] = {}
_ALL_MODELS.update(TRADITIONAL_ML_MODELS)
_ALL_MODELS.update(STATISTICAL_MODELS)
_ALL_MODELS.update(DEEP_MODELS)
_ALL_MODELS.update(TRANSFORMER_MODELS)
_ALL_MODELS.update(FOUNDATION_MODELS)
_ALL_MODELS.update(IRREGULAR_MODELS)
_ALL_MODELS.update(TSLIB_MODELS)
_ALL_MODELS.update(ACTIVE_AUTOFIT_MODELS)


# ============================================================================
# Lookup helpers
# ============================================================================

def get_model(name: str, **kwargs) -> ModelBase:
    """Get a model instance by name."""
    if name not in _ALL_MODELS:
        if is_retired_autofit_model(name):
            raise ValueError(
                f"Retired AutoFit model blocked from current registry: {name}. "
                f"Reason: {get_retired_autofit_reason(name)}. "
                f"Only {CURRENT_AUTOFIT_BASELINE} remains active."
            )
        raise ValueError(
            f"Unknown model: {name}. "
            f"Use list_models() to see available models."
        )
    return _ALL_MODELS[name](**kwargs)


def get_models_by_category(category: str, **kwargs) -> List[ModelBase]:
    """Get all models in a category."""
    if category not in MODEL_CATEGORIES:
        raise ValueError(
            f"Unknown category: {category}. "
            f"Available: {list(MODEL_CATEGORIES.keys())}"
        )
    return [get_model(name, **kwargs) for name in MODEL_CATEGORIES[category]]


def list_models() -> Dict[str, List[str]]:
    """List all available models by category."""
    return MODEL_CATEGORIES.copy()


def list_all_models() -> List[str]:
    """Flat list of every model name."""
    return list(_ALL_MODELS.keys())


def get_model_info(name: str) -> Dict[str, Any]:
    """Get metadata for a model."""
    if name not in _ALL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    model = _ALL_MODELS[name]()
    config = model.config
    category = None
    for cat, models in MODEL_CATEGORIES.items():
        if name in models:
            category = cat
            break
    return {
        "name": name,
        "category": category,
        "model_type": config.model_type,
        "supports_probabilistic": config.supports_probabilistic,
        "supports_missing": config.supports_missing,
        "optional_dependency": config.optional_dependency,
    }


def check_model_available(name: str) -> bool:
    """Check if a model can be instantiated (deps present)."""
    if name not in _ALL_MODELS:
        return False
    try:
        _ALL_MODELS[name]()
        return True
    except (ImportError, Exception):
        return False


def get_available_models() -> Dict[str, List[str]]:
    """Get models whose dependencies are actually installed."""
    available: Dict[str, List[str]] = {}
    for category, models in MODEL_CATEGORIES.items():
        available[category] = [m for m in models if check_model_available(m)]
    return available


# ============================================================================
# Task helpers
# ============================================================================

def get_models_for_task(task_type: str, **kwargs) -> List[ModelBase]:
    """Get appropriate models for a task type."""
    if task_type == "classification":
        return [get_model(n, **kwargs) for n in
                ["LogisticRegression", "RandomForest", "HistGradientBoosting",
                 "LightGBM", "XGBoost"]]
    if task_type == "regression":
        return [get_model(n, **kwargs) for n in
                ["Ridge", "Lasso", "RandomForest", "HistGradientBoosting",
                 "LightGBM", "XGBoost", "CatBoost"]]
    if task_type == "forecasting":
        models = []
        for n in (list(STATISTICAL_MODELS) + list(DEEP_MODELS) +
                  list(TRANSFORMER_MODELS)):
            try:
                models.append(get_model(n, **kwargs))
            except (ImportError, Exception):
                pass
        return models
    raise ValueError(f"Unknown task type: {task_type}")


def get_baseline_models(**kwargs) -> List[ModelBase]:
    """Baseline models for reference."""
    return [get_model("MeanPredictor", **kwargs),
            get_model("SeasonalNaive", **kwargs)]


# ============================================================================
# Presets
# ============================================================================

PRESET_CONFIGS = {
    "smoke_test": {
        "models": ["MeanPredictor", "Ridge", "LightGBM"],
        "params": {"n_estimators": 10, "max_depth": 3},
    },
    "quick": {
        "models": ["MeanPredictor", "SeasonalNaive", "Ridge", "LightGBM",
                    "AutoARIMA"],
        "params": {"n_estimators": 50},
    },
    "standard": {
        "models": [
            "MeanPredictor", "SeasonalNaive",
            "Ridge", "Lasso", "RandomForest", "LightGBM", "XGBoost",
            "AutoARIMA", "AutoETS", "AutoTheta",
        ],
        "params": {"n_estimators": 100},
    },
    "comprehensive": {
        "models": list(_ALL_MODELS.keys()),
        "params": {},
    },
}


def get_preset_models(preset: str) -> List[ModelBase]:
    """Get models from a named preset."""
    if preset not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset: {preset}. "
            f"Available: {list(PRESET_CONFIGS.keys())}"
        )
    config = PRESET_CONFIGS[preset]
    params = config.get("params", {})
    models = []
    for name in config["models"]:
        try:
            models.append(get_model(name, **params))
        except (ImportError, Exception):
            continue
    return models


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "get_model",
    "get_models_by_category",
    "list_models",
    "list_all_models",
    "get_model_info",
    "check_model_available",
    "get_available_models",
    "get_models_for_task",
    "get_baseline_models",
    "get_preset_models",
    "PRESET_CONFIGS",
    "MODEL_CATEGORIES",
    "ModelBase",
    "ModelConfig",
]
