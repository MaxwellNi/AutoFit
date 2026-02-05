#!/usr/bin/env python3
"""
Model Registry for Block 3.

Unified interface for all model categories:
- Traditional ML (sklearn-compatible)
- Statistical (StatsForecast)
- Deep Classical (NBEATS, NHITS, TFT, DeepAR)
- Transformer SOTA (PatchTST, iTransformer, TimesNet)
- Foundation (TimesFM, Chronos, Moirai)
- Irregular-aware (GRU-D, SAITS)

Usage:
    from src.narrative.block3.models import get_model, list_models
    
    model = get_model("LightGBM", n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import ModelBase, ModelConfig
from .traditional_ml import TRADITIONAL_ML_MODELS, get_traditional_model
from .statistical import STATISTICAL_MODELS, get_statistical_model
from .deep_models import DEEP_MODELS, FOUNDATION_MODELS, get_deep_model
from .irregular_models import IRREGULAR_MODELS, get_irregular_model


# ============================================================================
# Unified Registry
# ============================================================================

MODEL_CATEGORIES = {
    "statistical": list(STATISTICAL_MODELS.keys()),
    "ml_tabular": list(TRADITIONAL_ML_MODELS.keys()),
    "deep_classical": ["NBEATS", "NHITS", "TFT", "DeepAR"],
    "transformer_sota": ["PatchTST", "iTransformer", "TimesNet", "TSMixer"],
    "foundation": list(FOUNDATION_MODELS.keys()),
    "irregular_aware": list(IRREGULAR_MODELS.keys()),
}

# Flattened registry for direct lookup
_ALL_MODELS = {}
_ALL_MODELS.update(TRADITIONAL_ML_MODELS)
_ALL_MODELS.update(STATISTICAL_MODELS)
_ALL_MODELS.update(DEEP_MODELS)
_ALL_MODELS.update(FOUNDATION_MODELS)
_ALL_MODELS.update(IRREGULAR_MODELS)


def get_model(name: str, **kwargs) -> ModelBase:
    """
    Get a model by name.
    
    Args:
        name: Model name (case-sensitive)
        **kwargs: Model-specific parameters
    
    Returns:
        ModelBase instance
    
    Example:
        >>> model = get_model("LightGBM", n_estimators=100)
        >>> model = get_model("AutoARIMA")
        >>> model = get_model("PatchTST", input_size=30)
    """
    if name not in _ALL_MODELS:
        raise ValueError(
            f"Unknown model: {name}. "
            f"Use list_models() to see available models."
        )
    
    return _ALL_MODELS[name](**kwargs)


def get_models_by_category(category: str, **kwargs) -> List[ModelBase]:
    """
    Get all models in a category.
    
    Args:
        category: One of 'statistical', 'ml_tabular', 'deep_classical',
                  'transformer_sota', 'foundation', 'irregular_aware'
        **kwargs: Model-specific parameters (applied to all)
    
    Returns:
        List of ModelBase instances
    """
    if category not in MODEL_CATEGORIES:
        raise ValueError(
            f"Unknown category: {category}. "
            f"Available: {list(MODEL_CATEGORIES.keys())}"
        )
    
    return [get_model(name, **kwargs) for name in MODEL_CATEGORIES[category]]


def list_models() -> Dict[str, List[str]]:
    """
    List all available models by category.
    
    Returns:
        Dict mapping category -> list of model names
    """
    return MODEL_CATEGORIES.copy()


def list_all_models() -> List[str]:
    """
    List all model names (flat list).
    
    Returns:
        List of all model names
    """
    return list(_ALL_MODELS.keys())


def get_model_info(name: str) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        name: Model name
    
    Returns:
        Dict with model metadata
    """
    if name not in _ALL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    
    # Create instance to get config
    model = _ALL_MODELS[name]()
    config = model.config
    
    # Find category
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
    """
    Check if a model's dependencies are available.
    
    Args:
        name: Model name
    
    Returns:
        True if model can be instantiated
    """
    if name not in _ALL_MODELS:
        return False
    
    try:
        model = get_model(name)
        return True
    except ImportError:
        return False


def get_available_models() -> Dict[str, List[str]]:
    """
    Get models that are actually available (dependencies installed).
    
    Returns:
        Dict mapping category -> list of available model names
    """
    available = {}
    for category, models in MODEL_CATEGORIES.items():
        available[category] = [
            m for m in models if check_model_available(m)
        ]
    return available


# ============================================================================
# Model Selection Helpers
# ============================================================================

def get_models_for_task(task_type: str, **kwargs) -> List[ModelBase]:
    """
    Get appropriate models for a task type.
    
    Args:
        task_type: One of 'classification', 'regression', 'forecasting'
        **kwargs: Model parameters
    
    Returns:
        List of suitable models
    """
    if task_type == "classification":
        # Classification models
        return [
            get_model("LogisticRegression", **kwargs),
            get_model("RandomForest", **kwargs),
            get_model("HistGradientBoosting", **kwargs),
            get_model("LightGBM", **kwargs),
            get_model("XGBoost", **kwargs),
        ]
    
    elif task_type == "regression":
        # Regression models (tabular)
        return [
            get_model("Ridge", **kwargs),
            get_model("Lasso", **kwargs),
            get_model("RandomForest", **kwargs),
            get_model("HistGradientBoosting", **kwargs),
            get_model("LightGBM", **kwargs),
            get_model("XGBoost", **kwargs),
            get_model("CatBoost", **kwargs),
        ]
    
    elif task_type == "forecasting":
        # Time series forecasting models
        models = []
        
        # Statistical
        for name in ["AutoARIMA", "ETS", "Theta", "MSTL"]:
            try:
                models.append(get_model(name, **kwargs))
            except (ImportError, Exception):
                pass
        
        # Deep (if available)
        for name in ["NBEATS", "NHITS", "TFT", "DeepAR"]:
            try:
                models.append(get_model(name, **kwargs))
            except (ImportError, Exception):
                pass
        
        return models
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def get_baseline_models(**kwargs) -> List[ModelBase]:
    """
    Get baseline models for benchmark comparison.
    
    Returns:
        List of baseline models
    """
    return [
        get_model("MeanPredictor", **kwargs),
        get_model("SeasonalNaive", **kwargs),
    ]


# ============================================================================
# Model Presets
# ============================================================================

PRESET_CONFIGS = {
    "smoke_test": {
        # Fast models for testing
        "models": ["MeanPredictor", "Ridge", "LightGBM"],
        "params": {"n_estimators": 10, "max_depth": 3},
    },
    "quick": {
        # Quick benchmark
        "models": ["MeanPredictor", "SeasonalNaive", "Ridge", "LightGBM", "AutoARIMA"],
        "params": {"n_estimators": 50},
    },
    "standard": {
        # Standard benchmark
        "models": [
            # Baselines
            "MeanPredictor", "SeasonalNaive",
            # ML
            "Ridge", "Lasso", "RandomForest", "LightGBM", "XGBoost",
            # Statistical
            "AutoARIMA", "ETS", "Theta",
        ],
        "params": {"n_estimators": 100},
    },
    "comprehensive": {
        # Full benchmark
        "models": list(_ALL_MODELS.keys()),
        "params": {},
    },
}


def get_preset_models(preset: str) -> List[ModelBase]:
    """
    Get models from a preset configuration.
    
    Args:
        preset: One of 'smoke_test', 'quick', 'standard', 'comprehensive'
    
    Returns:
        List of models
    """
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
# Module exports
# ============================================================================

__all__ = [
    # Main functions
    "get_model",
    "get_models_by_category",
    "list_models",
    "list_all_models",
    "get_model_info",
    "check_model_available",
    "get_available_models",
    
    # Task-specific
    "get_models_for_task",
    "get_baseline_models",
    
    # Presets
    "get_preset_models",
    "PRESET_CONFIGS",
    
    # Categories
    "MODEL_CATEGORIES",
    
    # Base class
    "ModelBase",
    "ModelConfig",
]
