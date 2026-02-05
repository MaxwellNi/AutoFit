#!/usr/bin/env python3
"""
Block 3 Models Package.

Provides unified access to all model categories:
- Traditional ML (sklearn-compatible)
- Statistical (StatsForecast)
- Deep Classical (NBEATS, NHITS, TFT, DeepAR)
- Transformer SOTA (PatchTST, iTransformer, TimesNet)
- Foundation (TimesFM, Chronos, Moirai)
- Irregular-aware (GRU-D, SAITS)

Usage:
    from src.narrative.block3.models import get_model, list_models
    
    # Get a specific model
    model = get_model("LightGBM", n_estimators=100)
    
    # List all available models
    print(list_models())
    
    # Get models by category
    from src.narrative.block3.models import get_models_by_category
    ml_models = get_models_by_category("ml_tabular")
"""
from .base import ModelBase, ModelConfig
from .registry import (
    get_model,
    get_models_by_category,
    list_models,
    list_all_models,
    get_model_info,
    check_model_available,
    get_available_models,
    get_models_for_task,
    get_baseline_models,
    get_preset_models,
    PRESET_CONFIGS,
    MODEL_CATEGORIES,
)

__all__ = [
    # Base classes
    "ModelBase",
    "ModelConfig",
    
    # Main registry functions
    "get_model",
    "get_models_by_category",
    "list_models",
    "list_all_models",
    "get_model_info",
    "check_model_available",
    "get_available_models",
    
    # Task-specific helpers
    "get_models_for_task",
    "get_baseline_models",
    
    # Presets
    "get_preset_models",
    "PRESET_CONFIGS",
    
    # Constants
    "MODEL_CATEGORIES",
]
