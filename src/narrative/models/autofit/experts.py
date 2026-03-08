#!/usr/bin/env python3
"""
AutoFit v2 — Expert Model Implementations.

Each expert wraps the corresponding Block 3 model category and provides
a unified interface for the AutoFit pipeline.

Experts:
    TabularExpert       → LightGBM / CatBoost / XGBoost
    DeepTSExpert        → NHITS / NBEATS / TFT / DeepAR
    TransformerExpert   → PatchTST / iTransformer / TimesNet
    FoundationExpert    → Chronos / Moirai / TimesFM
    StatisticalExpert   → AutoARIMA / AutoETS / MSTL
    IrregularExpert     → GRU-D / SAITS

All experts implement:
    .fit(X_train, y_train, **kwargs) -> self
    .predict(X_test) -> np.ndarray
    .score(X_test, y_test) -> float   (MAE by default)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExpertBase(ABC):
    """Base class for all AutoFit experts."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self._model = None
        self._fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "ExpertBase":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute MAE on (X, y). Lower is better."""
        preds = self.predict(X)
        return float(np.mean(np.abs(y.values - preds)))

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class TabularExpert(ExpertBase):
    """
    Wrapper for tree-based tabular models.

    Uses the Block 3 model registry for actual model construction.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "TabularExpert":
        from narrative.block3.models.registry import get_model
        self._model = get_model(self.model_name)
        self._model.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError(f"TabularExpert({self.model_name}) not fitted")
        return self._model.predict(X)


class DeepTSExpert(ExpertBase):
    """Wrapper for deep time-series models (NHITS, NBEATS, TFT, DeepAR)."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "DeepTSExpert":
        from narrative.block3.models.registry import get_model
        self._model = get_model(self.model_name)
        self._model.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError(f"DeepTSExpert({self.model_name}) not fitted")
        return self._model.predict(X)


class TransformerExpert(ExpertBase):
    """Wrapper for transformer-SOTA models."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "TransformerExpert":
        from narrative.block3.models.registry import get_model
        self._model = get_model(self.model_name)
        self._model.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError(f"TransformerExpert({self.model_name}) not fitted")
        return self._model.predict(X)


class FoundationExpert(ExpertBase):
    """Wrapper for foundation models (zero-shot / few-shot)."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "FoundationExpert":
        from narrative.block3.models.registry import get_model
        self._model = get_model(self.model_name)
        self._model.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError(f"FoundationExpert({self.model_name}) not fitted")
        return self._model.predict(X)


class StatisticalExpert(ExpertBase):
    """Wrapper for classical statistical models."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "StatisticalExpert":
        from narrative.block3.models.registry import get_model
        self._model = get_model(self.model_name)
        self._model.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError(f"StatisticalExpert({self.model_name}) not fitted")
        return self._model.predict(X)


class IrregularExpert(ExpertBase):
    """Wrapper for irregular-time-series models (GRU-D, SAITS)."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "IrregularExpert":
        from narrative.block3.models.registry import get_model
        self._model = get_model(self.model_name)
        self._model.fit(X, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError(f"IrregularExpert({self.model_name}) not fitted")
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

EXPERT_CLASSES = {
    "tabular": TabularExpert,
    "deep_ts": DeepTSExpert,
    "transformer": TransformerExpert,
    "foundation": FoundationExpert,
    "statistical": StatisticalExpert,
    "irregular": IrregularExpert,
}


def create_expert(
    expert_name: str,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
) -> ExpertBase:
    """
    Factory: create an expert instance by category.

    Parameters
    ----------
    expert_name : str
        One of: tabular, deep_ts, transformer, foundation, statistical, irregular
    model_name : str
        Specific model within the expert category.
    config : dict
        Model-specific configuration overrides.
    """
    cls = EXPERT_CLASSES.get(expert_name)
    if cls is None:
        raise ValueError(
            f"Unknown expert '{expert_name}'. "
            f"Available: {list(EXPERT_CLASSES.keys())}"
        )
    return cls(model_name=model_name, config=config)
