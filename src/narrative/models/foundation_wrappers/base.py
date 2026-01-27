from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch


class FoundationModelWrapper(ABC):
    """
    Unified interface for foundation models (Chronos / Lag-Llama / Moirai).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None

    @abstractmethod
    def load(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: torch.Tensor, *, horizon: int = 1) -> torch.Tensor:
        raise NotImplementedError

    def to(self, device: str) -> "FoundationModelWrapper":
        self.device = device
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(device)
        return self

    def metadata(self) -> Dict[str, str]:
        return {"device": self.device, "loaded": str(self.model is not None)}
