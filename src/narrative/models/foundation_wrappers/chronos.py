from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import FoundationModelWrapper


class _ChronosMock(nn.Module):
    def __init__(self, input_dim: int, horizon: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C] -> [B,horizon]
        return self.proj(x.mean(dim=1))


class ChronosWrapper(FoundationModelWrapper):
    """
    Chronos wrapper (mocked unless external dependency is available).
    """

    def __init__(self, device: str = "cpu", input_dim: int = 4):
        super().__init__(device=device)
        self.input_dim = int(input_dim)
        self._horizon = 1

    def load(self, **kwargs) -> None:
        horizon = int(kwargs.get("horizon", 1))
        self._horizon = horizon
        # Placeholder mock; replace with real Chronos loading when dependency available.
        self.model = _ChronosMock(self.input_dim, horizon).to(self.device)

    def predict(self, x: torch.Tensor, *, horizon: int = 1) -> torch.Tensor:
        if self.model is None:
            self.load(horizon=horizon)
        if horizon != self._horizon:
            self.load(horizon=horizon)
        x = x.to(self.device)
        return self.model(x)
