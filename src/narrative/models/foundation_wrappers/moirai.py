from __future__ import annotations

import torch
import torch.nn as nn

from .base import FoundationModelWrapper


class _MoiraiMock(nn.Module):
    def __init__(self, input_dim: int, horizon: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(input_dim, num_heads=2, batch_first=True)
        self.head = nn.Linear(input_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(x, x, x)
        return self.head(h.mean(dim=1))


class MoiraiWrapper(FoundationModelWrapper):
    """
    Moirai wrapper (mocked unless external dependency is available).
    """

    def __init__(self, device: str = "cpu", input_dim: int = 4):
        super().__init__(device=device)
        self.input_dim = int(input_dim)
        self._horizon = 1

    def load(self, **kwargs) -> None:
        horizon = int(kwargs.get("horizon", 1))
        self._horizon = horizon
        self.model = _MoiraiMock(self.input_dim, horizon).to(self.device)

    def predict(self, x: torch.Tensor, *, horizon: int = 1) -> torch.Tensor:
        if self.model is None:
            self.load(horizon=horizon)
        if horizon != self._horizon:
            self.load(horizon=horizon)
        x = x.to(self.device)
        return self.model(x)
