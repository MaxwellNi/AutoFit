from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FusionOutput(tuple):
    __slots__ = ()
    _fields = ("x", "attn_mask")

    def __new__(cls, x: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        return tuple.__new__(cls, (x, attn_mask))

    @property
    def x(self) -> torch.Tensor:
        return self[0]

    @property
    def attn_mask(self) -> Optional[torch.Tensor]:
        return self[1]


class BaseFusion(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        edgar_x: Optional[torch.Tensor] = None,
        edgar_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> FusionOutput:
        return FusionOutput(x, attn_mask)


def _expand_edgar(edgar_x: torch.Tensor, seq_len: int) -> torch.Tensor:
    if edgar_x.dim() == 2:
        return edgar_x.unsqueeze(1).expand(-1, seq_len, -1)
    return edgar_x


def _to_time_mask(edgar_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if edgar_mask is None:
        return None
    if edgar_mask.dim() == 3:
        return edgar_mask.all(dim=-1)
    if edgar_mask.dim() == 2:
        return edgar_mask
    return None


class ConcatFusion(BaseFusion):
    def __init__(self, d_model: int, edgar_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_model + edgar_dim, d_model)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x, edgar_x=None, edgar_mask=None, attn_mask=None) -> FusionOutput:
        if edgar_x is None:
            return FusionOutput(x, attn_mask)
        edgar_x = _expand_edgar(edgar_x, x.shape[1])
        out = self.drop(self.proj(torch.cat([x, edgar_x], dim=-1)))
        return FusionOutput(out, attn_mask)


class CrossAttentionFusion(BaseFusion):
    def __init__(self, d_model: int, edgar_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(edgar_dim, d_model)
        self.v_proj = nn.Linear(edgar_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, int(n_heads), dropout=float(dropout), batch_first=True)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x, edgar_x=None, edgar_mask=None, attn_mask=None) -> FusionOutput:
        if edgar_x is None:
            return FusionOutput(x, attn_mask)
        edgar_x = _expand_edgar(edgar_x, x.shape[1])
        key_padding_mask = _to_time_mask(edgar_mask)
        q = self.q_proj(x)
        k = self.k_proj(edgar_x)
        v = self.v_proj(edgar_x)
        attn_out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        out = x + self.drop(self.out(attn_out))
        return FusionOutput(out, attn_mask)


class BridgeTokenFusion(BaseFusion):
    def __init__(self, d_model: int, edgar_dim: int, dropout: float = 0.1):
        super().__init__()
        self.bridge = nn.Parameter(torch.zeros(1, 1, d_model))
        self.edgar_proj = nn.Linear(edgar_dim, d_model)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x, edgar_x=None, edgar_mask=None, attn_mask=None) -> FusionOutput:
        if edgar_x is None:
            return FusionOutput(x, attn_mask)
        edgar_x = _expand_edgar(edgar_x, x.shape[1])
        time_mask = _to_time_mask(edgar_mask)
        if time_mask is None:
            weights = torch.ones((x.shape[0], x.shape[1], 1), device=x.device, dtype=x.dtype)
        else:
            weights = (~time_mask).to(x.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        summary = (self.edgar_proj(edgar_x) * weights).sum(dim=1, keepdim=True) / denom
        bridge = self.bridge.expand(x.shape[0], 1, -1) + summary
        bridge = self.drop(bridge)

        x2 = torch.cat([x, bridge], dim=1)
        if attn_mask is not None:
            extra = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            attn_mask = torch.cat([attn_mask, extra], dim=1)
        return FusionOutput(x2, attn_mask)


class FiLMFusion(BaseFusion):
    def __init__(self, d_model: int, edgar_dim: int):
        super().__init__()
        self.gamma = nn.Linear(edgar_dim, d_model)
        self.beta = nn.Linear(edgar_dim, d_model)

    def forward(self, x, edgar_x=None, edgar_mask=None, attn_mask=None) -> FusionOutput:
        if edgar_x is None:
            return FusionOutput(x, attn_mask)
        edgar_x = _expand_edgar(edgar_x, x.shape[1])
        gamma = self.gamma(edgar_x)
        beta = self.beta(edgar_x)
        time_mask = _to_time_mask(edgar_mask)
        if time_mask is not None:
            keep = (~time_mask).to(x.dtype).unsqueeze(-1)
            gamma = gamma * keep
            beta = beta * keep
        out = x * (1.0 + gamma) + beta
        return FusionOutput(out, attn_mask)


def build_fusion(
    fusion_type: str,
    *,
    d_model: int,
    edgar_dim: int,
    n_heads: int = 4,
    dropout: float = 0.1,
) -> Optional[BaseFusion]:
    if not fusion_type or fusion_type.lower() in {"none", "null", "off"}:
        return None
    if edgar_dim <= 0:
        return None

    f = fusion_type.lower()
    if f in {"concat", "concat_fusion"}:
        return ConcatFusion(d_model=d_model, edgar_dim=edgar_dim, dropout=dropout)
    if f in {"cross_attn", "cross_attention", "cross-attention"}:
        return CrossAttentionFusion(d_model=d_model, edgar_dim=edgar_dim, n_heads=n_heads, dropout=dropout)
    if f in {"bridge", "bridge_token", "bridge-token"}:
        return BridgeTokenFusion(d_model=d_model, edgar_dim=edgar_dim, dropout=dropout)
    if f in {"film", "temporal_fm", "temporal-fm"}:
        return FiLMFusion(d_model=d_model, edgar_dim=edgar_dim)
    raise ValueError(f"Unknown fusion_type={fusion_type}")


__all__ = [
    "BaseFusion",
    "ConcatFusion",
    "CrossAttentionFusion",
    "BridgeTokenFusion",
    "FiLMFusion",
    "build_fusion",
]
