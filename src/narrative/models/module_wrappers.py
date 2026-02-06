from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from narrative.models.mamba_encoder import MambaEncoder
from narrative.models.fusion import build_fusion
from narrative.models.ssm_encoder_v2 import SSMEncoderV2
from narrative.models.fusion import build_fusion
from narrative.models.ssm_encoder_v2 import SSMEncoderV2


class NonStatAdapter(nn.Module):
    """RevIN-style per-sample normalization (no inverse)."""

    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, n_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, n_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(self.eps)
        out = (x - mean) / std
        if self.affine:
            out = out * self.gamma + self.beta
        return out


class MultiScaleBranch(nn.Module):
    """Multi-scale temporal smoothing + optional FFT branch."""

    def __init__(self, n_features: int, window_sizes: Sequence[int] = (3, 7, 15), use_fft: bool = False):
        super().__init__()
        self.window_sizes = [int(w) for w in window_sizes]
        self.use_fft = bool(use_fft)
        n_branches = 1 + len(self.window_sizes) + (1 if self.use_fft else 0)
        self.proj = nn.Linear(n_features * n_branches, n_features)

    def _avg_pool(self, x: torch.Tensor, k: int) -> torch.Tensor:
        # x: [B,T,C] -> [B,C,T]
        xt = x.transpose(1, 2)
        pad_left = k // 2
        pad_right = k - 1 - pad_left
        xt = F.pad(xt, (pad_left, pad_right), mode="replicate")
        yt = F.avg_pool1d(xt, kernel_size=k, stride=1)
        return yt.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for k in self.window_sizes:
            feats.append(self._avg_pool(x, k))
        if self.use_fft:
            fft_mag = torch.fft.rfft(x, dim=1).abs().mean(dim=1, keepdim=True)
            fft_mag = fft_mag.expand(-1, x.shape[1], -1)
            feats.append(fft_mag)
        fused = torch.cat(feats, dim=-1)
        return self.proj(fused)


class SSMEncoderAdapter(nn.Module):
    """SSM encoder branch (Mamba or V2) projected back to input space."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        variant: str = "mamba",
        chunk_size: Optional[int] = None,
        causal: bool = True,
        kernel_size: int = 5,
    ):
        super().__init__()
        if str(variant).lower() == "v2":
            self.encoder = SSMEncoderV2(
                input_dim=n_features,
                d_model=d_model,
                n_layers=n_layers,
                dropout=dropout,
                chunk_size=chunk_size,
                causal=causal,
                kernel_size=kernel_size,
            )
        else:
            self.encoder = MambaEncoder(input_dim=n_features, d_model=d_model, n_layers=n_layers, dropout=dropout)
        self.proj = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder(x)
        bias = self.proj(pooled).unsqueeze(1)
        return x + bias


@dataclass
class ModuleFlags:
    nonstat: bool = False
    multiscale: bool = False
    multiscale_fft: bool = False
    ssm: bool = False


class ModelWithModules(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        n_features: int,
        *,
        flags: ModuleFlags,
        ssm_dim: int = 128,
        ssm_layers: int = 2,
        dropout: float = 0.1,
        ssm_variant: str = "mamba",
        ssm_chunk_size: Optional[int] = None,
        ssm_causal: bool = True,
        ssm_kernel_size: int = 5,
    ):
        """Initialize the wrapped model with optional module adapters."""
        super().__init__()
        self.base_model = base_model
        self.nonstat = NonStatAdapter(n_features) if flags.nonstat else None
        self.multiscale = (
            MultiScaleBranch(n_features, window_sizes=multiscale_windows, use_fft=flags.multiscale_fft)
            if flags.multiscale
            else None
        )
        self.ssm = (
            SSMEncoderAdapter(
                n_features,
                d_model=ssm_dim,
                n_layers=ssm_layers,
                dropout=dropout,
                variant=ssm_variant,
                chunk_size=ssm_chunk_size,
                causal=ssm_causal,
                kernel_size=ssm_kernel_size,
            )
            if flags.ssm
            else None
        )

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.nonstat is not None:
            x = self.nonstat(x)
        if self.multiscale is not None:
            x = self.multiscale(x)
        if self.ssm is not None:
            x = self.ssm(x)
        return self.base_model(x, *args, **kwargs)


class ModelWithFusion(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        fusion_type: str,
        d_model: int,
        edgar_dim: int,
        fusion_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.edgar_dim = int(edgar_dim)
        self.fusion = build_fusion(
            fusion_type,
            d_model=int(d_model),
            edgar_dim=self.edgar_dim,
            n_heads=int(fusion_heads),
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor, edgar_x: Optional[torch.Tensor] = None, edgar_mask: Optional[torch.Tensor] = None):
        if self.fusion is None:
            return self.base_model(x)
        if edgar_x is None:
            edgar_x = torch.zeros((x.shape[0], x.shape[1], self.edgar_dim), device=x.device, dtype=x.dtype)
        fused = self.fusion(x, edgar_x=edgar_x, edgar_mask=edgar_mask, attn_mask=None)
        return self.base_model(fused.x)


class ModelWithFusion(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        fusion_type: str,
        d_model: int,
        edgar_dim: int,
        fusion_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.edgar_dim = int(edgar_dim)
        self.fusion = build_fusion(
            fusion_type,
            d_model=int(d_model),
            edgar_dim=self.edgar_dim,
            n_heads=int(fusion_heads),
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor, edgar_x: Optional[torch.Tensor] = None, edgar_mask: Optional[torch.Tensor] = None):
        if self.fusion is None:
            return self.base_model(x)
        if edgar_x is None:
            edgar_x = torch.zeros((x.shape[0], x.shape[1], self.edgar_dim), device=x.device, dtype=x.dtype)
        fused = self.fusion(x, edgar_x=edgar_x, edgar_mask=edgar_mask, attn_mask=None)
        return self.base_model(fused.x)


__all__ = [
    "ModuleFlags",
    "NonStatAdapter",
    "MultiScaleBranch",
    "SSMEncoderAdapter",
    "ModelWithModules",
    "ModelWithFusion",
]
