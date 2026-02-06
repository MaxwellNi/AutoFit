"""
Local model registry for building models.

This module provides a registry for building time series models.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class DLinear(nn.Module):
    """Simple DLinear baseline model."""
    
    def __init__(self, seq_len: int, enc_in: int, pred_len: int = 1, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.pred_len = pred_len
        
        self.linear = nn.Linear(seq_len * enc_in, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        return self.linear(x_flat)


class PatchTST(nn.Module):
    """Simple PatchTST-like model."""
    
    def __init__(
        self,
        seq_len: int,
        enc_in: int,
        pred_len: int = 1,
        patch_len: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.pred_len = pred_len
        self.patch_len = patch_len
        
        # Simple implementation
        n_patches = max(1, seq_len // patch_len)
        self.patch_embed = nn.Linear(patch_len * enc_in, 64)
        self.output = nn.Linear(n_patches * 64, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        batch_size, seq_len, channels = x.shape
        n_patches = max(1, seq_len // self.patch_len)
        
        # Reshape to patches
        if seq_len >= self.patch_len:
            x_patched = x[:, :n_patches * self.patch_len, :].reshape(
                batch_size, n_patches, self.patch_len * channels
            )
        else:
            # Pad if needed
            pad_len = self.patch_len - seq_len
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
            x_patched = x_padded.reshape(batch_size, 1, self.patch_len * channels)
            n_patches = 1
        
        # Embed patches
        patches = self.patch_embed(x_patched)  # [B, n_patches, 64]
        patches_flat = patches.reshape(batch_size, -1)
        
        return self.output(patches_flat)


MODEL_REGISTRY = {
    "dlinear": DLinear,
    "patchtst": PatchTST,
}


def build_local_model(
    name: str,
    seq_len: int,
    enc_in: int,
    pred_len: int = 1,
    **kwargs,
) -> nn.Module:
    """
    Build a model from the local registry.
    
    Args:
        name: Model name
        seq_len: Sequence length
        enc_in: Number of input features
        pred_len: Prediction length
        **kwargs: Additional model arguments
        
    Returns:
        Instantiated model
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_cls = MODEL_REGISTRY[name]
    return model_cls(seq_len=seq_len, enc_in=enc_in, pred_len=pred_len, **kwargs)


__all__ = [
    "MODEL_REGISTRY",
    "build_local_model",
    "DLinear",
    "PatchTST",
]
