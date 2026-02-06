from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _build_patch_indices(time_delta: torch.Tensor, patch_size: float) -> torch.Tensor:
    """
    Compute patch ids by cumulative time.
    """
    if time_delta.dim() != 2:
        raise ValueError("time_delta must be [B,T]")
    t = torch.cumsum(time_delta, dim=1)
    patch_id = torch.floor(t / float(patch_size)).to(torch.long)
    return patch_id


class IrregularPatchEmbed(nn.Module):
    """
    Irregular patch embedding using time deltas to group events by time span.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        patch_size: float = 7.0,
        max_patches: Optional[int] = None,
        agg: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.patch_size = float(patch_size)
        self.max_patches = max_patches
        self.agg = agg
        self.proj = nn.Linear(self.input_dim, self.d_model)
        self.drop = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        time_delta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C]
            time_delta: [B, T] in days
        Returns:
            patches: [B, P, d_model]
            mask: [B, P] boolean
        """
        if x.dim() != 3:
            raise ValueError("x must be [B,T,C]")
        B, T, C = x.shape
        if C != self.input_dim:
            raise ValueError(f"input_dim mismatch: expected {self.input_dim}, got {C}")

        if time_delta is None:
            # fallback: single patch over all tokens
            pooled = x.mean(dim=1, keepdim=True)
            patches = self.drop(self.proj(pooled))
            mask = torch.ones((B, 1), device=x.device, dtype=torch.bool)
            return patches, mask

        patch_ids = _build_patch_indices(time_delta, self.patch_size)
        patches_list = []
        mask_list = []

        for b in range(B):
            ids = patch_ids[b]
            xb = x[b]
            uniq = torch.unique(ids, sorted=True)
            feats = []
            for pid in uniq.tolist():
                idx = (ids == pid).nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                if self.agg == "mean":
                    feats.append(xb[idx].mean(dim=0))
                elif self.agg == "sum":
                    feats.append(xb[idx].sum(dim=0))
                else:
                    raise ValueError(f"Unknown agg={self.agg}")
            feats = torch.stack(feats, dim=0)
            valid_len = feats.shape[0]
            if self.max_patches is not None:
                if feats.shape[0] > self.max_patches:
                    feats = feats[-self.max_patches :]
                    valid_len = feats.shape[0]
                elif feats.shape[0] < self.max_patches:
                    pad = torch.zeros(
                        (self.max_patches - feats.shape[0], feats.shape[1]),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    feats = torch.cat([pad, feats], dim=0)
            patches_list.append(feats)
            if self.max_patches is not None:
                mask = torch.zeros((self.max_patches,), device=x.device, dtype=torch.bool)
                mask[-valid_len:] = True
            else:
                mask = torch.ones((feats.shape[0],), device=x.device, dtype=torch.bool)
            mask_list.append(mask)

        patches = torch.stack([self.proj(p) for p in patches_list], dim=0)
        patches = self.drop(patches)
        masks = torch.stack(mask_list, dim=0)
        return patches, masks


__all__ = ["IrregularPatchEmbed"]
