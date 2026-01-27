from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DimensionMoERouter(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 4):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), self.num_experts)
        self.gate = nn.Linear(input_dim, self.num_experts)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, D]
        Returns:
            gate_weights: [B, E]
            topk_indices: [B, K]
            topk_weights: [B, K]
            load_balance_loss: scalar
            sparsity_loss: scalar
        """
        logits = self.gate(x)
        gate_weights = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)

        dispatch = torch.zeros_like(gate_weights)
        dispatch.scatter_(1, topk_indices, 1.0)
        load = dispatch.mean(dim=0)
        importance = gate_weights.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(importance * load)

        sparsity_loss = 1.0 - topk_weights.sum(dim=-1).mean()

        return {
            "gate_weights": gate_weights,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            "load_balance_loss": load_balance_loss,
            "sparsity_loss": sparsity_loss,
        }


class DimensionMoE(nn.Module):
    """
    Dimension-wise MoE: one expert per bias dimension.
    """

    def __init__(
        self,
        input_dim: int,
        num_dims: int,
        hidden_dim: int = 128,
        top_k: int = 4,
    ):
        super().__init__()
        self.num_dims = int(num_dims)
        self.router = DimensionMoERouter(input_dim, self.num_dims, top_k=top_k)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.num_dims)
            ]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, D] or [B, D]
        Returns:
            dim_scores: [B, num_dims]
            gating_diagnostics: dict
        """
        if x.dim() == 3:
            x_pool = x.mean(dim=1)
        elif x.dim() == 2:
            x_pool = x
        else:
            raise ValueError("x must be [B,T,D] or [B,D]")

        gate = self.router(x_pool)
        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(x_pool))
        expert_outs = torch.cat(expert_outs, dim=-1)  # [B, E]

        dim_scores = expert_outs * gate["gate_weights"]
        return {
            "dim_scores": dim_scores,
            "gating_diagnostics": gate,
        }


__all__ = ["DimensionMoERouter", "DimensionMoE"]
