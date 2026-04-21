#!/usr/bin/env python3
"""Learnable Sparse MoE Trunk — replaces random projection backbone + barrier.

Motivation (Battle 2 evidence, 2026-04-20):
  - Random projection backbone HURTS 22/24 evaluation cells (91.7%)
  - No-trunk baseline (R²-IN → raw features → lane) uniformly better
  - Root cause: random Gaussian matrix + static statistics = information
    destruction with no learnable recovery path

Architecture:
  1. LearnableProjection: 2-layer MLP (input_dim → hidden → compact_dim)
     replaces the random Gaussian projection that destroyed information.
  2. ExpertBlock: lightweight MLP per expert (compact_dim → expert_dim)
  3. SparseMoETrunk: top-k expert selection with task-conditional gating
     + expert diversity penalty (orthogonal expert weight matrices)
  4. LearnableTrunkAdapter: sklearn-compatible wrapper (fit/transform API)
     that trains the PyTorch modules and exports NumPy lane features.

Training (during wrapper.fit()):
  - Mini-batch AdamW on auxiliary prediction loss + load balance + diversity
  - Auxiliary head predicts y from trunk output (BCE for binary, MSE otherwise)
  - After training, trunk output feeds into existing lane models (HGBC etc.)

Reference:
  - Switch Transformer (Fedus et al., JMLR 2022) — load balance loss
  - ST-MoE (Zoph et al., 2022) — router z-loss for stability
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("learnable_trunk")

# ─────────────────────────────────────────────────────────────────────
# R²-IN robust normalization (mirrors backbone._location_scale)
# ─────────────────────────────────────────────────────────────────────
_MAD_CONSISTENCY = np.float32(1.4826)


def _robust_loc_scale(
    arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Median / MAD location-scale (R²-IN)."""
    arr = np.asarray(arr, dtype=np.float32)
    loc = np.median(arr, axis=0).astype(np.float32)
    mad = np.median(np.abs(arr - loc), axis=0).astype(np.float32)
    scale = mad * _MAD_CONSISTENCY
    scale[scale < 1e-6] = np.float32(1.0)
    return loc, scale


# ─────────────────────────────────────────────────────────────────────
# Module 1: Learnable Projection
# ─────────────────────────────────────────────────────────────────────
class LearnableProjection(nn.Module):
    """2-layer MLP replacing the random Gaussian projection.

    Input:  R²-IN standardized features  (B, input_dim)
    Output: compact learned representation (B, compact_dim)
    """

    def __init__(
        self,
        input_dim: int,
        compact_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, compact_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
# Module 2: Expert Block
# ─────────────────────────────────────────────────────────────────────
class ExpertBlock(nn.Module):
    """Single lightweight expert: compact_dim → expert_dim."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
# Module 3: Sparse MoE Trunk
# ─────────────────────────────────────────────────────────────────────
TASK_NAMES: Tuple[str, ...] = ("binary", "funding", "investors")


class SparseMoETrunk(nn.Module):
    """Sparse Mixture-of-Experts trunk with task-conditional routing.

    Architecture flow::

        input (B, input_dim)
            → LearnableProjection → z (B, compact_dim)
            → N ExpertBlocks     → expert_outputs (B, N, expert_dim)
            → Gating(z ‖ task_id) → top-k selection
            → weighted sum       → output (B, expert_dim)

    Regularisation losses:
      - **Load balance**: variance of per-expert load → uniform usage
      - **Expert diversity**: pairwise cosine similarity of expert first-layer
        weights → orthogonal specialisation
    """

    def __init__(
        self,
        input_dim: int,
        compact_dim: int = 64,
        n_experts: int = 6,
        expert_dim: int = 32,
        top_k: int = 2,
        n_tasks: int = 3,
        projection_hidden: int = 128,
        expert_hidden: int = 64,
        load_balance_weight: float = 0.01,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.expert_dim = expert_dim
        self.top_k = min(top_k, n_experts)
        self.n_tasks = n_tasks
        self.load_balance_weight = load_balance_weight
        self.diversity_weight = diversity_weight

        # Shared learned projection
        self.projection = LearnableProjection(input_dim, compact_dim, projection_hidden)

        # Expert pool
        self.experts = nn.ModuleList(
            [ExpertBlock(compact_dim, expert_dim, expert_hidden) for _ in range(n_experts)]
        )

        # Task-conditional gating network
        self.gate = nn.Sequential(
            nn.Linear(compact_dim + n_tasks, n_experts * 2),
            nn.GELU(),
            nn.Linear(n_experts * 2, n_experts),
        )
        # Kaiming init for gate
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Per-task auxiliary prediction heads (for supervised trunk training)
        self.aux_heads = nn.ModuleDict(
            {name: nn.Linear(expert_dim, 1) for name in TASK_NAMES[:n_tasks]}
        )

    # ── forward ──────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        task_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) — R²-IN standardized features.
            task_id: (B, n_tasks) — one-hot task vector.

        Returns:
            dict with keys: output, z, gate_logits, gate_probs,
                load_balance_loss, expert_outputs, topk_indices, topk_weights
        """
        B = x.shape[0]

        # 1. Shared projection
        z = self.projection(x)  # (B, compact_dim)

        # 2. All expert outputs
        expert_outs = torch.stack(
            [expert(z) for expert in self.experts], dim=1
        )  # (B, n_experts, expert_dim)

        # 3. Task-conditional gating
        gate_input = torch.cat([z, task_id], dim=-1)  # (B, compact_dim + n_tasks)
        gate_logits = self.gate(gate_input)  # (B, n_experts)

        # 4. Top-k sparse selection
        topk_logits, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)  # (B, top_k)

        # Gather selected expert outputs
        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, self.expert_dim)
        selected = expert_outs.gather(1, idx_exp)  # (B, top_k, expert_dim)

        # Weighted combination
        output = (selected * topk_weights.unsqueeze(-1)).sum(dim=1)  # (B, expert_dim)

        # 5. Load balancing loss (Switch Transformer)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B, n_experts)
        load = gate_probs.mean(dim=0)  # average probability mass per expert
        load_balance_loss = self.load_balance_weight * (load.var() * self.n_experts)

        return {
            "output": output,
            "z": z,
            "gate_logits": gate_logits,
            "gate_probs": gate_probs,
            "load_balance_loss": load_balance_loss,
            "expert_outputs": expert_outs,
            "topk_indices": topk_idx,
            "topk_weights": topk_weights,
        }

    # ── expert diversity loss ────────────────────────────────────────
    def expert_diversity_loss(self) -> torch.Tensor:
        """Penalize cosine similarity between expert first-layer weights.

        For each pair (i, j) of experts, compute |cos(w_i, w_j)| and
        average.  Drives experts to learn orthogonal feature detectors.
        """
        device = next(self.parameters()).device
        penalty = torch.tensor(0.0, device=device)
        weights = []
        for expert in self.experts:
            w = expert.net[0].weight  # (hidden, compact_dim)
            w_flat = w.reshape(-1)
            w_norm = w_flat.norm() + 1e-8
            weights.append(w_flat / w_norm)

        n_pairs = 0
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                penalty = penalty + torch.dot(weights[i], weights[j]).abs()
                n_pairs += 1

        if n_pairs > 0:
            penalty = penalty / n_pairs

        return self.diversity_weight * penalty

    # ── auxiliary head ───────────────────────────────────────────────
    def predict_aux(self, output: torch.Tensor, task_name: str) -> torch.Tensor:
        """Auxiliary prediction for supervised trunk training."""
        return self.aux_heads[task_name](output).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────
# Module 4: Sklearn-compatible Adapter
# ─────────────────────────────────────────────────────────────────────
class LearnableTrunkAdapter:
    """Sklearn-compatible wrapper: fit(X, y, target) → transform(X, target).

    Trains :class:`SparseMoETrunk` via mini-batch AdamW during ``fit()``,
    then exports trunk output as NumPy arrays for downstream lane models
    (HGBC, HGBR, NumPy MLP, etc.).
    """

    TASK_TO_INDEX: Dict[str, int] = {"binary": 0, "funding": 1, "investors": 2}
    TARGET_TO_TASK: Dict[str, str] = {
        "is_funded": "binary",
        "funding_raised_usd": "funding",
        "investors_count": "investors",
    }

    def __init__(
        self,
        compact_dim: int = 64,
        n_experts: int = 6,
        expert_dim: int = 32,
        top_k: int = 2,
        projection_hidden: int = 128,
        expert_hidden: int = 64,
        load_balance_weight: float = 0.01,
        diversity_weight: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 30,
        batch_size: int = 4096,
        max_train_rows: int = 200_000,
        random_state: int = 42,
        device: str = "cpu",
    ):
        self.compact_dim = compact_dim
        self.n_experts = n_experts
        self.expert_dim = expert_dim
        self.top_k = top_k
        self.projection_hidden = projection_hidden
        self.expert_hidden = expert_hidden
        self.load_balance_weight = load_balance_weight
        self.diversity_weight = diversity_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_train_rows = max_train_rows
        self.random_state = random_state
        self.device = device

        self._model: SparseMoETrunk | None = None
        self._loc: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._fitted = False

    # ── helpers ──────────────────────────────────────────────────────
    def _task_one_hot(self, task_name: str, n: int) -> torch.Tensor:
        idx = self.TASK_TO_INDEX.get(task_name, 0)
        t = torch.zeros(n, len(self.TASK_TO_INDEX), device=self.device)
        t[:, idx] = 1.0
        return t

    # ── fit ──────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_name: str = "funding_raised_usd",
    ) -> "LearnableTrunkAdapter":
        task_name = self.TARGET_TO_TASK.get(target_name, "funding")
        is_binary = task_name == "binary"

        # R²-IN standardization
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self._loc, self._scale = _robust_loc_scale(X)
        X_std = (X - self._loc) / self._scale

        # Subsample if too large
        n = X_std.shape[0]
        if n > self.max_train_rows:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n, self.max_train_rows, replace=False)
            X_std = X_std[idx]
            y = y[idx]
            n = self.max_train_rows

        input_dim = X_std.shape[1]

        # Build model
        self._model = SparseMoETrunk(
            input_dim=input_dim,
            compact_dim=self.compact_dim,
            n_experts=self.n_experts,
            expert_dim=self.expert_dim,
            top_k=self.top_k,
            projection_hidden=self.projection_hidden,
            expert_hidden=self.expert_hidden,
            load_balance_weight=self.load_balance_weight,
            diversity_weight=self.diversity_weight,
        ).to(self.device)

        # Tensors
        X_t = torch.from_numpy(X_std).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)
        task_id = self._task_one_hot(task_name, n)

        # Normalize y for regression (not binary)
        if not is_binary:
            y_mean = y_t.mean()
            y_std = y_t.std().clamp(min=1e-8)
            y_norm = (y_t - y_mean) / y_std
        else:
            y_norm = y_t

        # Training loop
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.n_epochs,
            eta_min=self.lr * 0.01,
        )

        self._model.train()
        torch.manual_seed(self.random_state)

        best_loss = float("inf")
        patience_counter = 0
        patience_limit = 8

        for epoch in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                bi = perm[start:end]
                xb, yb, tb = X_t[bi], y_norm[bi], task_id[bi]

                result = self._model(xb, tb)
                pred = self._model.predict_aux(result["output"], task_name)

                if is_binary:
                    task_loss = F.binary_cross_entropy_with_logits(pred, yb)
                else:
                    task_loss = F.mse_loss(pred, yb)

                loss = (
                    task_loss
                    + result["load_balance_loss"]
                    + self._model.expert_diversity_loss()
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    logger.info(
                        "Early stop at epoch %d (patience=%d, best_loss=%.6f)",
                        epoch,
                        patience_limit,
                        best_loss,
                    )
                    break

        self._model.eval()
        self._fitted = True
        logger.info(
            "LearnableTrunk fitted: %d params, %d epochs, final_loss=%.6f",
            sum(p.numel() for p in self._model.parameters()),
            epoch + 1,
            avg_loss,
        )
        return self

    # ── transform ────────────────────────────────────────────────────
    def transform(
        self,
        X: np.ndarray,
        target_name: str = "funding_raised_usd",
    ) -> np.ndarray:
        """Export trunk output as NumPy array for downstream lane models."""
        if not self._fitted or self._model is None:
            raise ValueError("LearnableTrunkAdapter is not fitted")

        task_name = self.TARGET_TO_TASK.get(target_name, "funding")
        X_std = ((np.asarray(X, dtype=np.float32) - self._loc) / self._scale).astype(
            np.float32
        )

        self._model.eval()
        outputs = []
        chunk = 8192
        with torch.no_grad():
            X_t = torch.from_numpy(X_std).to(self.device)
            n = X_t.shape[0]
            task_id = self._task_one_hot(task_name, n)
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                result = self._model(X_t[start:end], task_id[start:end])
                outputs.append(result["output"].cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32)

    # ── fit_transform ────────────────────────────────────────────────
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_name: str = "funding_raised_usd",
    ) -> np.ndarray:
        self.fit(X, y, target_name)
        return self.transform(X, target_name)

    # ── describe ─────────────────────────────────────────────────────
    def describe(self) -> Dict[str, object]:
        if not self._fitted or self._model is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "input_dim": int(self._model.projection.net[0].in_features),
            "compact_dim": self.compact_dim,
            "n_experts": self.n_experts,
            "expert_dim": self.expert_dim,
            "top_k": self.top_k,
            "total_params": sum(p.numel() for p in self._model.parameters()),
            "trainable_params": sum(
                p.numel() for p in self._model.parameters() if p.requires_grad
            ),
        }
