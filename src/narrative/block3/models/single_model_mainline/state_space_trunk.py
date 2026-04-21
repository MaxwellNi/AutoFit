#!/usr/bin/env python3
"""State-Space Trunk (S5-style diagonal linear-time-invariant SSM) for the
single-model mainline.

Motivation — 2026-04-22 verdict (`mainline_trunk_collapse_verdict_20260421`):
    Current MoE-ish trunks (SparseMoETrunk, SequentialMoETrunk) collapse
    under the hurdle–tail regime: on mn_v2_naked_t1 / mn_v2_full_t2 / t3
    the audit gate `audit_mainline_degeneracy_gate.py` returns
    7/7 triples failing with cell_fail_rate=1.00 and identical MAE across
    horizons — the signature of a constant-predictor collapse on the
    funding target (MAE = 647893.017) and the investors target
    (MAE = 275.96). The verdict demands swapping the trunk itself.

This module implements a minimal, dependency-free S5-style diagonal
state-space trunk that (i) has well-studied stability properties via
HiPPO-LegS initialisation, (ii) does not require `mamba-ssm`, and
(iii) exposes the same sklearn-style `fit / transform` interface as
`LearnableTrunkAdapter` so that it is a drop-in replacement in
`MainlineTrunkAdapter`.

References:
    Gu, Goel, Re. "Efficiently Modeling Long Sequences with Structured
    State Spaces." ICLR 2022 (S4).
    Smith, Warrington, Linderman. "Simplified State Space Layers for
    Sequence Modeling." ICLR 2023 (S5).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("state_space_trunk")


# ─────────────────────────────────────────────────────────────────────────────
# S5-style diagonal SSM block
# ─────────────────────────────────────────────────────────────────────────────
class DiagonalSSMBlock(nn.Module):
    """One S5-style diagonal state-space block.

    The per-feature recurrence is
        h_{t+1} = diag(a) * h_t + B * u_t,
        y_t    = C * h_t + D * u_t,
    with complex-diagonal transition `a ∈ C^d_state` parameterised via
    HiPPO-LegS-like log-real + imaginary initialisation.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # HiPPO-LegS diagonal init (real part negative, imag part linear).
        log_neg_real = -0.5 * torch.ones(d_state)  # a_re = -exp(log_neg_real) < 0
        imag = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.log_neg_real = nn.Parameter(log_neg_real)
        self.imag = nn.Parameter(imag)

        self.B = nn.Parameter(torch.randn(d_state, d_model) / (d_model ** 0.5))
        self.C = nn.Parameter(torch.randn(d_model, d_state) / (d_state ** 0.5))
        self.D = nn.Parameter(torch.zeros(d_model))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @property
    def a(self) -> torch.Tensor:
        """Stable complex eigenvalues: real part strictly negative."""
        return torch.complex(-torch.exp(self.log_neg_real), self.imag)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Apply the block along the sequence dim.

        Parameters
        ----------
        u : Tensor of shape (B, L, d_model)
        """
        B_size, L, d = u.shape
        assert d == self.d_model

        # Discretise via zero-order hold with unit step delta_t = 1.
        # a_bar = exp(a), B_bar = (exp(a) - 1) / a * B
        a = self.a  # (d_state,) complex
        a_bar = torch.exp(a)
        b_bar = ((a_bar - 1.0) / a).unsqueeze(1) * self.B.to(a.dtype)  # (d_state, d_model) complex

        # u_complex shape (B, L, d_model)
        u_complex = u.to(a.dtype)

        # Unroll recurrence (T ≤ few hundred in practice).
        h = torch.zeros(B_size, self.d_state, dtype=a.dtype, device=u.device)
        ys = []
        for t in range(L):
            h = h * a_bar + u_complex[:, t, :] @ b_bar.T
            y_t = (h @ self.C.to(a.dtype).T).real + self.D * u[:, t, :]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d_model)

        y = self.dropout(y)
        return self.norm(u + y)


# ─────────────────────────────────────────────────────────────────────────────
# State-space trunk: stack of blocks + readout
# ─────────────────────────────────────────────────────────────────────────────
class StateSpaceTrunk(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 64,
        n_blocks: int = 3,
        dropout: float = 0.1,
        readout_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            DiagonalSSMBlock(d_model=d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.readout = nn.Linear(d_model, readout_dim or d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Accepts either:
          * (B, L, input_dim)    — full sequence input, returns (B, readout_dim)
            via last-step pooling.
          * (B, input_dim)       — static input, treated as single-step sequence.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h)
        last = h[:, -1, :]
        return self.readout(last)


# ─────────────────────────────────────────────────────────────────────────────
# sklearn-style adapter matching LearnableTrunkAdapter's interface
# ─────────────────────────────────────────────────────────────────────────────
class StateSpaceTrunkAdapter:
    """Drop-in sklearn-style adapter producing NumPy trunk features.

    This mirrors :class:`LearnableTrunkAdapter` so mainline wrappers can
    simply swap the class without any other change.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 64,
        n_blocks: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 32,
        batch_size: int = 512,
        patience: int = 5,
        device: str = "cpu",
    ) -> None:
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.n_blocks = int(n_blocks)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.device = torch.device(device)

        self.net: Optional[StateSpaceTrunk] = None
        self._loc = None
        self._scale = None

    def _robust_fit_scale(self, X: np.ndarray) -> None:
        loc = np.median(X, axis=0).astype(np.float32)
        mad = np.median(np.abs(X - loc), axis=0).astype(np.float32)
        scale = mad * np.float32(1.4826)
        scale[scale < 1e-6] = np.float32(1.0)
        self._loc, self._scale = loc, scale

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=np.float32) - self._loc) / self._scale).astype(np.float32)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        aux_task: str = "regression",
    ) -> "StateSpaceTrunkAdapter":
        """Fit the trunk with an auxiliary head over y (regression or binary)."""
        self._robust_fit_scale(X)
        Xs = self._standardise(X)
        self.net = StateSpaceTrunk(
            input_dim=Xs.shape[1],
            d_model=self.d_model,
            d_state=self.d_state,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
        ).to(self.device)

        # Auxiliary head for training the trunk representation.
        head = nn.Linear(self.net.readout.out_features, 1).to(self.device)
        params = list(self.net.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr)

        Xt = torch.from_numpy(Xs).to(self.device)
        yt = None
        if y is not None:
            yt = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)

        n = Xt.shape[0]
        best_loss = float("inf")
        stale = 0
        for epoch in range(self.max_epochs):
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = Xt.index_select(0, idx)
                h = self.net(xb)
                if yt is not None:
                    yhat = head(h).squeeze(-1)
                    yb = yt.index_select(0, idx)
                    if aux_task == "binary":
                        loss = F.binary_cross_entropy_with_logits(yhat, yb)
                    else:
                        loss = F.smooth_l1_loss(yhat, yb)
                else:
                    # Representation-only: variance-preserving loss to avoid
                    # constant collapse (encourages h to have unit variance).
                    loss = (h.var(dim=0).mean() - 1.0).pow(2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * xb.shape[0]
            epoch_loss = total / max(n, 1)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    logger.info("early stop at epoch %d (loss %.6f)", epoch, epoch_loss)
                    break
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.net is not None, "fit() first"
        self.net.eval()
        with torch.no_grad():
            Xs = self._standardise(X)
            Xt = torch.from_numpy(Xs).to(self.device)
            out = self.net(Xt).cpu().numpy().astype(np.float32)
        return out

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        self.fit(X, y=y, **kwargs)
        return self.transform(X)


__all__ = [
    "DiagonalSSMBlock",
    "StateSpaceTrunk",
    "StateSpaceTrunkAdapter",
]
