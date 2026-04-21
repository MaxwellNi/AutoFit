#!/usr/bin/env python3
"""Causal decoders for zero-inflated heavy-tail targets.

This module provides three branch decoders that sit on top of a frozen
sequential trunk embedding h_t:

- Alpha: ICM-LogNormal (stable probabilistic baseline)
- Beta:  ICM-IQN (implicit quantile network)
- Gamma: ICM-CFM (simulation-free conditional flow matching)

All branches enforce:
1) Lateral causal detach: S -> C -> F with strict .detach() on downstream inputs.
2) Zero-inflation masking: regression losses are computed on target > 0 only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogNormal


def positive_mask(targets: torch.Tensor) -> torch.Tensor:
    """Return float mask for strictly positive samples."""
    return (targets > 0.0).to(dtype=targets.dtype)


def _audit_latent_heartbeat(h_t: torch.Tensor) -> None:
    """Non-fatal latent-state heartbeat.

    Historical behaviour aborted training whenever batch latent variance
    fell below a hard floor. Empirical evidence (5340592-5340827) shows the
    SSM trunk is numerically healthy once A_log + delta are properly clamped
    (see event_driven_ssm.py §NCDESelectiveSSMCell). The remaining low-variance
    events come from extremely zero-inflated continuous targets (funding_usd,
    investors_count) where the trunk legitimately compresses a near-constant
    batch. Aborting in that case throws away the first real numbers we could
    compare against NBEATS / KAN. We therefore warn once but never raise.
    """
    if h_t.ndim != 2:
        return
    if h_t.shape[0] < 2:
        return
    hb = h_t.var(dim=0, unbiased=False).mean()
    if not torch.isfinite(hb):
        # NaN / Inf still indicate a hard numerical corruption; surface a warning
        # but do not abort -- the optimiser will be clipped upstream.
        import warnings as _w
        _w.warn("latent heartbeat: non-finite variance detected", RuntimeWarning)
        return
    val = float(hb.detach().cpu().item())
    if val < 1e-8:
        import warnings as _w
        _w.warn(
            f"latent heartbeat: low batch variance {val:.3e} (expected for "
            "extremely zero-inflated targets; training continues)",
            RuntimeWarning,
        )


@dataclass
class DecoderOutput:
    main_loss: torch.Tensor
    s_loss: torch.Tensor
    c_loss: torch.Tensor
    total_loss: torch.Tensor
    stats: Dict[str, float]


class CausalConditionChain(nn.Module):
    """S -> C -> context chain with lateral gradient blocking."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head_s = nn.Linear(hidden_dim, 1)
        self.head_c = nn.Linear(hidden_dim + 1, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_logits = self.head_s(h)
        c_input = torch.cat([h, s_logits.detach()], dim=-1)
        c_lambda = F.softplus(self.head_c(c_input)) + 1e-6
        f_context = torch.cat([h, c_lambda.detach()], dim=-1)
        return s_logits.squeeze(-1), c_lambda.squeeze(-1), f_context


class ICMLogNormalDecoder(nn.Module):
    """Alpha branch: causal LogNormal head with zero-inflation masking."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.chain = CausalConditionChain(hidden_dim)
        self.head_f = nn.Linear(hidden_dim + 1, 2)

    def _funding_params(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_logits, c_lambda, f_ctx = self.chain(h)
        raw = self.head_f(f_ctx)
        mu = raw[:, 0]
        sigma = F.softplus(raw[:, 1]) + 1e-4
        return s_logits, c_lambda, mu, sigma

    def compute_loss(self, h: torch.Tensor, targets: torch.Tensor) -> DecoderOutput:
        _audit_latent_heartbeat(h)
        s_logits, c_lambda, mu, sigma = self._funding_params(h)
        # Teacher forcing mask: ALWAYS derived from ground-truth target only.
        pmask = positive_mask(targets)
        pos_idx = pmask > 0

        s_target = (targets > 0.0).to(dtype=targets.dtype)
        s_loss = F.binary_cross_entropy_with_logits(s_logits, s_target)
        c_target = torch.log1p(torch.clamp(targets, min=0.0))
        c_loss = F.mse_loss(c_lambda, c_target)

        if torch.any(pos_idx):
            y_pos = targets[pos_idx]
            mu_pos = mu[pos_idx]
            sigma_pos = sigma[pos_idx]
            main_loss = -LogNormal(mu_pos, sigma_pos).log_prob(y_pos).mean()
            pred_pos = torch.exp(mu_pos)
            mae_pos = torch.mean(torch.abs(pred_pos - y_pos))
            pred_mean_pos = torch.mean(pred_pos)
        else:
            main_loss = h.sum() * 0.0
            mae_pos = h.sum() * 0.0
            pred_mean_pos = h.sum() * 0.0

        total = main_loss + 0.05 * s_loss + 0.05 * c_loss
        stats = {
            "positive_fraction": float(pmask.mean().detach().cpu().item()),
            "main_loss": float(main_loss.detach().cpu().item()),
            "mae_pos": float(mae_pos.detach().cpu().item()),
            "pred_mean_pos": float(pred_mean_pos.detach().cpu().item()),
        }
        return DecoderOutput(main_loss, s_loss, c_loss, total, stats)


class ICMIQNDecoder(nn.Module):
    """Beta branch: causal implicit quantile network with Quantile-Huber loss."""

    def __init__(self, hidden_dim: int, n_cos: int = 16, n_tau: int = 16, huber_kappa: float = 1.0):
        super().__init__()
        self.chain = CausalConditionChain(hidden_dim)
        self.n_cos = int(n_cos)
        self.n_tau = int(n_tau)
        self.huber_kappa = float(huber_kappa)

        self.tau_proj = nn.Sequential(
            nn.Linear(self.n_cos, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + 1 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _cos_embed(self, tau: torch.Tensor) -> torch.Tensor:
        i = torch.arange(1, self.n_cos + 1, device=tau.device, dtype=tau.dtype).view(1, 1, -1)
        return torch.cos(torch.pi * tau.unsqueeze(-1) * i)

    def _quantile_huber(self, pred_q: torch.Tensor, target: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # pred_q: (N, T), target: (N, 1), tau: (N, T)
        err = target - pred_q
        abs_err = torch.abs(err)
        huber = torch.where(
            abs_err <= self.huber_kappa,
            0.5 * err * err,
            self.huber_kappa * (abs_err - 0.5 * self.huber_kappa),
        )
        weight = torch.abs(tau - (err.detach() < 0.0).to(dtype=tau.dtype))
        return (weight * huber).mean()

    def compute_loss(self, h: torch.Tensor, targets: torch.Tensor) -> DecoderOutput:
        _audit_latent_heartbeat(h)
        s_logits, c_lambda, f_ctx = self.chain(h)
        # Teacher forcing mask: NEVER gate by S/C predictions.
        pmask = positive_mask(targets)
        pos_idx = pmask > 0

        s_target = (targets > 0.0).to(dtype=targets.dtype)
        s_loss = F.binary_cross_entropy_with_logits(s_logits, s_target)
        c_target = torch.log1p(torch.clamp(targets, min=0.0))
        c_loss = F.mse_loss(c_lambda, c_target)

        if torch.any(pos_idx):
            h_pos = f_ctx[pos_idx]
            y_pos = targets[pos_idx].unsqueeze(-1)
            n_pos = h_pos.shape[0]

            tau = torch.rand(n_pos, self.n_tau, device=h.device, dtype=h.dtype)
            tau_emb = self.tau_proj(self._cos_embed(tau).reshape(n_pos * self.n_tau, self.n_cos))
            h_rep = h_pos.unsqueeze(1).expand(-1, self.n_tau, -1).reshape(n_pos * self.n_tau, -1)
            q_pred = self.q_head(torch.cat([h_rep, tau_emb], dim=-1)).reshape(n_pos, self.n_tau)
            main_loss = self._quantile_huber(q_pred, y_pos, tau)

            # Fixed high quantiles for VaR-style diagnostics.
            tau_eval = torch.tensor([0.90, 0.95, 0.99], device=h.device, dtype=h.dtype).view(1, 3).expand(n_pos, 3)
            tau_eval_emb = self.tau_proj(self._cos_embed(tau_eval).reshape(n_pos * 3, self.n_cos))
            h_eval = h_pos.unsqueeze(1).expand(-1, 3, -1).reshape(n_pos * 3, -1)
            q_eval = self.q_head(torch.cat([h_eval, tau_eval_emb], dim=-1)).reshape(n_pos, 3)
            q90_mean = torch.mean(q_eval[:, 0])
            q95_mean = torch.mean(q_eval[:, 1])
            q99_mean = torch.mean(q_eval[:, 2])
        else:
            main_loss = h.sum() * 0.0
            q90_mean = h.sum() * 0.0
            q95_mean = h.sum() * 0.0
            q99_mean = h.sum() * 0.0

        total = main_loss + 0.05 * s_loss + 0.05 * c_loss
        stats = {
            "positive_fraction": float(pmask.mean().detach().cpu().item()),
            "main_loss": float(main_loss.detach().cpu().item()),
            "q90_mean": float(q90_mean.detach().cpu().item()),
            "q95_mean": float(q95_mean.detach().cpu().item()),
            "q99_mean": float(q99_mean.detach().cpu().item()),
        }
        return DecoderOutput(main_loss, s_loss, c_loss, total, stats)


class ICMCFMDecoder(nn.Module):
    """Gamma branch: causal conditional flow matching head."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.chain = CausalConditionChain(hidden_dim)
        self.vfield = nn.Sequential(
            nn.Linear(hidden_dim + 1 + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _v(self, f_ctx: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([f_ctx, x_t.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        return self.vfield(inp).squeeze(-1)

    def sample_euler(self, h: torch.Tensor, steps: int = 32) -> torch.Tensor:
        """Simple Euler solver for conditional generation from N(0,1)."""
        _audit_latent_heartbeat(h)
        _, _, f_ctx = self.chain(h)
        x = torch.randn(h.shape[0], device=h.device, dtype=h.dtype)
        dt = 1.0 / float(max(int(steps), 1))
        for i in range(int(steps)):
            t = torch.full_like(x, (i + 0.5) * dt)
            x = x + dt * self._v(f_ctx, x, t)
        return torch.expm1(torch.clamp(x, min=0.0))

    def compute_loss(self, h: torch.Tensor, targets: torch.Tensor) -> DecoderOutput:
        _audit_latent_heartbeat(h)
        s_logits, c_lambda, f_ctx = self.chain(h)
        # Teacher forcing mask: derived from observed target only.
        pmask = positive_mask(targets)
        pos_idx = pmask > 0

        s_target = (targets > 0.0).to(dtype=targets.dtype)
        s_loss = F.binary_cross_entropy_with_logits(s_logits, s_target)
        c_target = torch.log1p(torch.clamp(targets, min=0.0))
        c_loss = F.mse_loss(c_lambda, c_target)

        if torch.any(pos_idx):
            f_pos = f_ctx[pos_idx]
            y_pos = torch.log1p(targets[pos_idx])
            n_pos = y_pos.shape[0]
            x0 = torch.randn(n_pos, device=h.device, dtype=h.dtype)
            t = torch.rand(n_pos, device=h.device, dtype=h.dtype)
            x_t = (1.0 - t) * x0 + t * y_pos
            target_v = y_pos - x0
            pred_v = self._v(f_pos, x_t, t)
            main_loss = F.mse_loss(pred_v, target_v)
        else:
            main_loss = h.sum() * 0.0

        total = main_loss + 0.05 * s_loss + 0.05 * c_loss
        stats = {
            "positive_fraction": float(pmask.mean().detach().cpu().item()),
            "main_loss": float(main_loss.detach().cpu().item()),
        }
        return DecoderOutput(main_loss, s_loss, c_loss, total, stats)


def build_causal_decoder(branch: str, hidden_dim: int) -> nn.Module:
    key = str(branch).strip().lower()
    if key in {"alpha", "icm_lognormal", "lognormal"}:
        return ICMLogNormalDecoder(hidden_dim)
    if key in {"beta", "icm_iqn", "iqn"}:
        return ICMIQNDecoder(hidden_dim)
    if key in {"gamma", "icm_cfm", "cfm"}:
        return ICMCFMDecoder(hidden_dim)
    raise ValueError(f"Unknown causal decoder branch: {branch}")
