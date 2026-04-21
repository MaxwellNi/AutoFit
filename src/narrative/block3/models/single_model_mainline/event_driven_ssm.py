#!/usr/bin/env python3
"""Frequency-Decoupled Event-Driven SSM with Neural CDE Modulation (FD-SSM).

Architecture overview:
    1. SeriesDecomp splits input:  X -> X_trend + X_seasonal
    2. X_trend -> Linear projection -> y_trend   (preserves low-freq)
    3. X_seasonal -> Selective SSM -> y_seasonal  (captures periodic / short-range)
    4. Events (optional) -> NCDE Vector-Field Modulation on Delta_t and B_t
       (events change *how fast* and *how sensitively* the SSM absorbs input,
        NOT the state itself -- preserving flow manifold smoothness)
    5. y = y_trend + y_seasonal -> readout -> (B, d_output)

Mathematical formulations:

Frequency-Decoupled SSM:
    X_trend    = AvgPool_1d(X, kernel=K)              (moving average)
    X_seasonal = X - X_trend                          (residual)

Selective SSM (on X_seasonal only):
    Delta_t = softplus(W_d * x_seasonal_t + W_e * event_t + b_d)   [NCDE-modulated]
    B_t = sigma(MLP_B(x_seasonal_t || event_t))                    [NCDE-modulated]
    C_t = W_C * x_seasonal_t
    A_bar   = exp(Delta_t * A)
    B_bar   = Delta_t * B_t
    h_t = A_bar * h_{t-1} + B_bar * x_seasonal_t
    y_t = C_t * h_t + D * x_seasonal_t

Trend channel:
    y_trend = W_trend * X_trend[:, -1, :]             (last-step linear)

Physical interpretation of NCDE modulation:
    - Delta_t encodes the *sampling rate*: events accelerate or decelerate
      the effective time step, changing how quickly the SSM forgets/retains.
    - B_t encodes *input sensitivity*: events change which input dimensions
      the state space absorbs, analogous to a controlled vector field dh/dt = f(h, dX/dt).
    - This is mathematically equivalent to a Neural CDE (Kidger et al. 2020)
      discretized to step-wise updates, preserving Lipschitz continuity
      of the state trajectory.

References:
    - Gu & Dao (2024). "Mamba: Linear-Time Sequence Modeling." ICLR 2024.
    - Zeng et al. (2023). "Are Transformers Effective for Time Series Forecasting?"
      AAAI 2023. (DLinear decomposition)
    - Kidger et al. (2020). "Neural Controlled Differential Equations for
      Irregular Time Series." NeurIPS 2020.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
#  Series Decomposition (DLinear-style)
# ---------------------------------------------------------------------


class SeriesDecomp(nn.Module):
    """Moving-average decomposition: X -> (X_trend, X_seasonal).

    Uses causal (left-padded) average pooling so each timestep t only
    sees information from [t-K+1, t].  No future leakage.
    """

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D)
        Returns:
            trend:    (B, L, D) -- moving average (low frequency)
            seasonal: (B, L, D) -- residual (high frequency)
        """
        pad = self.kernel_size - 1
        x_t = x.permute(0, 2, 1)  # (B, D, L)
        x_padded = F.pad(x_t, (pad, 0), mode="replicate")
        trend = F.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1)
        trend = trend.permute(0, 2, 1)  # (B, L, D)
        seasonal = x - trend
        return trend, seasonal


# ---------------------------------------------------------------------
#  NCDE-Modulated Selective SSM Cell
# ---------------------------------------------------------------------


class NCDESelectiveSSMCell(nn.Module):
    """Selective SSM cell with Neural CDE event modulation.

    When events are present, they modulate the step size Delta_t and the
    input-to-state matrix B_t -- changing the *vector field* rather than
    the state directly.  This preserves Lipschitz continuity of h(t).

    Recurrence:
        Delta[d] = softplus(W_d * x_t + W_e * e_t + b_d)    [event-modulated]
        B[n] = sigma(MLP_B(x_t || e_t))                      [event-modulated]
        C[n] = W_C * x_t
        A_bar[d,n] = exp(Delta[d] * A[d,n])
        B_bar[d,n] = Delta[d] * B[n]
        h[d,n] <- A_bar[d,n] * h[d,n] + B_bar[d,n] * x[d]
        y[d]   = sum_n C[n] * h[d,n] + D[d] * x[d]
    """

    def __init__(self, d_model: int, d_state: int = 16, d_event: int = 0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_event = d_event

        # A: log-parameterised diagonal (negative eigenvalues -> stability).
        # S4D-inv-style init: A_base in [-0.55, -0.05]. The old init
        #     A_log = log([1..d_state])
        # produced A_base=[-1..-16]; with softplus-biased delta~1.5 this gave
        # delta*A ~ -24, and exp(-24) ~ 3.8e-11 -- fp32 underflow after a few
        # steps, causing h_t to collapse to literal zero (var<1e-12 assert).
        # See audit note in docs/references/agents_handover.md §3.
        idx = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_init = torch.log(idx / d_state * 0.5 + 0.05)
        self.A_log = nn.Parameter(
            A_init.unsqueeze(0).expand(d_model, -1).clone()
        )  # (d_model, d_state)

        # D: feedthrough / skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        # Unified algebraic modulation terms (all event perturbations are no-bias).
        self.delta_proj = nn.Linear(d_model, d_model, bias=True)
        self.delta_event_proj = nn.Linear(d_event, d_model, bias=False)

        self.B_proj = nn.Sequential(
            nn.Linear(d_model, d_state * 2),
            nn.SiLU(),
            nn.Linear(d_state * 2, d_state),
        )
        self.B_event_proj = nn.Linear(d_event, d_state, bias=False)

        # Eigen-modulation: event perturbation shifts poles directly.
        self.A_event_proj = nn.Linear(d_event, d_model, bias=False)

        # C: state-to-output (always from x only)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

        # Bias init: small step sizes so delta*A stays in a numerically safe band.
        # Softplus(-2)~0.13, softplus(-1)~0.31 -> delta*|A| max ~ 0.17 initially,
        # h_t decays gracefully instead of underflowing to zero.
        with torch.no_grad():
            self.delta_proj.bias.uniform_(-2.0, -1.0)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        event_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t:     (B, d_model)          -- projected seasonal input.
            h_prev:  (B, d_model, d_state) -- previous hidden state.
            event_t: (B, d_event) -- event features at time t. Can be zero-vector.
        Returns:
            y_t: (B, d_model)              -- cell output.
            h_t: (B, d_model, d_state)     -- new hidden state.
        """
        # 1) Delta modulation: base + no-bias event perturbation.
        delta = F.softplus(self.delta_proj(x_t) + self.delta_event_proj(event_t))

        # 2) B modulation: base + no-bias event perturbation.
        B_t = self.B_proj(x_t) + self.B_event_proj(event_t)

        # 3. State-to-output C (always from x only -- no event leak)
        C_t = self.C_proj(x_t)  # (B, N)

        # 4) Pole modulation in eigen-space: A_effective = A_base + event_perturbation.
        # Hard clamp on A_log keeps A_base in [-exp(1)=-2.72, -exp(-3)=-0.05].
        # Without this, large continuous targets (funding_usd, investors_count)
        # push A_log to extreme negative values during training, returning the
        # exp(delta*A) recurrence to fp32 underflow. See handover §3.
        A_log_safe = self.A_log.clamp(min=-3.0, max=1.0)
        A_base = -torch.exp(A_log_safe).unsqueeze(0)  # (1, D, N)
        event_pole = self.A_event_proj(event_t).unsqueeze(-1)  # (B, D, 1)
        A_eff = A_base + event_pole

        # 5. ZOH discretisation -- delta is also clamped to keep delta*|A| bounded.
        delta_safe = delta.clamp(max=2.0)
        dA = delta_safe.unsqueeze(-1) * A_eff            # (B, D, N)
        A_bar = torch.exp(dA)                            # (B, D, N)
        B_bar = delta_safe.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, D, N)

        # 6. State recurrence
        h_t = A_bar * h_prev + B_bar * x_t.unsqueeze(-1)  # (B, D, N)

        # 7. Output
        y_t = (h_t * C_t.unsqueeze(1)).sum(-1) + self.D * x_t  # (B, D)

        return y_t, h_t


# ---------------------------------------------------------------------
#  Frequency-Decoupled Event-Driven SSM
# ---------------------------------------------------------------------


class UnifiedJumpDiffusionSSM(nn.Module):
    """Unified Jump-Diffusion SSM with algebraic event perturbation.

    A single manifold is used for both continuous and event-driven regimes.
    Event effects enter only through no-bias perturbation operators, so when
    event input is exactly zero, the dynamics reduce exactly to pure continuous SSM.
    """

    def __init__(
        self,
        d_cont: int,
        d_event: int = 0,
        d_model: int = 64,
        d_state: int = 16,
        d_output: int = 64,
        n_layers: int = 2,
        decomp_kernel: int = 25,
        # Legacy parameter -- accepted but ignored (no more JumpGate)
        d_jump: int = 64,
    ):
        super().__init__()
        self.d_cont = d_cont
        self.d_event = d_event
        self.d_model = d_model
        self.d_state = d_state
        self.d_output = d_output

        # Unified input projection on continuous channels.
        self.input_proj = nn.Sequential(
            nn.Linear(d_cont, d_model),
            nn.LayerNorm(d_model),
        )

        # -- Stacked NCDE-modulated SSM layers --
        self.ssm_cells = nn.ModuleList([
            NCDESelectiveSSMCell(d_model, d_state, d_event=d_event)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        # -- Readout --
        self.readout = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, d_output),
            nn.GELU(),
            nn.LayerNorm(d_output),
        )

        # Track flat state dim for diagnostics
        self.d_state_flat = d_model * d_state

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.input_proj, self.readout]:
            for sub in m.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.kaiming_normal_(sub.weight, nonlinearity="relu")
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_states: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, L, d_cont) or (B, L, d_cont + d_event).
            lengths: (B,) optional -- valid timesteps per sequence.
            return_states: if True, return dict with diagnostics.

        Returns:
            (B, d_output) or dict with 'output', 'trajectory', 'jump_gates'.
        """
        B, L, D_in = x.shape
        device = x.device

        x_cont = x[..., :self.d_cont]
        x_event_raw = x[..., self.d_cont:]
        if self.d_event > 0:
            event_short = max(0, self.d_event - x_event_raw.shape[-1])
            x_event = F.pad(x_event_raw, (0, event_short))[:, :, :self.d_event]
        else:
            x_event = x_cont.new_zeros(B, L, 0)

        # Unified projection + recurrent dynamics.
        # CRITICAL FIX: Heavy-tailed input normalization to prevent matrix exp blowup
        x_cont = torch.sign(x_cont) * torch.log1p(torch.abs(x_cont))
        x_proj = self.input_proj(x_cont)

        h_layers = [
            torch.zeros(B, self.d_model, self.d_state, device=device)
            for _ in self.ssm_cells
        ]
        y_last = torch.zeros(B, self.d_model, device=device)

        if return_states:
            traj_buf: list[torch.Tensor] = []

        for t in range(L):
            xt = x_proj[:, t]
            et = x_event[:, t]

            for i, (cell, norm) in enumerate(
                zip(self.ssm_cells, self.layer_norms)
            ):
                yt, h_layers[i] = cell(xt, h_layers[i], event_t=et)
                xt = norm(yt + xt)

            if lengths is not None:
                valid = (t < lengths).float().view(B, 1)
                y_last = valid * xt + (1.0 - valid) * y_last
            else:
                y_last = xt

            if return_states:
                traj_buf.append(h_layers[-1].reshape(B, -1))

        output = self.readout(y_last)

        if return_states:
            return {
                "output": output,
                "trajectory": torch.stack(traj_buf, dim=1),
                "jump_gates": torch.zeros(B, L, device=device),
            }
        return output


class EventDrivenSSM(UnifiedJumpDiffusionSSM):
    """Backward-compatible alias for unified jump-diffusion implementation."""


# ---------------------------------------------------------------------
#  Composite: FD-SSM -> SparseMoETrunk pipeline
# ---------------------------------------------------------------------

_SparseMoETrunk = None


def _get_sparse_moe_trunk():
    global _SparseMoETrunk
    if _SparseMoETrunk is None:
        from .learnable_trunk import SparseMoETrunk
        _SparseMoETrunk = SparseMoETrunk
    return _SparseMoETrunk


class SequentialMoETrunk(nn.Module):
    """Full pipeline: EventDrivenSSM -> SparseMoETrunk.

    Processes sequential input through the frequency-decoupled SSM,
    then routes through sparse expert mixture for multi-task decoding.
    """

    def __init__(
        self,
        d_cont: int,
        d_event: int = 0,
        d_model: int = 64,
        d_state: int = 16,
        n_ssm_layers: int = 2,
        d_jump: int = 64,       # legacy, accepted but ignored
        decomp_kernel: int = 25,
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
        SparseMoETrunk = _get_sparse_moe_trunk()

        d_output = d_model

        self.ssm = EventDrivenSSM(
            d_cont=d_cont,
            d_event=d_event,
            d_model=d_model,
            d_state=d_state,
            d_output=d_output,
            n_layers=n_ssm_layers,
            decomp_kernel=decomp_kernel,
        )
        self.moe = SparseMoETrunk(
            input_dim=d_output,
            compact_dim=compact_dim,
            n_experts=n_experts,
            expert_dim=expert_dim,
            top_k=top_k,
            n_tasks=n_tasks,
            projection_hidden=projection_hidden,
            expert_hidden=expert_hidden,
            load_balance_weight=load_balance_weight,
            diversity_weight=diversity_weight,
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        task_id: torch.Tensor,
        lengths: torch.Tensor | None = None,
        return_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        ssm_out = self.ssm(x_seq, lengths, return_states=return_states)

        if return_states:
            z_seq = ssm_out["output"]
            moe_out = self.moe(z_seq, task_id)
            moe_out["trajectory"] = ssm_out["trajectory"]
            moe_out["jump_gates"] = ssm_out["jump_gates"]
            return moe_out
        else:
            return self.moe(ssm_out, task_id)


# ---------------------------------------------------------------------
#  Standalone forecaster for ETT-style benchmarks
# ---------------------------------------------------------------------


class FDSSMForecaster(nn.Module):
    """Unified jump-diffusion forecaster for univariate/multivariate TSF."""

    def __init__(
        self,
        d_input: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 2,
        decomp_kernel: int = 25,
        individual: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_input = d_input
        self.individual = individual

        # Single-manifold SSM, no external DLinear branch.
        self.ssm = UnifiedJumpDiffusionSSM(
            d_cont=d_input,
            d_event=0,
            d_model=d_model,
            d_state=d_state,
            d_output=d_model,
            n_layers=n_layers,
            decomp_kernel=decomp_kernel,
        )

        self.forecast_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_input) -- input sequence.
        Returns:
            (B, pred_len) -- forecast for target channel (last channel).
        """
        ssm_emb = self.ssm(x)
        return self.forecast_head(ssm_emb)
