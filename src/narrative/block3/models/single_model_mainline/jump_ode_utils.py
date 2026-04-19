#!/usr/bin/env python3
"""Minimal Jump ODE state evolution for the shared trunk.

Implements a lightweight, numpy-compatible approximation of the Jump Neural ODE
(Jia & Benson, ICML 2020) for modeling financing process state evolution.

Mathematical formulation:

  Between events:  x_{t+1} = x_t + f(x_t) · Δt        (Euler ODE step)
  At jump times:   x_{t^+} = x_t + g(x_t, mark)        (jump correction)

where:
  - f(x_t) is a learned drift function (linear + nonlinear residual)
  - g(x_t, mark) is a mark-dependent jump function
  - Jump times are detected from event state atoms (phase_transition_flag,
    funding_boundary_up, investors_boundary_up, etc.)

This avoids torchdiffeq dependency by using:
  - Euler discretization for the continuous drift
  - Ridge/HGBR for learning drift and jump functions
  - Entity-isolated integration (no cross-entity contamination)

Reference:
  - Jia & Benson, "Neural Jump Stochastic Differential Equations" (ICML 2020)
  - Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
  - Merton, "Option Pricing when Underlying Stock Returns are Discontinuous"
    (J. Finance 1976) — jump-diffusion economic foundation
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


# ---------------------------------------------------------------------------
# 1. Jump ODE state builder
# ---------------------------------------------------------------------------

def build_jump_ode_state(
    compact_state: np.ndarray,
    event_atoms: np.ndarray | None,
    entity_ids: np.ndarray | None,
    *,
    drift_model: _DriftModel | None = None,
    jump_model: _JumpModel | None = None,
    n_ode_dims: int = 8,
) -> np.ndarray:
    """Build Jump ODE-enhanced state features from compact backbone state.

    Parameters
    ----------
    compact_state : (n, d) float32
        Compact state from SharedTemporalBackbone (after z-score + projection).
    event_atoms : (n, n_atoms) float32 or None
        Event state columns: [phase_transition_flag, funding_boundary_up,
        investors_boundary_up, funding_jump_log, investor_jump_log, ...].
        If None, returns zeros.
    entity_ids : (n,) or None
        Entity identifiers for per-entity isolation.
    drift_model : _DriftModel or None
        Fitted drift model. If None, uses zero drift.
    jump_model : _JumpModel or None
        Fitted jump model. If None, uses raw jump signal.
    n_ode_dims : int
        Number of ODE state dimensions to produce.

    Returns
    -------
    ode_state : (n, n_ode_dims + 4) float32
        Jump ODE state: [ode_dims..., drift_norm, jump_count,
                         cumulative_jump_energy, state_smoothness]
    """
    n = compact_state.shape[0]
    n_out = n_ode_dims + 4
    if event_atoms is None or entity_ids is None or n == 0:
        return np.zeros((n, n_out), dtype=np.float32)

    state = np.asarray(compact_state, dtype=np.float64)
    events = np.asarray(event_atoms, dtype=np.float64)
    ent = np.asarray(entity_ids)

    # Project state down to ODE dims via truncated SVD-style selection
    d = state.shape[1]
    if d <= n_ode_dims:
        ode_input = np.pad(state, ((0, 0), (0, n_ode_dims - d)), constant_values=0)
    else:
        # Use first n_ode_dims PCs (variance-maximizing)
        ode_input = state[:, :n_ode_dims]

    # Detect jump indicators from event atoms
    jump_mask = _detect_jumps(events)

    # Per-entity Euler integration with jumps
    ode_output = np.zeros_like(ode_input)
    drift_norms = np.zeros(n, dtype=np.float64)
    jump_counts = np.zeros(n, dtype=np.float64)
    cum_jump_energy = np.zeros(n, dtype=np.float64)
    state_smoothness = np.zeros(n, dtype=np.float64)

    unique_ents = np.unique(ent)
    for e in unique_ents:
        mask = ent == e
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue

        ent_state = ode_input[idx]
        ent_jumps = jump_mask[idx]
        ent_events = events[idx]

        ode_traj, drifts, jcounts, jenergy, smooth = _euler_integrate_entity(
            ent_state, ent_jumps, ent_events,
            drift_model=drift_model,
            jump_model=jump_model,
        )

        ode_output[idx] = ode_traj
        drift_norms[idx] = drifts
        jump_counts[idx] = jcounts
        cum_jump_energy[idx] = jenergy
        state_smoothness[idx] = smooth

    # Assemble output
    result = np.column_stack([
        ode_output,
        drift_norms.reshape(-1, 1),
        jump_counts.reshape(-1, 1),
        cum_jump_energy.reshape(-1, 1),
        state_smoothness.reshape(-1, 1),
    ])

    # Safety: clip and replace non-finite
    result = np.nan_to_num(result, nan=0.0, posinf=5.0, neginf=-5.0)
    result = np.clip(result, -10.0, 10.0)

    return result.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 2. Drift model: f(x) = Ax + b (linear) + nonlinear residual
# ---------------------------------------------------------------------------

class _DriftModel:
    """Learned drift function f(x_t) for continuous state evolution."""

    def __init__(self, n_dims: int, alpha: float = 1.0, random_state: int = 0):
        self.n_dims = n_dims
        self.alpha = alpha
        self.random_state = random_state
        self._linear: Ridge | None = None
        self._fitted = False

    def fit(self, x_t: np.ndarray, x_next: np.ndarray, dt: np.ndarray) -> "_DriftModel":
        """Fit drift from consecutive non-jump state pairs.

        f(x_t) ≈ (x_{t+1} - x_t) / Δt for non-jump transitions.
        """
        if x_t.shape[0] < 10:
            self._fitted = False
            return self

        # Target: velocity = (x_next - x_t) / dt
        dt_safe = np.maximum(dt.reshape(-1, 1), 1e-6)
        velocity = (x_next - x_t) / dt_safe

        self._linear = Ridge(alpha=self.alpha)
        self._linear.fit(x_t, velocity)
        self._fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict drift f(x) for given states."""
        if not self._fitted or self._linear is None:
            return np.zeros_like(x)
        return self._linear.predict(x)


# ---------------------------------------------------------------------------
# 3. Jump model: g(x, mark) = jump correction
# ---------------------------------------------------------------------------

class _JumpModel:
    """Learned jump function g(x_t, mark) for discrete state corrections."""

    def __init__(self, n_dims: int, random_state: int = 0):
        self.n_dims = n_dims
        self.random_state = random_state
        self._models: list[HistGradientBoostingRegressor | None] = []
        self._fitted = False

    def fit(
        self,
        x_pre_jump: np.ndarray,
        jump_marks: np.ndarray,
        x_post_jump: np.ndarray,
    ) -> "_JumpModel":
        """Fit jump correction from pre/post-jump state pairs.

        g(x, mark) ≈ x_{t^+} - x_{t^-} conditioned on (x_{t^-}, mark).
        """
        if x_pre_jump.shape[0] < 5:
            self._fitted = False
            return self

        delta = x_post_jump - x_pre_jump
        features = np.column_stack([x_pre_jump, jump_marks])

        self._models = []
        for dim in range(min(self.n_dims, delta.shape[1])):
            model = HistGradientBoostingRegressor(
                max_depth=2,
                max_iter=50,
                learning_rate=0.05,
                min_samples_leaf=3,
                random_state=self.random_state + dim,
            )
            model.fit(features, delta[:, dim])
            self._models.append(model)

        self._fitted = True
        return self

    def predict(self, x: np.ndarray, marks: np.ndarray) -> np.ndarray:
        """Predict jump correction g(x, mark)."""
        if not self._fitted or not self._models:
            return np.zeros((x.shape[0], self.n_dims), dtype=np.float64)

        features = np.column_stack([x, marks])
        corrections = np.zeros((x.shape[0], self.n_dims), dtype=np.float64)
        for dim, model in enumerate(self._models):
            if model is not None:
                corrections[:, dim] = model.predict(features)
        return corrections


# ---------------------------------------------------------------------------
# 4. Per-entity Euler integration
# ---------------------------------------------------------------------------

def _euler_integrate_entity(
    state: np.ndarray,
    jump_mask: np.ndarray,
    event_atoms: np.ndarray,
    *,
    drift_model: _DriftModel | None = None,
    jump_model: _JumpModel | None = None,
    dt: float = 1.0,
    drift_scale: float = 0.3,
    jump_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Euler-integrate one entity's state trajectory with jumps.

    Returns
    -------
    ode_traj : (T, d)
    drift_norms : (T,)
    jump_counts : (T,) cumulative
    cum_jump_energy : (T,) cumulative
    smoothness : (T,) local state continuity measure
    """
    T, d = state.shape
    ode_traj = np.zeros_like(state)
    drift_norms = np.zeros(T, dtype=np.float64)
    jump_counts = np.zeros(T, dtype=np.float64)
    cum_energy = np.zeros(T, dtype=np.float64)
    smoothness = np.zeros(T, dtype=np.float64)

    # Initialize first step
    ode_traj[0] = state[0]
    running_jump_count = 0.0
    running_energy = 0.0

    for t in range(1, T):
        x_prev = ode_traj[t - 1]

        # Continuous drift: x_t = x_{t-1} + f(x_{t-1}) * dt * scale
        if drift_model is not None and drift_model._fitted:
            drift = drift_model.predict(x_prev.reshape(1, -1))[0]
        else:
            # Default drift: gentle mean reversion toward data
            drift = (state[t] - x_prev) * 0.5

        drift = np.clip(drift, -5.0, 5.0)
        drift_norm = float(np.linalg.norm(drift))
        x_new = x_prev + drift * dt * drift_scale

        # Discrete jump at event boundaries
        if jump_mask[t]:
            running_jump_count += 1.0
            if jump_model is not None and jump_model._fitted:
                marks = _extract_jump_marks(event_atoms[t])
                correction = jump_model.predict(
                    x_prev.reshape(1, -1), marks.reshape(1, -1)
                )[0]
            else:
                # Default: use the raw data jump as correction
                correction = (state[t] - x_new) * jump_scale

            correction = np.clip(correction, -5.0, 5.0)
            x_new = x_new + correction
            running_energy += float(np.linalg.norm(correction))

        # Clip to prevent explosion
        x_new = np.clip(x_new, -10.0, 10.0)

        ode_traj[t] = x_new
        drift_norms[t] = drift_norm
        jump_counts[t] = running_jump_count
        cum_energy[t] = running_energy

        # Smoothness: inverse of state jerk (2nd derivative)
        if t >= 2:
            accel = ode_traj[t] - 2 * ode_traj[t - 1] + ode_traj[t - 2]
            smoothness[t] = 1.0 / (1.0 + float(np.linalg.norm(accel)))
        else:
            smoothness[t] = 1.0

    return ode_traj, drift_norms, jump_counts, cum_energy, smoothness


# ---------------------------------------------------------------------------
# 5. Jump detection from event atoms
# ---------------------------------------------------------------------------

def _detect_jumps(event_atoms: np.ndarray) -> np.ndarray:
    """Detect jump times from event state atoms.

    A jump occurs when any of:
    - phase_transition_flag > 0.5
    - funding_boundary_up > 0.5
    - investors_boundary_up > 0.5
    - funded_flip_up > 0.5

    Assumes event_atoms columns follow the backbone event_state schema.
    """
    n = event_atoms.shape[0]
    n_cols = event_atoms.shape[1]

    # Sum all binary event indicators (any positive event = jump)
    # Use absolute values to detect any non-zero event signal
    if n_cols == 0:
        return np.zeros(n, dtype=bool)

    # Threshold: any column with value > 0.5 triggers a jump
    jump_signals = np.abs(event_atoms) > 0.5
    return np.any(jump_signals, axis=1)


def _extract_jump_marks(event_row: np.ndarray) -> np.ndarray:
    """Extract jump mark features from a single event row."""
    # Return the raw event atom values as mark features
    return np.asarray(event_row, dtype=np.float64)


# ---------------------------------------------------------------------------
# 6. Full fit/predict pipeline
# ---------------------------------------------------------------------------

def fit_jump_ode(
    compact_states: np.ndarray,
    event_atoms: np.ndarray,
    entity_ids: np.ndarray,
    *,
    n_ode_dims: int = 8,
    random_state: int = 0,
    min_entities: int = 3,
    min_pairs: int = 20,
) -> dict:
    """Fit drift and jump models from training entity trajectories.

    Parameters
    ----------
    compact_states : (n, d)
        Backbone compact state for training rows.
    event_atoms : (n, k)
        Event state atoms for training rows.
    entity_ids : (n,)
        Entity identifiers.
    n_ode_dims : int
        Number of ODE state dimensions.
    random_state : int
    min_entities : int
        Minimum entities required to fit models.
    min_pairs : int
        Minimum consecutive pairs required.

    Returns
    -------
    result : dict
        Keys: drift_model, jump_model, n_ode_dims, fitted, diagnostics.
    """
    state = np.asarray(compact_states, dtype=np.float64)
    events = np.asarray(event_atoms, dtype=np.float64)
    ent = np.asarray(entity_ids)

    d = state.shape[1]
    ode_dim = min(n_ode_dims, d)
    ode_state = state[:, :ode_dim] if d > ode_dim else np.pad(
        state, ((0, 0), (0, ode_dim - d)), constant_values=0
    )

    unique_ents = np.unique(ent)
    if len(unique_ents) < min_entities:
        return {
            "drift_model": None,
            "jump_model": None,
            "n_ode_dims": ode_dim,
            "fitted": False,
            "diagnostics": {"reason": "too_few_entities"},
        }

    jump_mask = _detect_jumps(events)

    # Collect consecutive pairs
    x_t_list, x_next_list, dt_list = [], [], []
    x_pre_list, x_post_list, mark_list = [], [], []

    for e in unique_ents:
        idx = np.where(ent == e)[0]
        if len(idx) < 2:
            continue

        ent_ode = ode_state[idx]
        ent_jumps = jump_mask[idx]
        ent_events = events[idx]

        for i in range(len(idx) - 1):
            if not ent_jumps[i + 1]:
                # Non-jump pair: use for drift fitting
                x_t_list.append(ent_ode[i])
                x_next_list.append(ent_ode[i + 1])
                dt_list.append(1.0)
            else:
                # Jump pair: use for jump fitting
                x_pre_list.append(ent_ode[i])
                x_post_list.append(ent_ode[i + 1])
                mark_list.append(ent_events[i + 1])

    # Fit drift model
    drift_model = _DriftModel(n_dims=ode_dim, random_state=random_state)
    if len(x_t_list) >= min_pairs:
        x_t_arr = np.array(x_t_list)
        x_next_arr = np.array(x_next_list)
        dt_arr = np.array(dt_list)
        drift_model.fit(x_t_arr, x_next_arr, dt_arr)

    # Fit jump model
    jump_model = _JumpModel(n_dims=ode_dim, random_state=random_state)
    if len(x_pre_list) >= 5:
        x_pre_arr = np.array(x_pre_list)
        x_post_arr = np.array(x_post_list)
        mark_arr = np.array(mark_list)
        jump_model.fit(x_pre_arr, mark_arr, x_post_arr)

    return {
        "drift_model": drift_model if drift_model._fitted else None,
        "jump_model": jump_model if jump_model._fitted else None,
        "n_ode_dims": ode_dim,
        "fitted": drift_model._fitted or jump_model._fitted,
        "diagnostics": {
            "n_drift_pairs": len(x_t_list),
            "n_jump_pairs": len(x_pre_list),
            "drift_fitted": drift_model._fitted,
            "jump_fitted": jump_model._fitted,
            "n_entities": len(unique_ents),
        },
    }


def jump_ode_diagnostics(
    compact_states: np.ndarray,
    ode_states: np.ndarray,
) -> dict[str, float]:
    """Compute diagnostics comparing original and ODE-enhanced states.

    Parameters
    ----------
    compact_states : (n, d)
        Original compact backbone states.
    ode_states : (n, d_ode + 4)
        Jump ODE output from build_jump_ode_state.

    Returns
    -------
    dict with:
        - state_correlation: mean Pearson correlation between original
          and ODE states (first min(d, d_ode) dims)
        - drift_energy: mean drift norm
        - jump_density: fraction of rows with detected jumps
        - smoothness_mean: mean state smoothness
    """
    if compact_states.shape[0] == 0 or ode_states.shape[0] == 0:
        return {
            "state_correlation": 0.0,
            "drift_energy": 0.0,
            "jump_density": 0.0,
            "smoothness_mean": 0.0,
        }

    # ODE state layout: [ode_dims..., drift_norm, jump_count, cum_energy, smoothness]
    n_ode = ode_states.shape[1] - 4
    ode_dims = ode_states[:, :n_ode]
    drift_norm = ode_states[:, n_ode]
    jump_count = ode_states[:, n_ode + 1]
    smoothness = ode_states[:, n_ode + 3]

    # Correlation between original and ODE states
    overlap = min(compact_states.shape[1], n_ode)
    if overlap > 0:
        corrs = []
        for dim in range(overlap):
            x = compact_states[:, dim]
            y = ode_dims[:, dim]
            if np.std(x) > 1e-8 and np.std(y) > 1e-8:
                corrs.append(float(np.corrcoef(x, y)[0, 1]))
        state_corr = float(np.mean(corrs)) if corrs else 0.0
    else:
        state_corr = 0.0

    return {
        "state_correlation": state_corr,
        "drift_energy": float(np.mean(drift_norm)),
        "jump_density": float(np.mean(jump_count > 0)) if len(jump_count) > 0 else 0.0,
        "smoothness_mean": float(np.mean(smoothness)),
    }
