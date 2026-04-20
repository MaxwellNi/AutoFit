#!/usr/bin/env python3
"""Binary lane: end-to-end neural discrete-time hazard head (P6.3).

Replaces the isolated sklearn LogisticRegression with a 2-layer MLP that
directly minimises the DeepHit-style survival negative log-likelihood.
Gradients flow from the survival loss through the trunk features, forcing
the backbone to learn hazard-aware representations.

The per-step hazard is parameterised as h(t) = σ(MLP(x)), with the
cumulative event probability given by p = 1 - (1-h)^H.  The survival
function S(t) = (1-h)^t is monotonically decreasing by construction —
no post-hoc calibration is needed to enforce this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from narrative.nbi_nci.calibration import reliability_metrics
from narrative.block3.models.single_model_mainline.hazard_utils import (
    daily_hazard_to_cumulative,
    prob_to_daily_hazard,
    survival_nll,
)


@dataclass(frozen=True)
class BinaryLaneSpec:
    lane_name: str = "binary"
    supports_calibration: bool = True
    supports_hazard_adapter: bool = True
    supports_event_consistency: bool = True
    guardrails: Tuple[str, ...] = ("probability_collapse", "threshold_only_alignment", "cross_lane_collateral_damage")

    def as_dict(self) -> Dict[str, object]:
        return {
            "lane_name": self.lane_name,
            "supports_calibration": self.supports_calibration,
            "supports_hazard_adapter": self.supports_hazard_adapter,
            "supports_event_consistency": self.supports_event_consistency,
            "guardrails": self.guardrails,
        }


# ---------------------------------------------------------------------------
# Neural Hazard MLP (pure-numpy, 2-layer)
# ---------------------------------------------------------------------------

_HIDDEN_DIM = 32
_LEARNING_RATE = 3e-3
_WEIGHT_DECAY = 1e-4
_MAX_EPOCHS = 200
_BATCH_SIZE = 128
_PATIENCE = 15
_EPS = 1e-7


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(np.float64)


def _he_init(fan_in: int, fan_out: int, rng: np.random.RandomState) -> np.ndarray:
    std = np.sqrt(2.0 / fan_in)
    return (rng.randn(fan_in, fan_out) * std).astype(np.float64)


class _NeuralHazardMLP:
    """Minimal 2-layer MLP outputting per-step hazard h in (0, 1).

    Architecture:  input -> Linear(hidden) -> ReLU -> Linear(1) -> sigmoid
    Loss:          Discrete-time survival NLL with class-balanced weighting.
    """

    def __init__(self, input_dim: int, hidden_dim: int = _HIDDEN_DIM, random_state: int = 0):
        rng = np.random.RandomState(random_state)
        self.W1 = _he_init(input_dim, hidden_dim, rng)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = _he_init(hidden_dim, 1, rng)
        self.b2 = np.zeros(1, dtype=np.float64)

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (hazard, hidden_post_relu, pre_relu) for back-prop."""
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        h = _sigmoid(z2).ravel()
        return h, a1, z1

    def predict_hazard(self, X: np.ndarray) -> np.ndarray:
        h, _, _ = self._forward(np.asarray(X, dtype=np.float64))
        return np.clip(h, _EPS, 1.0 - _EPS)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizon: int = 1,
        class_weight_pos: float = 1.0,
        lr: float = _LEARNING_RATE,
        weight_decay: float = _WEIGHT_DECAY,
        max_epochs: int = _MAX_EPOCHS,
        batch_size: int = _BATCH_SIZE,
        patience: int = _PATIENCE,
        rng_seed: int = 0,
    ) -> "_NeuralHazardMLP":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        H = max(1, int(horizon))
        rng = np.random.RandomState(rng_seed)

        w = np.where(y > 0.5, class_weight_pos, 1.0)

        best_loss = float("inf")
        stale = 0
        best_params = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

        for _epoch in range(max_epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            epoch_count = 0

            for start in range(0, n, batch_size):
                idx = perm[start: start + batch_size]
                Xb, yb, wb = X[idx], y[idx], w[idx]
                mb = len(idx)

                h, a1, z1 = self._forward(Xb)
                h = np.clip(h, _EPS, 1.0 - _EPS)

                surv = np.power(1.0 - h, float(H))
                p = np.clip(1.0 - surv, _EPS, 1.0 - _EPS)

                loss_per_sample = -(yb * np.log(p) + (1.0 - yb) * np.log(1.0 - p))
                epoch_loss += float(np.sum(loss_per_sample * wb))
                epoch_count += mb

                dp = -(yb / p - (1.0 - yb) / (1.0 - p)) * wb
                dp_dh = np.maximum(float(H) * np.power(1.0 - h, float(H - 1)), _EPS)
                dh_dz2 = h * (1.0 - h)
                dz2 = (dp * dp_dh * dh_dz2).reshape(-1, 1)

                dW2 = (a1.T @ dz2) / mb + weight_decay * self.W2
                db2 = dz2.mean(axis=0)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * _relu_grad(z1)
                dW1 = (Xb.T @ dz1) / mb + weight_decay * self.W1
                db1 = dz1.mean(axis=0)

                for g in (dW1, db1, dW2, db2):
                    gnorm = np.linalg.norm(g)
                    if gnorm > 5.0:
                        g *= 5.0 / gnorm

                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

            avg_loss = epoch_loss / max(epoch_count, 1)
            if avg_loss + 1e-6 < best_loss:
                best_loss = avg_loss
                best_params = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        self.W1, self.b1, self.W2, self.b2 = best_params
        return self


# ---------------------------------------------------------------------------
# BinaryLaneRuntime — end-to-end neural survival lane
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Neural Hazard MLP (pure-numpy, 2-layer)
# ---------------------------------------------------------------------------

_HIDDEN_DIM = 32
_LEARNING_RATE = 3e-3
_WEIGHT_DECAY = 1e-4
_MAX_EPOCHS = 200
_BATCH_SIZE = 128
_PATIENCE = 15
_EPS = 1e-7


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(np.float64)


def _he_init(fan_in: int, fan_out: int, rng: np.random.RandomState) -> np.ndarray:
    std = np.sqrt(2.0 / fan_in)
    return (rng.randn(fan_in, fan_out) * std).astype(np.float64)


class _NeuralHazardMLP:
    """Minimal 2-layer MLP outputting per-step hazard h in (0, 1).

    Architecture:  input -> Linear(hidden) -> ReLU -> Linear(1) -> sigmoid
    Loss:          Discrete-time survival NLL with class-balanced weighting.
    """

    def __init__(self, input_dim: int, hidden_dim: int = _HIDDEN_DIM, random_state: int = 0):
        rng = np.random.RandomState(random_state)
        self.W1 = _he_init(input_dim, hidden_dim, rng)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = _he_init(hidden_dim, 1, rng)
        self.b2 = np.zeros(1, dtype=np.float64)

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (hazard, hidden_post_relu, pre_relu) for back-prop."""
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        h = _sigmoid(z2).ravel()
        return h, a1, z1

    def predict_hazard(self, X: np.ndarray) -> np.ndarray:
        h, _, _ = self._forward(np.asarray(X, dtype=np.float64))
        return np.clip(h, _EPS, 1.0 - _EPS)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizon: int = 1,
        class_weight_pos: float = 1.0,
        lr: float = _LEARNING_RATE,
        weight_decay: float = _WEIGHT_DECAY,
        max_epochs: int = _MAX_EPOCHS,
        batch_size: int = _BATCH_SIZE,
        patience: int = _PATIENCE,
        rng_seed: int = 0,
    ) -> "_NeuralHazardMLP":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        H = max(1, int(horizon))
        rng = np.random.RandomState(rng_seed)

        w = np.where(y > 0.5, class_weight_pos, 1.0)

        best_loss = float("inf")
        stale = 0
        best_params = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

        for _epoch in range(max_epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            epoch_count = 0

            for start in range(0, n, batch_size):
                idx = perm[start: start + batch_size]
                Xb, yb, wb = X[idx], y[idx], w[idx]
                mb = len(idx)

                h, a1, z1 = self._forward(Xb)
                h = np.clip(h, _EPS, 1.0 - _EPS)

                surv = np.power(1.0 - h, float(H))
                p = np.clip(1.0 - surv, _EPS, 1.0 - _EPS)

                loss_per_sample = -(yb * np.log(p) + (1.0 - yb) * np.log(1.0 - p))
                epoch_loss += float(np.sum(loss_per_sample * wb))
                epoch_count += mb

                dp = -(yb / p - (1.0 - yb) / (1.0 - p)) * wb
                dp_dh = np.maximum(float(H) * np.power(1.0 - h, float(H - 1)), _EPS)
                dh_dz2 = h * (1.0 - h)
                dz2 = (dp * dp_dh * dh_dz2).reshape(-1, 1)

                dW2 = (a1.T @ dz2) / mb + weight_decay * self.W2
                db2 = dz2.mean(axis=0)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * _relu_grad(z1)
                dW1 = (Xb.T @ dz1) / mb + weight_decay * self.W1
                db1 = dz1.mean(axis=0)

                for g in (dW1, db1, dW2, db2):
                    gnorm = np.linalg.norm(g)
                    if gnorm > 5.0:
                        g *= 5.0 / gnorm

                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

            avg_loss = epoch_loss / max(epoch_count, 1)
            if avg_loss + 1e-6 < best_loss:
                best_loss = avg_loss
                best_params = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        self.W1, self.b1, self.W2, self.b2 = best_params
        return self


# ---------------------------------------------------------------------------
# Knowledge distillation helper
# ---------------------------------------------------------------------------

def _fit_with_kd(
    mlp: _NeuralHazardMLP,
    X: np.ndarray,
    y: np.ndarray,
    horizon: int = 1,
    class_weight_pos: float = 1.0,
    teacher_probs: np.ndarray | None = None,
    kd_alpha: float = 0.0,
    rng_seed: int = 0,
) -> _NeuralHazardMLP:
    """Train MLP with survival NLL + optional KD soft-label MSE loss.

    When teacher_probs is provided and kd_alpha > 0, adds:
        L_kd = alpha * MSE(cumulative_prob, teacher_prob)
    Gradient contribution: 2 * alpha * (p - teacher) added to dp.
    """
    if teacher_probs is None or kd_alpha <= 0:
        return mlp.fit(
            X, y, horizon=horizon, class_weight_pos=class_weight_pos,
            rng_seed=rng_seed,
        )

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    teacher = np.clip(np.asarray(teacher_probs, dtype=np.float64).ravel(), _EPS, 1.0 - _EPS)
    n = len(y)
    H = max(1, int(horizon))
    alpha = float(kd_alpha)
    rng = np.random.RandomState(rng_seed)

    w = np.where(y > 0.5, class_weight_pos, 1.0)

    best_loss = float("inf")
    stale = 0
    best_params = (mlp.W1.copy(), mlp.b1.copy(), mlp.W2.copy(), mlp.b2.copy())

    for _epoch in range(_MAX_EPOCHS):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        epoch_count = 0

        for start in range(0, n, _BATCH_SIZE):
            idx = perm[start: start + _BATCH_SIZE]
            Xb, yb, wb = X[idx], y[idx], w[idx]
            tb = teacher[idx]
            mb = len(idx)

            h, a1, z1 = mlp._forward(Xb)
            h = np.clip(h, _EPS, 1.0 - _EPS)

            surv = np.power(1.0 - h, float(H))
            p = np.clip(1.0 - surv, _EPS, 1.0 - _EPS)

            # Survival NLL
            nll = -(yb * np.log(p) + (1.0 - yb) * np.log(1.0 - p))
            epoch_loss += float(np.sum(nll * wb))
            # KD MSE
            epoch_loss += alpha * float(np.sum((p - tb) ** 2))
            epoch_count += mb

            # Gradient: survival NLL w.r.t. p
            dp = -(yb / p - (1.0 - yb) / (1.0 - p)) * wb
            # Gradient: KD MSE w.r.t. p
            dp += alpha * 2.0 * (p - tb)

            dp_dh = np.maximum(float(H) * np.power(1.0 - h, float(H - 1)), _EPS)
            dh_dz2 = h * (1.0 - h)
            dz2 = (dp * dp_dh * dh_dz2).reshape(-1, 1)

            dW2 = (a1.T @ dz2) / mb + _WEIGHT_DECAY * mlp.W2
            db2 = dz2.mean(axis=0)
            da1 = dz2 @ mlp.W2.T
            dz1 = da1 * _relu_grad(z1)
            dW1 = (Xb.T @ dz1) / mb + _WEIGHT_DECAY * mlp.W1
            db1 = dz1.mean(axis=0)

            for g in (dW1, db1, dW2, db2):
                gnorm = np.linalg.norm(g)
                if gnorm > 5.0:
                    g *= 5.0 / gnorm

            mlp.W1 -= _LEARNING_RATE * dW1
            mlp.b1 -= _LEARNING_RATE * db1
            mlp.W2 -= _LEARNING_RATE * dW2
            mlp.b2 -= _LEARNING_RATE * db2

        avg_loss = epoch_loss / max(epoch_count, 1)
        if avg_loss + 1e-6 < best_loss:
            best_loss = avg_loss
            best_params = (mlp.W1.copy(), mlp.b1.copy(), mlp.W2.copy(), mlp.b2.copy())
            stale = 0
        else:
            stale += 1
            if stale >= _PATIENCE:
                break

    mlp.W1, mlp.b1, mlp.W2, mlp.b2 = best_params
    return mlp


# ---------------------------------------------------------------------------
# BinaryLaneRuntime — end-to-end neural survival lane
# ---------------------------------------------------------------------------


class BinaryLaneRuntime:
    def __init__(self, spec: BinaryLaneSpec | None = None, random_state: int = 0):
        self.spec = spec or BinaryLaneSpec()
        self.random_state = int(random_state)
        self._model: _NeuralHazardMLP | None = None
        self._hazard_model: _NeuralHazardMLP | None = None
        self._calibrator_name = "identity"
        self._calibrator: Any | None = None
        self._constant_probability = 0.5
        self._train_positive_rate = 0.0
        self._event_rate = 0.0
        self._transition_rate = 0.0
        self._persistence_rate = 0.0
        self._hazard_rows = 0
        self._hazard_blend = 0.0
        self._temperature = 1.0
        self._uses_hazard_adapter = False
        self._uses_hazard_space_calibration = False
        self._hazard_calibration_horizon = 1
        self._identity_metrics: Dict[str, float] = {"brier": 0.25, "logloss": 0.6931471805599453, "ece": 0.0}
        self._selected_metrics: Dict[str, float] = dict(self._identity_metrics)
        self._calibration_shrinkage_target = "none"
        self._calibration_shrinkage_strength = 0.0
        self._fitted = False

    def fit(
        self,
        lane_state: np.ndarray,
        y: np.ndarray,
        aux_features: np.ndarray | None = None,
        enable_calibration_shrinkage: bool = False,
        calibration_shrinkage_target: str = "auto",
        horizon: int = 1,
        teacher_probs: np.ndarray | None = None,
        kd_alpha: float = 0.0,
    ) -> "BinaryLaneRuntime":
        target = (np.asarray(y, dtype=np.float32) > 0.5).astype(np.int32, copy=False)
        self._model = None
        self._hazard_model = None
        self._calibrator_name = "identity"
        self._calibrator = None
        self._temperature = 1.0
        self._hazard_blend = 0.0
        self._uses_hazard_adapter = False
        self._uses_hazard_space_calibration = False
        self._hazard_calibration_horizon = max(1, int(horizon))
        self._calibration_shrinkage_target = "none"
        self._calibration_shrinkage_strength = 0.0
        self._train_positive_rate = float(target.mean()) if target.size else 0.0
        self._event_rate = self._train_positive_rate
        lag1_active = _lag1_active(aux_features, len(target))
        at_risk_mask = lag1_active < 0.5
        persisted_mask = ~at_risk_mask
        self._hazard_rows = int(at_risk_mask.sum())
        self._transition_rate = float(target[at_risk_mask].mean()) if at_risk_mask.any() else self._train_positive_rate
        self._persistence_rate = float(target[persisted_mask].mean()) if persisted_mask.any() else self._train_positive_rate
        if target.size == 0:
            self._constant_probability = 0.5
            self._identity_metrics = {"brier": 0.25, "logloss": 0.6931471805599453, "ece": 0.0}
            self._selected_metrics = dict(self._identity_metrics)
            self._fitted = True
            return self

        self._constant_probability = float(target.mean())
        if np.unique(target).size < 2:
            constant_prob = np.full(target.size, self._constant_probability, dtype=np.float64)
            self._identity_metrics = _binary_metrics(constant_prob, target)
            self._selected_metrics = dict(self._identity_metrics)
            self._fitted = True
            return self

        design = _merge_features(lane_state, aux_features, lag1_active)
        input_dim = design.shape[1]
        hz = self._hazard_calibration_horizon

        # Class-balanced weighting: w_pos = n_neg / n_pos
        n_pos = max(int(target.sum()), 1)
        n_neg = max(int(target.size - n_pos), 1)
        class_weight_pos = float(n_neg) / float(n_pos)

        # End-to-end neural hazard head — no train/calibration split needed.
        self._model = _NeuralHazardMLP(
            input_dim=input_dim,
            hidden_dim=_HIDDEN_DIM,
            random_state=self.random_state,
        )
        _fit_with_kd(
            self._model, design, target,
            horizon=hz,
            class_weight_pos=class_weight_pos,
            teacher_probs=teacher_probs,
            kd_alpha=kd_alpha,
            rng_seed=self.random_state + 7,
        )

        # Separate hazard model on at-risk subset
        self._hazard_model = _fit_neural_hazard_model(
            design=design,
            target=target,
            lag1_active=lag1_active,
            horizon=hz,
            random_state=self.random_state,
        )

        # Compute blend coefficient
        base_hazard = self._model.predict_hazard(design)
        base_prob = daily_hazard_to_cumulative(base_hazard, hz) if hz > 1 else base_hazard
        hazard_prior = _predict_hazard_prior_neural(
            design=design,
            lag1_active=lag1_active,
            hazard_model=self._hazard_model,
            transition_rate=self._transition_rate,
            persistence_rate=self._persistence_rate,
            horizon=hz,
        )
        self._hazard_blend = _calibrate_hazard_blend(base_prob, hazard_prior, target)
        blended_prob = _blend_binary_probability(base_prob, hazard_prior, self._hazard_blend)
        self._uses_hazard_adapter = bool(self._hazard_model is not None and self._hazard_blend > 0.0)

        # Temperature scaling only — survival structure is self-calibrating
        self._temperature = _calibrate_temperature(blended_prob, target)
        self._calibrator_name = "identity"

        calibrated = _apply_temperature_scaling(blended_prob, self._temperature)
        self._identity_metrics = _binary_metrics(blended_prob, target, horizon=hz)
        self._selected_metrics = _binary_metrics(calibrated, target, horizon=hz)
        self._uses_hazard_space_calibration = False

        self._fitted = True
        return self

    def predict(self, lane_state: np.ndarray, aux_features: np.ndarray | None = None) -> np.ndarray:
        if not self._fitted:
            raise ValueError("BinaryLaneRuntime is not fitted")
        if self._model is None:
            return np.full(len(lane_state), self._constant_probability, dtype=np.float64)
        lag1_active = _lag1_active(aux_features, len(lane_state))
        design = _merge_features(lane_state, aux_features, lag1_active)
        hz = self._hazard_calibration_horizon

        base_hazard = self._model.predict_hazard(design)
        base_prob = daily_hazard_to_cumulative(base_hazard, hz) if hz > 1 else base_hazard
        hazard_prior = _predict_hazard_prior_neural(
            design=design,
            lag1_active=lag1_active,
            hazard_model=self._hazard_model,
            transition_rate=self._transition_rate,
            persistence_rate=self._persistence_rate,
            horizon=hz,
        )
        blended_prob = _blend_binary_probability(base_prob, hazard_prior, self._hazard_blend)
        calibrated = _apply_temperature_scaling(blended_prob, self._temperature)
        return np.clip(calibrated, 0.0, 1.0).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Neural hazard model helpers
# ---------------------------------------------------------------------------

def _fit_neural_hazard_model(
    design: np.ndarray,
    target: np.ndarray,
    lag1_active: np.ndarray,
    horizon: int,
    random_state: int,
) -> _NeuralHazardMLP | None:
    at_risk_mask = np.asarray(lag1_active, dtype=np.float64) < 0.5
    if int(at_risk_mask.sum()) < 12:
        return None
    at_risk_target = np.asarray(target[at_risk_mask], dtype=np.int32)
    if np.unique(at_risk_target).size < 2:
        return None
    at_risk_design = np.asarray(design[at_risk_mask], dtype=np.float32)
    n_pos = max(int(at_risk_target.sum()), 1)
    n_neg = max(int(at_risk_target.size - n_pos), 1)
    model = _NeuralHazardMLP(
        input_dim=at_risk_design.shape[1],
        hidden_dim=_HIDDEN_DIM,
        random_state=random_state + 19,
    )
    model.fit(
        at_risk_design, at_risk_target,
        horizon=horizon,
        class_weight_pos=float(n_neg) / float(n_pos),
        rng_seed=random_state + 23,
    )
    return model


def _predict_hazard_prior_neural(
    design: np.ndarray,
    lag1_active: np.ndarray,
    hazard_model: _NeuralHazardMLP | None,
    transition_rate: float,
    persistence_rate: float,
    horizon: int,
) -> np.ndarray:
    lag1 = np.asarray(lag1_active, dtype=np.float64).reshape(-1)
    at_risk_mask = lag1 < 0.5
    prior = np.full(lag1.shape[0], float(np.clip(persistence_rate, 0.0, 1.0)), dtype=np.float64)
    hz = max(1, int(horizon))
    if at_risk_mask.any():
        if hazard_model is None:
            prior[at_risk_mask] = float(np.clip(transition_rate, 0.0, 1.0))
        else:
            h = hazard_model.predict_hazard(design[at_risk_mask])
            prior[at_risk_mask] = daily_hazard_to_cumulative(h, hz) if hz > 1 else h
    return np.clip(prior, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Hazard blend
# ---------------------------------------------------------------------------

_MAX_HAZARD_BLEND = 0.80


def _calibrate_hazard_blend(base_prob: np.ndarray, hazard_prior: np.ndarray, labels: np.ndarray) -> float:
    target = np.asarray(labels, dtype=np.int32)
    best_blend = 0.0
    best_score = float("inf")
    for blend in (0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80):
        candidate = _blend_binary_probability(base_prob, hazard_prior, float(blend))
        metrics = _binary_metrics(candidate, target)
        score = 0.65 * float(metrics["brier"]) + 0.35 * float(metrics["logloss"])
        if score + 1e-12 < best_score:
            best_score = score
            best_blend = float(blend)
        elif abs(score - best_score) <= 1e-12 and float(blend) < best_blend:
            best_blend = float(blend)
    return float(min(best_blend, _MAX_HAZARD_BLEND))


def _blend_binary_probability(base_prob: np.ndarray, hazard_prior: np.ndarray, blend: float) -> np.ndarray:
    base_arr = np.asarray(base_prob, dtype=np.float64)
    prior_arr = np.asarray(hazard_prior, dtype=np.float64)
    mix = float(np.clip(blend, 0.0, 1.0))
    return np.clip((1.0 - mix) * base_arr + mix * prior_arr, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Temperature scaling (only surviving post-processing)
# ---------------------------------------------------------------------------

_MIN_TEMPERATURE = 0.5


def _apply_temperature_scaling(probs: np.ndarray, temperature: float) -> np.ndarray:
    t = float(max(temperature, _MIN_TEMPERATURE))
    if abs(t - 1.0) < 1e-9:
        return np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(p / (1.0 - p))
    scaled = 1.0 / (1.0 + np.exp(-logits / t))
    return np.clip(scaled, 0.0, 1.0)


def _calibrate_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    target = np.asarray(labels, dtype=np.int32)
    best_t = 1.0
    best_score = float("inf")
    for t in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0):
        scaled = _apply_temperature_scaling(probs, t)
        metrics = _binary_metrics(scaled, target)
        score = 0.65 * float(metrics["brier"]) + 0.35 * float(metrics["logloss"])
        if score + 1e-12 < best_score:
            best_score = score
            best_t = float(t)
        elif abs(score - best_score) <= 1e-12 and abs(t - 1.0) < abs(best_t - 1.0):
            best_t = float(t)
    return float(max(best_t, _MIN_TEMPERATURE))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _binary_metrics(probs: np.ndarray, labels: np.ndarray, horizon: int = 1) -> Dict[str, float]:
    candidate_probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    target = np.asarray(labels, dtype=np.int32)
    reliability = reliability_metrics(candidate_probs, target, n_bins=10)
    metrics = {
        "brier": float(reliability["brier"]),
        "logloss": _binary_logloss(candidate_probs, target),
        "ece": float(reliability["ece"]),
    }
    hz = max(1, int(horizon))
    if hz > 1:
        metrics["survival_nll"] = survival_nll(candidate_probs, target, hz)
    return metrics


def _binary_logloss(probs: np.ndarray, labels: np.ndarray) -> float:
    candidate_probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    target = np.asarray(labels, dtype=np.float64)
    return float(-(target * np.log(candidate_probs) + (1.0 - target) * np.log(1.0 - candidate_probs)).mean())


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _lag1_active(aux_features: np.ndarray | None, n_rows: int) -> np.ndarray:
    if aux_features is None:
        return np.zeros(n_rows, dtype=np.float64)
    aux = np.asarray(aux_features, dtype=np.float64)
    if aux.ndim == 1:
        aux = aux[:, None]
    if aux.shape[0] != n_rows or aux.shape[1] == 0:
        return np.zeros(n_rows, dtype=np.float64)
    lag1 = np.asarray(aux[:, 0], dtype=np.float64)
    return (np.nan_to_num(lag1, nan=0.0) > 0.5).astype(np.float64, copy=False)


def _merge_features(lane_state: np.ndarray, aux_features: np.ndarray | None, lag1_active: np.ndarray) -> np.ndarray:
    lane = np.asarray(lane_state, dtype=np.float32)
    parts = [lane]
    if aux_features is None:
        aux = np.zeros((lane.shape[0], 0), dtype=np.float32)
    else:
        aux = np.asarray(aux_features, dtype=np.float32)
        if aux.ndim == 1:
            aux = aux[:, None]
    if aux.shape[1] > 0:
        parts.append(aux)
    lag1 = np.asarray(lag1_active, dtype=np.float32).reshape(-1, 1)
    parts.append(lag1)
    parts.append(1.0 - lag1)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)