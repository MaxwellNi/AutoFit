"""
Adaptive RL Policy for AutoFit V7.3 Hyperparameter Optimization.

Implements a factored Neural Contextual Bandit (LinUCB + Thompson Sampling)
that learns optimal hyperparameter configurations per routing condition.

Design:
    - State vector: encoding of (lane, horizon_band, ablation, missingness,
      target meta-features) → ~30-dim continuous vector
    - Action: factored across 5 independent hyperparameter dimensions
    - Reward: negative MAE on temporal CV holdout (normalized by naive baseline)
    - Update: online Bayesian linear regression per arm (LinUCB-style)

Architecture decisions:
    1. Factored bandits avoid combinatorial explosion (1280 → 5×5 = 25 arms)
    2. Ridge regression posterior for each arm → closed-form Thompson Sampling
    3. Warm-start from prior benchmark results via offline policy replay
    4. Exploration-exploitation tradeoff via UCB confidence bound

References:
    - Li et al. (2010) "A Contextual-Bandit Approach to Personalized News"
    - Riquelme et al. (2018) "Deep Bayesian Bandits Showdown" (NeurIPS)
    - Foster & Rakhlin (2020) "Beyond UCB: Optimal and Efficient Contextual Bandits"
"""

from __future__ import annotations

import json
import logging
import math
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Encoder
# ---------------------------------------------------------------------------

# Categorical feature vocabularies
_LANE_VOCAB = ["binary", "count", "heavy_tail", "general"]
_HORIZON_BAND_VOCAB = ["short", "mid", "long"]
_ABLATION_VOCAB = ["core_only", "core_text", "core_edgar", "full"]
_MISS_VOCAB = ["low", "medium", "high"]

# Continuous meta-feature keys expected from target regime analysis
_META_KEYS = [
    "kurtosis", "cv", "zero_frac", "n_unique_ratio",
    "exog_corr_max", "skewness", "iqr_ratio",
]

# Extended meta-features for V7.3 (from block3_profile_data.py)
_META_KEYS_V73 = _META_KEYS + [
    "nonstationarity_score", "periodicity_score", "multiscale_score",
    "long_memory_score", "irregular_score", "heavy_tail_score",
    "exog_strength", "text_strength",
]

# Total state dimension: 4 + 3 + 4 + 3 + 15 + 1(bias) = 30
STATE_DIM = (
    len(_LANE_VOCAB)
    + len(_HORIZON_BAND_VOCAB)
    + len(_ABLATION_VOCAB)
    + len(_MISS_VOCAB)
    + len(_META_KEYS_V73)
    + 1  # bias term
)


def _encode_state(
    lane: str,
    horizon_band: str,
    ablation: str,
    missingness: str,
    meta_features: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Encode routing context into a fixed-dim state vector."""
    parts = []

    # One-hot categoricals
    for vocab, val in [
        (_LANE_VOCAB, lane),
        (_HORIZON_BAND_VOCAB, horizon_band),
        (_ABLATION_VOCAB, ablation),
        (_MISS_VOCAB, missingness),
    ]:
        oh = np.zeros(len(vocab), dtype=np.float64)
        if val in vocab:
            oh[vocab.index(val)] = 1.0
        parts.append(oh)

    # Continuous meta features (use extended V7.3 keys)
    mf = meta_features or {}
    cont = np.zeros(len(_META_KEYS_V73), dtype=np.float64)
    for i, key in enumerate(_META_KEYS_V73):
        v = mf.get(key, 0.0)
        if v is None or not np.isfinite(v):
            v = 0.0
        # Clip to reasonable range
        cont[i] = np.clip(float(v), -10.0, 10.0)
    parts.append(cont)

    # Bias
    parts.append(np.array([1.0], dtype=np.float64))

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Hyperparameter Action Space (factored)
# ---------------------------------------------------------------------------

@dataclass
class HyperparamSpec:
    """Specification of a single hyperparameter axis."""
    name: str
    values: List[Any]
    default_idx: int = 0

    @property
    def n_arms(self) -> int:
        return len(self.values)


# 7 factored hyperparameter axes (5 original + 2 V7.3 champion-first)
HP_SPECS: List[HyperparamSpec] = [
    HyperparamSpec("top_k", [4, 6, 8, 10, 12, 16], default_idx=4),       # idx 4 = 12
    HyperparamSpec("moe_max_experts", [2, 3, 4, 5, 6], default_idx=3),   # idx 3 = 5
    HyperparamSpec("moe_temperature", [0.15, 0.25, 0.35, 0.45, 0.60], default_idx=2),  # idx 2 = 0.35
    HyperparamSpec("blend_alpha", [0.05, 0.15, 0.25, 0.35, 0.45], default_idx=2),       # idx 2 = 0.25
    HyperparamSpec("qs_threshold_mult", [0.75, 0.80, 0.85, 0.90, 0.95, 1.00], default_idx=3),  # idx 3 = 0.90
    # V7.3 champion-first axes
    HyperparamSpec("champion_gate_tol", [0.00, 0.005, 0.01, 0.02, 0.03], default_idx=0),
    # 0.00 = strict champion-first (ensemble must strictly beat single best)
    # 0.03 = legacy V71 tolerance
    HyperparamSpec("ensemble_diversity_min", [0.0, 0.05, 0.10, 0.15, 0.20], default_idx=1),
    # Minimum ensemble diversity (1 - mean_corr) required to accept ensemble over single
]


# ---------------------------------------------------------------------------
# Bayesian Linear Bandit Arm
# ---------------------------------------------------------------------------

class BayesianLinearArm:
    """
    Bayesian linear regression arm for contextual bandit.

    Maintains a Gaussian posterior N(mu, Sigma) for the weight vector w
    such that reward = x^T w + noise.

    Update rule (ridge regression):
        A = A + x x^T        (precision matrix)
        b = b + r * x        (response accumulator)
        mu = A^{-1} b        (posterior mean)
        Sigma = sigma^2 A^{-1}  (posterior covariance)

    Thompson Sampling: sample w ~ N(mu, Sigma), predict x^T w.
    UCB: x^T mu + alpha * sqrt(x^T Sigma x).
    """

    def __init__(self, dim: int, reg_lambda: float = 1.0, noise_sigma: float = 0.1):
        self._dim = dim
        self._lambda = reg_lambda
        self._sigma = noise_sigma
        self._A = reg_lambda * np.eye(dim, dtype=np.float64)  # precision
        self._b = np.zeros(dim, dtype=np.float64)             # response
        self._n_obs = 0

    def update(self, x: np.ndarray, reward: float) -> None:
        """Update posterior with (context, reward) observation."""
        self._A += np.outer(x, x)
        self._b += reward * x
        self._n_obs += 1

    def predict_ucb(self, x: np.ndarray, alpha: float = 1.0) -> float:
        """Return UCB estimate: mu^T x + alpha * sqrt(x^T A^{-1} x)."""
        A_inv = np.linalg.solve(self._A, np.eye(self._dim))
        mu = A_inv @ self._b
        mean = mu @ x
        var = x @ A_inv @ x
        return mean + alpha * math.sqrt(max(var, 0.0))

    def predict_thompson(self, x: np.ndarray, rng: np.random.RandomState) -> float:
        """Thompson Sampling: sample from posterior and predict."""
        A_inv = np.linalg.solve(self._A, np.eye(self._dim))
        mu = A_inv @ self._b
        try:
            cov = self._sigma ** 2 * A_inv
            # Ensure PSD
            cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(self._dim)
            w_sample = rng.multivariate_normal(mu, cov)
        except (np.linalg.LinAlgError, ValueError):
            w_sample = mu
        return w_sample @ x

    @property
    def n_observations(self) -> int:
        return self._n_obs


# ---------------------------------------------------------------------------
# Factored Contextual Bandit
# ---------------------------------------------------------------------------

class FactoredContextualBandit:
    """
    Factored contextual bandit for hyperparameter optimization.

    Each hyperparameter axis has its own set of BayesianLinearArm instances
    (one per discrete value). The arms share the same state encoding.

    Action selection:
        - If exploration phase (< min_observations): use default + explore
        - If exploitation phase: use UCB or Thompson Sampling per axis

    The factored design reduces action space from multiplicative to additive:
    |A| = Σ |A_i| instead of Π |A_i|.
    """

    def __init__(
        self,
        hp_specs: Optional[List[HyperparamSpec]] = None,
        strategy: str = "thompson",  # "thompson" or "ucb"
        ucb_alpha: float = 1.5,
        min_observations: int = 3,   # per arm before exploitation
        seed: int = 42,
        reg_lambda: float = 1.0,
        noise_sigma: float = 0.15,
    ):
        self._specs = hp_specs or HP_SPECS
        self._strategy = strategy
        self._ucb_alpha = ucb_alpha
        self._min_obs = min_observations
        self._rng = np.random.RandomState(seed)

        # Create arms: dict[hp_name][arm_idx] -> BayesianLinearArm
        self._arms: Dict[str, List[BayesianLinearArm]] = {}
        for spec in self._specs:
            self._arms[spec.name] = [
                BayesianLinearArm(STATE_DIM, reg_lambda, noise_sigma)
                for _ in range(spec.n_arms)
            ]

        # Action history for diagnostics
        self._history: List[Dict[str, Any]] = []

    def select_action(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Select hyperparameter configuration given state context.

        Returns dict mapping hp_name -> selected value.
        """
        action = {}
        action_indices = {}

        for spec in self._specs:
            arms = self._arms[spec.name]
            min_arm_obs = min(arm.n_observations for arm in arms)

            if min_arm_obs < self._min_obs:
                # Exploration phase: round-robin under-sampled arms
                least_observed = min(arm.n_observations for arm in arms)
                candidates = [
                    i for i, arm in enumerate(arms)
                    if arm.n_observations == least_observed
                ]
                idx = self._rng.choice(candidates)
            elif self._strategy == "thompson":
                scores = [arm.predict_thompson(state, self._rng) for arm in arms]
                idx = int(np.argmax(scores))
            else:  # ucb
                scores = [arm.predict_ucb(state, self._ucb_alpha) for arm in arms]
                idx = int(np.argmax(scores))

            action[spec.name] = spec.values[idx]
            action_indices[spec.name] = idx

        return action

    def update(
        self,
        state: np.ndarray,
        action: Dict[str, Any],
        reward: float,
    ) -> None:
        """Update all arms that participated in the action."""
        for spec in self._specs:
            val = action.get(spec.name)
            if val is None:
                continue
            try:
                idx = spec.values.index(val)
            except ValueError:
                continue
            self._arms[spec.name][idx].update(state, reward)

        self._history.append({
            "action": dict(action),
            "reward": float(reward),
        })

    def warm_start(self, records: List[Dict[str, Any]]) -> int:
        """
        Warm-start bandit from offline benchmark records.

        Each record should contain:
            - lane, horizon_band, ablation, missingness_bucket (for state)
            - meta_features (dict, optional)
            - top_k, moe_max_experts, moe_temperature, blend_alpha (for action)
            - reward (float, e.g. negative normalized MAE)

        Returns number of records processed.
        """
        n_processed = 0
        for rec in records:
            try:
                state = _encode_state(
                    lane=rec.get("lane", "general"),
                    horizon_band=rec.get("horizon_band", "mid"),
                    ablation=rec.get("ablation", "core_only"),
                    missingness=rec.get("missingness_bucket", "medium"),
                    meta_features=rec.get("meta_features"),
                )
                action = {}
                for spec in self._specs:
                    val = rec.get(spec.name)
                    if val is not None and val in spec.values:
                        action[spec.name] = val

                reward = float(rec.get("reward", 0.0))
                if not np.isfinite(reward):
                    continue

                if action:
                    self.update(state, action, reward)
                    n_processed += 1

            except Exception as e:
                logger.debug(f"[RLPolicy] Skipping warmstart record: {e}")
                continue

        logger.info(f"[RLPolicy] Warm-started from {n_processed}/{len(records)} records")
        return n_processed

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic summary of bandit state."""
        diag: Dict[str, Any] = {
            "strategy": self._strategy,
            "n_history": len(self._history),
        }
        for spec in self._specs:
            arms = self._arms[spec.name]
            obs_counts = [arm.n_observations for arm in arms]
            diag[f"{spec.name}_arm_obs"] = obs_counts
            diag[f"{spec.name}_total_obs"] = sum(obs_counts)
        return diag

    def serialize(self) -> Dict[str, Any]:
        """Serialize bandit state for persistence."""
        state_dict = {
            "strategy": self._strategy,
            "ucb_alpha": self._ucb_alpha,
            "min_observations": self._min_obs,
        }
        for spec in self._specs:
            arms = self._arms[spec.name]
            state_dict[spec.name] = {
                "values": spec.values,
                "arms": [
                    {
                        "A": arm._A.tolist(),
                        "b": arm._b.tolist(),
                        "n_obs": arm._n_obs,
                    }
                    for arm in arms
                ],
            }
        return state_dict

    @classmethod
    def deserialize(cls, state_dict: Dict[str, Any]) -> "FactoredContextualBandit":
        """Reconstruct bandit from serialized state."""
        bandit = cls(
            strategy=state_dict.get("strategy", "thompson"),
            ucb_alpha=state_dict.get("ucb_alpha", 1.5),
            min_observations=state_dict.get("min_observations", 3),
        )
        for spec in bandit._specs:
            if spec.name in state_dict:
                arms_data = state_dict[spec.name]["arms"]
                for i, arm_data in enumerate(arms_data):
                    if i < len(bandit._arms[spec.name]):
                        arm = bandit._arms[spec.name][i]
                        arm._A = np.array(arm_data["A"], dtype=np.float64)
                        arm._b = np.array(arm_data["b"], dtype=np.float64)
                        arm._n_obs = arm_data["n_obs"]
        return bandit


# ---------------------------------------------------------------------------
# Adaptive Policy Controller (integrates with V73 fit())
# ---------------------------------------------------------------------------

@dataclass
class PolicyDecision:
    """Result of an RL policy query."""
    top_k: int = 12
    moe_max_experts: int = 5
    moe_temperature: float = 0.35
    blend_alpha: float = 0.25
    qs_threshold_mult: float = 0.90
    champion_gate_tol: float = 0.00  # V7.3: strict champion-first by default
    ensemble_diversity_min: float = 0.05  # V7.3: minimum diversity to prefer ensemble
    confidence: float = 0.0
    source: str = "default"  # "default", "bandit", "replay", "warm_start"
    action_dict: Dict[str, Any] = field(default_factory=dict)


class AdaptiveRLPolicy:
    """
    High-level RL policy controller for V73.

    Usage in V73.fit():
        1. policy = AdaptiveRLPolicy.from_replay_buffer(path)
        2. decision = policy.query(lane, horizon_band, ablation, miss, meta)
        3. ... use decision.top_k, decision.moe_max_experts, etc. ...
        4. policy.observe(state, action, reward=-normalized_mae)

    The policy combines:
        - Offline replay buffer (from previous benchmark runs)
        - Online contextual bandit (adapts during current run)
        - Condition-specific overrides (hard-coded heuristics for known patterns)
    """

    def __init__(
        self,
        bandit: Optional[FactoredContextualBandit] = None,
        replay_buffer: Optional[List[Dict[str, Any]]] = None,
        enable_online_learning: bool = True,
        seed: int = 42,
    ):
        self._bandit = bandit or FactoredContextualBandit(seed=seed)
        self._replay = replay_buffer or []
        self._enable_online = enable_online_learning
        self._condition_cache: Dict[str, PolicyDecision] = {}

        # Auto warm-start from replay buffer
        if self._replay:
            self._bandit.warm_start(self._replay)

    @classmethod
    def from_replay_buffer(cls, path: Path, **kwargs) -> "AdaptiveRLPolicy":
        """Load replay buffer from JSON file."""
        replay = []
        if path.exists():
            try:
                with open(path, "r") as f:
                    replay = json.load(f)
                logger.info(f"[RLPolicy] Loaded {len(replay)} replay records from {path}")
            except Exception as e:
                logger.warning(f"[RLPolicy] Failed to load replay buffer: {e}")
        return cls(replay_buffer=replay, **kwargs)

    def query(
        self,
        lane: str,
        horizon_band: str,
        ablation: str,
        missingness: str,
        meta_features: Optional[Dict[str, float]] = None,
    ) -> PolicyDecision:
        """
        Query the policy for optimal hyperparameters given context.

        Returns PolicyDecision with selected values and confidence.
        """
        # Check condition-specific cache
        cache_key = f"{lane}|{horizon_band}|{ablation}|{missingness}"
        if cache_key in self._condition_cache:
            cached = self._condition_cache[cache_key]
            logger.info(f"[RLPolicy] Cache hit for {cache_key}: {cached.source}")
            return cached

        # Apply condition-specific heuristics first
        heuristic = self._apply_heuristics(lane, horizon_band, ablation, missingness)
        if heuristic is not None:
            self._condition_cache[cache_key] = heuristic
            return heuristic

        # Encode state and query bandit
        state = _encode_state(lane, horizon_band, ablation, missingness, meta_features)

        if self._bandit._history or any(
            arm.n_observations > 0
            for arms in self._bandit._arms.values()
            for arm in arms
        ):
            # Bandit has data → use it
            action = self._bandit.select_action(state)
            decision = PolicyDecision(
                top_k=action.get("top_k", 12),
                moe_max_experts=action.get("moe_max_experts", 5),
                moe_temperature=action.get("moe_temperature", 0.35),
                blend_alpha=action.get("blend_alpha", 0.25),
                qs_threshold_mult=action.get("qs_threshold_mult", 0.90),
                champion_gate_tol=action.get("champion_gate_tol", 0.00),
                ensemble_diversity_min=action.get("ensemble_diversity_min", 0.05),
                confidence=min(1.0, len(self._bandit._history) / 50.0),
                source="bandit",
                action_dict=action,
            )
        else:
            # No data → defaults optimized from domain knowledge
            action = self._domain_defaults(lane)
            decision = PolicyDecision(
                top_k=action.get("top_k", 12),
                moe_max_experts=action.get("moe_max_experts", 5),
                moe_temperature=action.get("moe_temperature", 0.35),
                blend_alpha=action.get("blend_alpha", 0.25),
                qs_threshold_mult=action.get("qs_threshold_mult", 0.90),
                champion_gate_tol=action.get("champion_gate_tol", 0.00),
                ensemble_diversity_min=action.get("ensemble_diversity_min", 0.05),
                confidence=0.0,
                source="default",
                action_dict=action,
            )

        logger.info(
            f"[RLPolicy] Decision for {cache_key}: "
            f"top_k={decision.top_k}, moe={decision.moe_max_experts}, "
            f"temp={decision.moe_temperature:.2f}, source={decision.source}"
        )
        return decision

    def observe(
        self,
        lane: str,
        horizon_band: str,
        ablation: str,
        missingness: str,
        action: Dict[str, Any],
        reward: float,
        meta_features: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Observe outcome for online learning.

        Args:
            reward: Should be normalized to [-1, 0] where 0 = perfect, -1 = worst.
        """
        if not self._enable_online:
            return

        state = _encode_state(lane, horizon_band, ablation, missingness, meta_features)
        self._bandit.update(state, action, reward)

        logger.info(
            f"[RLPolicy] Observed reward={reward:.4f} for "
            f"lane={lane}|hb={horizon_band}|abl={ablation}|miss={missingness}"
        )

    def _apply_heuristics(
        self, lane: str, horizon_band: str, ablation: str, missingness: str
    ) -> Optional[PolicyDecision]:
        """
        Hard-coded overrides for well-understood conditions.

        These encode domain knowledge from Phase 7 benchmark analysis:
        - Count lane + short horizon: tight ensemble, low temperature
        - Binary lane: small ensemble, high MoE experts
        - Heavy-tail + high missingness: aggressive exploration
        """
        # Count lane with short horizon: tight focused ensemble
        if lane == "count" and horizon_band == "short":
            return PolicyDecision(
                top_k=8,
                moe_max_experts=4,
                moe_temperature=0.25,
                blend_alpha=0.30,
                qs_threshold_mult=0.85,
                champion_gate_tol=0.00,
                ensemble_diversity_min=0.10,
                confidence=0.85,
                source="heuristic_count_short",
                action_dict={
                    "top_k": 8, "moe_max_experts": 4,
                    "moe_temperature": 0.25, "blend_alpha": 0.30,
                    "qs_threshold_mult": 0.85,
                    "champion_gate_tol": 0.00,
                    "ensemble_diversity_min": 0.10,
                },
            )

        # Binary lane: small sharp ensemble
        if lane == "binary":
            return PolicyDecision(
                top_k=6,
                moe_max_experts=3,
                moe_temperature=0.20,
                blend_alpha=0.15,
                qs_threshold_mult=0.80,
                champion_gate_tol=0.00,
                ensemble_diversity_min=0.05,
                confidence=0.80,
                source="heuristic_binary",
                action_dict={
                    "top_k": 6, "moe_max_experts": 3,
                    "moe_temperature": 0.20, "blend_alpha": 0.15,
                    "qs_threshold_mult": 0.80,
                    "champion_gate_tol": 0.00,
                    "ensemble_diversity_min": 0.05,
                },
            )

        # Heavy-tail with high missingness: aggressive exploration
        if lane == "heavy_tail" and missingness == "high":
            return PolicyDecision(
                top_k=16,
                moe_max_experts=6,
                moe_temperature=0.50,
                blend_alpha=0.35,
                qs_threshold_mult=0.95,
                champion_gate_tol=0.00,
                ensemble_diversity_min=0.10,
                confidence=0.75,
                source="heuristic_heavy_tail_high_miss",
                action_dict={
                    "top_k": 16, "moe_max_experts": 6,
                    "moe_temperature": 0.50, "blend_alpha": 0.35,
                    "qs_threshold_mult": 0.95,
                    "champion_gate_tol": 0.00,
                    "ensemble_diversity_min": 0.10,
                },
            )

        return None

    def _domain_defaults(self, lane: str) -> Dict[str, Any]:
        """Default configuration tuned per lane from domain analysis."""
        if lane == "count":
            return {
                "top_k": 10, "moe_max_experts": 5,
                "moe_temperature": 0.30, "blend_alpha": 0.25,
                "qs_threshold_mult": 0.85,
                "champion_gate_tol": 0.00,
                "ensemble_diversity_min": 0.10,
            }
        elif lane == "heavy_tail":
            return {
                "top_k": 12, "moe_max_experts": 5,
                "moe_temperature": 0.40, "blend_alpha": 0.30,
                "qs_threshold_mult": 0.95,
                "champion_gate_tol": 0.00,
                "ensemble_diversity_min": 0.10,
            }
        elif lane == "binary":
            return {
                "top_k": 8, "moe_max_experts": 4,
                "moe_temperature": 0.25, "blend_alpha": 0.20,
                "qs_threshold_mult": 0.85,
                "champion_gate_tol": 0.00,
                "ensemble_diversity_min": 0.05,
            }
        else:  # general
            return {
                "top_k": 12, "moe_max_experts": 5,
                "moe_temperature": 0.35, "blend_alpha": 0.25,
                "qs_threshold_mult": 0.90,
                "champion_gate_tol": 0.00,
                "ensemble_diversity_min": 0.05,
            }

    def save_state(self, path: Path) -> None:
        """Persist bandit state + replay history."""
        state = {
            "bandit": self._bandit.serialize(),
            "replay_size": len(self._replay),
            "cache_keys": list(self._condition_cache.keys()),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"[RLPolicy] Saved state to {path}")

    @classmethod
    def load_state(cls, path: Path) -> "AdaptiveRLPolicy":
        """Load persisted bandit state."""
        with open(path, "r") as f:
            state = json.load(f)
        bandit = FactoredContextualBandit.deserialize(state["bandit"])
        return cls(bandit=bandit, enable_online_learning=True)


# ---------------------------------------------------------------------------
# Reward Computation Utilities
# ---------------------------------------------------------------------------

def compute_reward(
    ensemble_mae: float,
    naive_mae: float,
    best_single_mae: float,
    elapsed_seconds: float = 0.0,
    max_time_budget: float = 3600.0,
    champion_mae: Optional[float] = None,
) -> float:
    """
    Compute normalized reward for the bandit.

    Reward design:
        - Primary: relative improvement over naive baseline
        - Bonus: improvement over best single model
        - Champion bonus (V7.3): extra signal for beating condition champion
        - Penalty: time cost (mild)

    Range: approximately [-1.0, 1.0]

    Args:
        champion_mae: MAE of the known champion for this condition (from truth pack).
                      When provided, shifts reward weight toward champion-beating.
    """
    if naive_mae <= 0 or not np.isfinite(naive_mae):
        return 0.0
    if not np.isfinite(ensemble_mae):
        return -1.0

    # Relative skill vs naive (higher = better)
    skill = 1.0 - (ensemble_mae / naive_mae)

    # Bonus for beating best single model
    if best_single_mae > 0 and np.isfinite(best_single_mae):
        beat_single = max(0.0, 1.0 - (ensemble_mae / best_single_mae))
    else:
        beat_single = 0.0

    # V7.3: Champion-beating bonus (strongest signal when champion_mae is known)
    beat_champion = 0.0
    if champion_mae is not None and champion_mae > 0 and np.isfinite(champion_mae):
        beat_champion = max(0.0, 1.0 - (ensemble_mae / champion_mae))

    # Time penalty (mild: 3% penalty at max budget)
    time_penalty = 0.03 * min(elapsed_seconds / max(max_time_budget, 1.0), 1.0)

    # Weight allocation: if champion known, prioritize beating it
    if champion_mae is not None and champion_mae > 0:
        reward = (
            0.40 * beat_champion
            + 0.25 * skill
            + 0.20 * beat_single
            - time_penalty
        )
    else:
        reward = 0.60 * skill + 0.35 * beat_single - time_penalty

    return np.clip(reward, -1.0, 1.0)


def build_replay_record(
    routing_info: Dict[str, Any],
    ensemble_mae: float,
    naive_mae: float,
    best_single_mae: float,
    elapsed: float,
) -> Dict[str, Any]:
    """
    Build a replay record from V73 routing_info for future warm-starts.
    """
    record = {
        "lane": routing_info.get("lane_selected", "general"),
        "horizon_band": routing_info.get("horizon_band", "mid"),
        "ablation": routing_info.get("ablation", "core_only"),
        "missingness_bucket": routing_info.get("missingness_bucket", "medium"),
        "meta_features": routing_info.get("meta_features", {}),
        "top_k": routing_info.get("top_k", 12),
        "moe_max_experts": routing_info.get("sparse_moe_max_experts", 5),
        "moe_temperature": routing_info.get("sparse_moe_temperature", 0.35),
        "blend_alpha": routing_info.get("sparse_moe_blend_alpha", 0.25),
        "qs_threshold_mult": routing_info.get("qs_threshold_mult", 0.90),
        "reward": compute_reward(ensemble_mae, naive_mae, best_single_mae, elapsed),
        "ensemble_mae": float(ensemble_mae),
        "naive_mae": float(naive_mae),
        "best_single_mae": float(best_single_mae),
        "elapsed_seconds": float(elapsed),
        "condition_key": routing_info.get("route_key", ""),
    }
    return record
