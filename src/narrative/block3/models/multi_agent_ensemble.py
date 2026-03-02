"""
Multi-Agent Ensemble Coordination Protocol for AutoFit V7.3.

Implements a blackboard-based multi-agent system where specialized agents
collaborate to build an optimal time series forecasting ensemble.

Architecture:
    ┌─────────────────────────────────────────────────┐
    │                  Orchestrator                     │
    │  (sequencing, timeout management, state machine) │
    └─────┬───────┬───────┬───────┬──────────────────┘
          │       │       │       │
          ▼       ▼       ▼       ▼
    ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
    │Recon │ │Scout │ │Compo │ │Critic│
    │Agent │ │Agent │ │Agent │ │Agent │
    └──────┘ └──────┘ └──────┘ └──────┘
          │       │       │       │
          └───────┴───────┴───────┘
                      │
                ┌─────▼─────┐
                │ Blackboard│
                │  (shared  │
                │   state)  │
                └───────────┘

Agents:
    1. ReconAgent: Analyzes target distribution, features, data quality
    2. ScoutAgent: Screens candidates, allocates compute budget
    3. ComposerAgent: Builds ensemble via forward selection + MoE
    4. CriticAgent: Validates quality, triggers guards, proposes restarts

Communication: via Blackboard (shared dict with versioned writes).

References:
    - Wooldridge (2009) "An Introduction to MultiAgent Systems"
    - Stone & Veloso (2000) "Multiagent Systems: A Survey from an ML Perspective"
    - Dorri et al. (2018) "Multi-Agent Systems: A Survey" (IEEE Access)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blackboard: Shared State Container
# ---------------------------------------------------------------------------

class Blackboard:
    """
    Shared state container for multi-agent coordination.

    Features:
        - Namespaced key-value store (agent writes to own namespace)
        - Version tracking per key
        - Read access across all namespaces
        - Append-only audit log
    """

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}
        self._versions: Dict[str, int] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._created_at = time.time()

    def write(self, key: str, value: Any, agent: str = "system") -> None:
        """Write a key-value pair with agent attribution."""
        ver = self._versions.get(key, 0) + 1
        self._store[key] = value
        self._versions[key] = ver
        self._audit_log.append({
            "t": time.time() - self._created_at,
            "agent": agent,
            "key": key,
            "version": ver,
        })

    def read(self, key: str, default: Any = None) -> Any:
        """Read a value from the blackboard."""
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def version(self, key: str) -> int:
        return self._versions.get(key, 0)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of the entire state."""
        return dict(self._store)


# ---------------------------------------------------------------------------
# Agent Base Class
# ---------------------------------------------------------------------------

class AgentBase(ABC):
    """Base class for all coordination agents."""

    name: str = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}

    @abstractmethod
    def execute(self, board: Blackboard) -> str:
        """
        Execute agent logic, reading from and writing to the blackboard.

        Returns a status string: "ok", "warn", "fail", "restart".
        """
        ...

    def _log(self, msg: str) -> None:
        logger.info(f"[{self.name}] {msg}")

    def _warn(self, msg: str) -> None:
        logger.warning(f"[{self.name}] {msg}")


# ---------------------------------------------------------------------------
# Agent 1: ReconAgent — Target & Data Analysis
# ---------------------------------------------------------------------------

class ReconAgent(AgentBase):
    """
    Analyzes target distribution, feature quality, and data characteristics.

    Writes to blackboard:
        - recon/lane: str
        - recon/meta_features: dict
        - recon/data_quality_report: dict
        - recon/recommended_objective: str
        - recon/risk_flags: list
        - recon/n_samples: int
        - recon/n_features: int
    """

    name = "ReconAgent"

    def execute(self, board: Blackboard) -> str:
        y = board.read("input/y_train")
        X = board.read("input/X_train")

        if y is None or X is None:
            self._warn("No training data on blackboard")
            return "fail"

        y_arr = np.asarray(y, dtype=float).ravel()
        n_samples = len(y_arr)
        n_features = X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else 0

        # Meta-features
        finite_y = y_arr[np.isfinite(y_arr)]
        if len(finite_y) == 0:
            self._warn("All target values are non-finite")
            board.write("recon/lane", "general", self.name)
            return "warn"

        meta = self._compute_meta_features(finite_y, X)
        lane = self._infer_lane(meta, finite_y)
        objective = self._lane_to_objective(lane)
        risk_flags = self._identify_risks(meta, n_samples, n_features)

        board.write("recon/lane", lane, self.name)
        board.write("recon/meta_features", meta, self.name)
        board.write("recon/recommended_objective", objective, self.name)
        board.write("recon/risk_flags", risk_flags, self.name)
        board.write("recon/n_samples", n_samples, self.name)
        board.write("recon/n_features", n_features, self.name)

        quality_report = self._assess_data_quality(X, y_arr)
        board.write("recon/data_quality_report", quality_report, self.name)

        self._log(
            f"lane={lane}, n={n_samples}, p={n_features}, "
            f"objective={objective}, risks={len(risk_flags)}"
        )
        return "ok"

    def _compute_meta_features(self, y: np.ndarray, X: Any) -> Dict[str, float]:
        """Extract lightweight meta-features from target and features."""
        from scipy import stats as sp_stats

        meta: Dict[str, float] = {}

        # Target distribution
        meta["mean"] = float(np.mean(y))
        meta["std"] = float(np.std(y))
        cv = meta["std"] / max(abs(meta["mean"]), 1e-10)
        meta["cv"] = float(cv)
        meta["kurtosis"] = float(sp_stats.kurtosis(y, fisher=True))
        meta["skewness"] = float(sp_stats.skew(y))
        meta["zero_frac"] = float(np.mean(np.abs(y) < 1e-10))
        n_unique = len(np.unique(y))
        meta["n_unique_ratio"] = float(n_unique / max(len(y), 1))

        # IQR ratio
        q25, q75 = np.percentile(y, [25, 75])
        iqr = q75 - q25
        meta["iqr_ratio"] = float(iqr / max(meta["std"], 1e-10))

        # Feature-target correlation (max absolute)
        meta["exog_corr_max"] = 0.0
        if hasattr(X, "values"):
            X_arr = X.values
        elif hasattr(X, "__array__"):
            X_arr = np.asarray(X)
        else:
            X_arr = None

        if X_arr is not None and X_arr.ndim == 2 and X_arr.shape[1] > 0:
            try:
                # Compute correlation with target for numeric columns
                valid_mask = np.isfinite(X_arr).all(axis=0) & np.isfinite(y).all()
                if valid_mask.any() if hasattr(valid_mask, "any") else valid_mask:
                    correlations = []
                    for j in range(X_arr.shape[1]):
                        col = X_arr[:, j].astype(float)
                        valid = np.isfinite(col) & np.isfinite(y)
                        if valid.sum() > 10 and np.std(col[valid]) > 1e-10:
                            r = np.corrcoef(col[valid], y[valid])[0, 1]
                            if np.isfinite(r):
                                correlations.append(abs(r))
                    if correlations:
                        meta["exog_corr_max"] = float(max(correlations))
                        meta["exog_corr_mean"] = float(np.mean(correlations))
            except Exception:
                pass

        # Missingness rate
        if X_arr is not None:
            meta["missing_rate"] = float(np.mean(~np.isfinite(X_arr)))
        else:
            meta["missing_rate"] = 0.0

        return meta

    def _infer_lane(self, meta: Dict[str, float], y: np.ndarray) -> str:
        """Determine target lane from meta-features."""
        unique_vals = np.unique(y[np.isfinite(y)])
        n_unique = len(unique_vals)

        # Binary check
        if n_unique <= 3 and set(unique_vals).issubset({0.0, 1.0}):
            return "binary"

        # Count check
        is_non_neg = np.all(y[np.isfinite(y)] >= 0)
        int_like_frac = np.mean(np.abs(y - np.round(y)) < 1e-6) if len(y) > 0 else 0
        if is_non_neg and int_like_frac > 0.90 and n_unique > 3 and np.max(y) > 2:
            return "count"

        # Heavy-tail check
        if is_non_neg and meta.get("kurtosis", 0) > 5:
            return "heavy_tail"

        return "general"

    def _lane_to_objective(self, lane: str) -> str:
        return {
            "binary": "binary",
            "count": "count",
            "heavy_tail": "huber",
            "general": "l2",
        }.get(lane, "l2")

    def _identify_risks(
        self, meta: Dict[str, float], n_samples: int, n_features: int
    ) -> List[str]:
        """Flag potential issues."""
        risks = []
        if n_samples < 100:
            risks.append("low_sample_size")
        if n_features > n_samples:
            risks.append("high_dimensional")
        if meta.get("missing_rate", 0) > 0.3:
            risks.append("high_missingness")
        if meta.get("kurtosis", 0) > 20:
            risks.append("extreme_kurtosis")
        if meta.get("zero_frac", 0) > 0.5:
            risks.append("zero_inflated")
        if meta.get("cv", 0) > 5:
            risks.append("high_cv")
        return risks

    def _assess_data_quality(self, X: Any, y: np.ndarray) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        report: Dict[str, Any] = {"grade": "A"}

        # Target quality
        n_nan = int(np.sum(~np.isfinite(y)))
        report["target_nan_count"] = n_nan
        report["target_nan_rate"] = n_nan / max(len(y), 1)

        # Feature quality
        if hasattr(X, "isnull"):
            miss_rate = float(X.isnull().mean().mean())
        elif hasattr(X, "__array__"):
            X_arr = np.asarray(X)
            miss_rate = float(np.mean(~np.isfinite(X_arr)))
        else:
            miss_rate = 0.0
        report["feature_missing_rate"] = miss_rate

        # Grade
        if miss_rate > 0.3 or report["target_nan_rate"] > 0.05:
            report["grade"] = "C"
        elif miss_rate > 0.1 or report["target_nan_rate"] > 0.01:
            report["grade"] = "B"

        return report


# ---------------------------------------------------------------------------
# Agent 2: ScoutAgent — Candidate Screening & Budget Allocation
# ---------------------------------------------------------------------------

class ScoutAgent(AgentBase):
    """
    Screens model candidates and allocates compute budget.

    Reads from blackboard:
        - recon/lane, recon/risk_flags, recon/n_samples
        - config/candidate_pool, config/top_k, config/qs_threshold_mult
        - config/time_budget

    Writes to blackboard:
        - scout/prioritized_candidates: list[str]  (ordered by expected value)
        - scout/time_allocation: dict[str, float]  (seconds per candidate)
        - scout/gpu_required: bool
        - scout/skip_list: list[str]
        - scout/budget_plan: dict
    """

    name = "ScoutAgent"

    # Expected evaluation time per category (seconds, with GPU)
    _CATEGORY_TIME_EST = {
        "ml_tabular": 30,
        "statistical": 60,
        "deep_classical": 180,
        "transformer_sota": 300,
        "foundation": 240,
        "irregular": 120,
        "autofit": 0,  # never self-refer
    }

    # Category priority ranking (for budget allocation)
    _CATEGORY_PRIORITY = {
        "deep_classical": 1.0,      # Highest: NBEATS, NHITS dominate
        "transformer_sota": 0.95,
        "foundation": 0.90,
        "ml_tabular": 0.70,
        "statistical": 0.50,
        "irregular": 0.40,
    }

    def execute(self, board: Blackboard) -> str:
        lane = board.read("recon/lane", "general")
        risks = board.read("recon/risk_flags", [])
        n_samples = board.read("recon/n_samples", 1000)
        candidates = board.read("config/candidate_pool", [])
        top_k = board.read("config/top_k", 12)
        time_budget = board.read("config/time_budget", 3600)
        has_gpu = board.read("config/has_gpu", False)

        if not candidates:
            self._warn("No candidates on blackboard")
            return "fail"

        # Get category for each candidate
        cat_fn = board.read("config/category_fn")

        # Priority scoring
        scored = []
        for name in candidates:
            cat = cat_fn(name) if cat_fn else "ml_tabular"
            priority = self._CATEGORY_PRIORITY.get(cat, 0.5)

            # Adjust by lane relevance
            priority *= self._lane_relevance(name, lane, cat)

            # Penalize GPU models if no GPU
            if cat in {"deep_classical", "transformer_sota", "foundation"} and not has_gpu:
                priority *= 0.01  # Near-zero but keep for logging

            # Penalize for risk flags
            if "low_sample_size" in risks and cat in {"deep_classical", "transformer_sota"}:
                priority *= 0.5  # Deep models struggle with small data

            scored.append((name, priority, cat))

        # Sort by priority descending
        scored.sort(key=lambda x: -x[1])

        # Allocate time budget
        prioritized = [s[0] for s in scored[:top_k * 2]]  # 2x candidates for screening
        skip_list = [s[0] for s in scored if s[1] < 0.05]

        time_alloc = {}
        remaining_budget = float(time_budget)
        for name, priority, cat in scored[:top_k * 2]:
            if name in skip_list:
                time_alloc[name] = 0.0
                continue
            est_time = self._CATEGORY_TIME_EST.get(cat, 120) * (1.5 if n_samples > 5000 else 1.0)
            alloc = min(est_time, remaining_budget * 0.15)
            time_alloc[name] = alloc
            remaining_budget -= alloc
            if remaining_budget <= 0:
                break

        budget_plan = {
            "total_budget": time_budget,
            "allocated": sum(time_alloc.values()),
            "n_candidates": len(prioritized),
            "n_skipped": len(skip_list),
            "gpu_required": has_gpu and any(
                s[2] in {"deep_classical", "transformer_sota", "foundation"}
                for s in scored[:top_k]
            ),
        }

        board.write("scout/prioritized_candidates", prioritized, self.name)
        board.write("scout/time_allocation", time_alloc, self.name)
        board.write("scout/skip_list", skip_list, self.name)
        board.write("scout/budget_plan", budget_plan, self.name)
        board.write("scout/gpu_required", budget_plan["gpu_required"], self.name)

        self._log(
            f"Prioritized {len(prioritized)} candidates, "
            f"skipping {len(skip_list)}, "
            f"budget={budget_plan['allocated']:.0f}/{time_budget:.0f}s"
        )
        return "ok"

    def _lane_relevance(self, model_name: str, lane: str, category: str) -> float:
        """Estimate model relevance for this target lane."""
        # Count-lane specialists
        if lane == "count" and model_name in {
            "XGBoostPoisson", "LightGBMTweedie", "NegativeBinomialGLM",
        }:
            return 1.5

        # Heavy-tail: tree models and deep models are robust
        if lane == "heavy_tail" and category in {"ml_tabular", "deep_classical"}:
            return 1.2

        # Binary: classifiers get a boost
        if lane == "binary" and "Classifier" in model_name:
            return 1.5

        return 1.0


# ---------------------------------------------------------------------------
# Agent 3: ComposerAgent — Ensemble Construction
# ---------------------------------------------------------------------------

class ComposerAgent(AgentBase):
    """
    Builds the final ensemble from evaluated candidates.

    Reads from blackboard:
        - eval/results: dict[str, dict]  (model_name -> {mae, oof_preds, adj_mae, ...})
        - recon/lane
        - config/moe_max_experts, config/moe_temperature, config/blend_alpha

    Writes to blackboard:
        - composer/selected_models: list[str]
        - composer/weights: dict[str, float]
        - composer/moe_route: dict
        - composer/ensemble_diversity: float
        - composer/composition_strategy: str
    """

    name = "ComposerAgent"

    def execute(self, board: Blackboard) -> str:
        results = board.read("eval/results", {})
        lane = board.read("recon/lane", "general")
        moe_max = board.read("config/moe_max_experts", 5)
        moe_temp = board.read("config/moe_temperature", 0.40)
        blend_alpha = board.read("config/blend_alpha", 0.25)
        y_train = board.read("input/y_train")

        if not results:
            self._warn("No evaluation results available")
            return "fail"

        # Sort by adjusted MAE
        ranked = sorted(results.items(), key=lambda x: x[1].get("adj_mae", float("inf")))
        ranked_names = [name for name, _ in ranked]

        self._log(f"Composing from {len(ranked)} evaluated models")

        # Step 1: Forward selection
        selected, selection_log = self._forward_select(ranked, y_train, lane)

        # Step 2: Diversity-aware MoE pruning
        active_experts, moe_weights = self._sparse_moe_select(
            selected, results, moe_max, moe_temp, lane
        )

        # Step 3: Compute final weights
        final_weights = self._compute_blended_weights(
            active_experts, results, moe_weights, blend_alpha, lane
        )

        # Step 4: Compute diversity metric
        diversity = self._compute_ensemble_diversity(active_experts, results)

        # Determine strategy label
        if len(active_experts) == 1:
            strategy = "single_best"
        elif len(active_experts) <= 3:
            strategy = "compact_ensemble"
        else:
            strategy = "diverse_moe_ensemble"

        board.write("composer/selected_models", active_experts, self.name)
        board.write("composer/weights", final_weights, self.name)
        board.write("composer/moe_route", {
            "active_experts": active_experts,
            "moe_weights": moe_weights,
            "selection_log": selection_log,
        }, self.name)
        board.write("composer/ensemble_diversity", diversity, self.name)
        board.write("composer/composition_strategy", strategy, self.name)

        self._log(
            f"Selected {len(active_experts)} models: {active_experts}, "
            f"diversity={diversity:.3f}, strategy={strategy}"
        )
        return "ok"

    def _forward_select(
        self,
        ranked: List[Tuple[str, Dict]],
        y_train: Any,
        lane: str,
    ) -> Tuple[List[str], List[str]]:
        """Monotone forward selection: add model only if ensemble MAE decreases."""
        if not ranked:
            return [], ["empty_ranked"]

        selected = [ranked[0][0]]
        best_mae = ranked[0][1].get("adj_mae", float("inf"))
        log = [f"init: {selected[0]} mae={best_mae:.6f}"]

        max_select = min(len(ranked), 16)

        for name, info in ranked[1:max_select]:
            oof_preds = info.get("oof_preds")
            if oof_preds is None:
                continue

            # Simulate adding this model (equal-weight average of OOF)
            existing_preds = [ranked_info.get("oof_preds") for n, ranked_info in ranked
                              if n in selected and ranked_info.get("oof_preds") is not None]
            if not existing_preds:
                selected.append(name)
                log.append(f"add (no existing preds): {name}")
                continue

            try:
                all_preds = existing_preds + [oof_preds]
                # Stack and average
                stacked = np.column_stack([np.asarray(p).ravel()[:len(np.asarray(y_train).ravel())]
                                           for p in all_preds
                                           if len(np.asarray(p).ravel()) >= len(np.asarray(y_train).ravel())])
                if stacked.ndim < 2 or stacked.shape[1] < 2:
                    selected.append(name)
                    log.append(f"add (stack fail): {name}")
                    continue

                ensemble_pred = np.mean(stacked, axis=1)
                y_arr = np.asarray(y_train).ravel()[:len(ensemble_pred)]
                new_mae = float(np.mean(np.abs(y_arr - ensemble_pred)))

                # Lane-specific anti-collapse tolerance
                tol = {"count": 0.05, "heavy_tail": 0.01, "general": 0.0, "binary": 0.0}.get(lane, 0.0)

                if new_mae <= best_mae * (1.0 + tol):
                    selected.append(name)
                    if new_mae < best_mae:
                        best_mae = new_mae
                    log.append(f"add: {name} new_mae={new_mae:.6f}")
                else:
                    log.append(f"reject: {name} mae={new_mae:.6f} > limit={best_mae * (1+tol):.6f}")
            except Exception as e:
                log.append(f"skip ({name}): {e}")
                continue

        return selected, log

    def _sparse_moe_select(
        self,
        selected: List[str],
        results: Dict[str, Dict],
        max_experts: int,
        temperature: float,
        lane: str,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select top experts using composite scoring + softmax."""
        if len(selected) <= max_experts:
            # No pruning needed
            equal_w = 1.0 / max(len(selected), 1)
            return selected, {name: equal_w for name in selected}

        # Score each selected model
        scores: Dict[str, float] = {}
        for name in selected:
            info = results.get(name, {})
            # Quality score (normalized rank)
            all_names = list(results.keys())
            try:
                rank = sorted(all_names, key=lambda n: results[n].get("adj_mae", float("inf"))).index(name)
                rank_quality = 1.0 - (rank / max(len(all_names) - 1, 1))
            except ValueError:
                rank_quality = 0.5

            # Diversity contribution (crude: model category diversity)
            cat = info.get("category", "unknown")
            category_bonus = 0.1 if cat in {"deep_classical", "transformer_sota", "foundation"} else 0.0

            # Lane specialist bonus
            lane_bonus = 0.0
            if lane == "count" and "Poisson" in name or "Tweedie" in name:
                lane_bonus = 0.15
            elif lane == "heavy_tail" and cat in {"ml_tabular", "deep_classical"}:
                lane_bonus = 0.10

            score = 1.30 * rank_quality + 0.35 * category_bonus + lane_bonus
            scores[name] = score

        # Select top-k by score
        sorted_by_score = sorted(scores.items(), key=lambda x: -x[1])
        active = [name for name, _ in sorted_by_score[:max_experts]]

        # Softmax weights
        active_scores = np.array([scores[n] for n in active])
        exp_scores = np.exp(active_scores / max(temperature, 0.01))
        weights = exp_scores / exp_scores.sum()

        # Floor weights
        min_w = 0.05
        weights = np.maximum(weights, min_w)
        weights /= weights.sum()

        moe_weights = {name: float(w) for name, w in zip(active, weights)}
        return active, moe_weights

    def _compute_blended_weights(
        self,
        models: List[str],
        results: Dict[str, Dict],
        moe_weights: Dict[str, float],
        blend_alpha: float,
        lane: str,
    ) -> Dict[str, float]:
        """Blend MoE weights with inverse-error weights."""
        if not models:
            return {}

        # Inverse-MAE weights (conformal-style)
        inv_weights = {}
        for name in models:
            mae = results.get(name, {}).get("mae", 1.0)
            inv_weights[name] = 1.0 / max(mae, 1e-10)

        inv_total = sum(inv_weights.values())
        inv_weights = {n: v / max(inv_total, 1e-10) for n, v in inv_weights.items()}

        # Blend: (1-alpha)*inverse_error + alpha*moe
        final = {}
        for name in models:
            w_inv = inv_weights.get(name, 0.0)
            w_moe = moe_weights.get(name, 0.0)
            final[name] = (1.0 - blend_alpha) * w_inv + blend_alpha * w_moe

        # Normalize
        total = sum(final.values())
        if total > 0:
            final = {n: v / total for n, v in final.items()}

        return final

    def _compute_ensemble_diversity(
        self, models: List[str], results: Dict[str, Dict]
    ) -> float:
        """Compute pairwise prediction correlation as diversity metric (lower = more diverse)."""
        if len(models) < 2:
            return 0.0

        preds = []
        for name in models:
            oof = results.get(name, {}).get("oof_preds")
            if oof is not None:
                preds.append(np.asarray(oof).ravel())

        if len(preds) < 2:
            return 0.0

        # Truncate to same length
        min_len = min(len(p) for p in preds)
        preds = [p[:min_len] for p in preds]

        # Mean pairwise correlation
        correlations = []
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                try:
                    valid = np.isfinite(preds[i]) & np.isfinite(preds[j])
                    if valid.sum() > 10:
                        r = np.corrcoef(preds[i][valid], preds[j][valid])[0, 1]
                        if np.isfinite(r):
                            correlations.append(abs(r))
                except Exception:
                    continue

        if not correlations:
            return 0.0

        # Diversity = 1 - mean_correlation (higher = more diverse)
        return float(1.0 - np.mean(correlations))


# ---------------------------------------------------------------------------
# Agent 4: CriticAgent — Quality Validation & Guard Logic
# ---------------------------------------------------------------------------

class CriticAgent(AgentBase):
    """
    Validates ensemble quality and triggers guards if needed.

    Reads from blackboard:
        - composer/selected_models, composer/weights
        - eval/results
        - recon/lane, recon/risk_flags
        - input/y_train

    Writes to blackboard:
        - critic/verdict: str ("accept", "warn", "reject", "restart")
        - critic/guard_triggered: bool
        - critic/quality_report: dict
        - critic/improvement_suggestions: list[str]
    """

    name = "CriticAgent"

    def execute(self, board: Blackboard) -> str:
        models = board.read("composer/selected_models", [])
        weights = board.read("composer/weights", {})
        results = board.read("eval/results", {})
        lane = board.read("recon/lane", "general")
        risks = board.read("recon/risk_flags", [])
        y_train = board.read("input/y_train")

        if not models or y_train is None:
            self._warn("Insufficient data for quality validation")
            board.write("critic/verdict", "warn", self.name)
            return "warn"

        y_arr = np.asarray(y_train).ravel()
        report: Dict[str, Any] = {}
        suggestions: List[str] = []
        guard_triggered = False

        # Check 1: Ensemble OOF MAE vs best single
        best_single_mae = float("inf")
        best_single_name = ""
        for name, info in results.items():
            mae = info.get("mae", float("inf"))
            if mae < best_single_mae:
                best_single_mae = mae
                best_single_name = name

        ensemble_oof = self._compute_ensemble_oof(models, weights, results, y_arr)
        if ensemble_oof is not None:
            ensemble_mae = float(np.mean(np.abs(y_arr[:len(ensemble_oof)] - ensemble_oof)))
            report["ensemble_mae"] = ensemble_mae
            report["best_single_mae"] = best_single_mae
            report["best_single_name"] = best_single_name

            # OOF guard: ensemble must beat 1.03× best single
            if ensemble_mae > 1.03 * best_single_mae:
                guard_triggered = True
                suggestions.append(
                    f"Ensemble MAE ({ensemble_mae:.4f}) > 1.03 × best single "
                    f"({best_single_name}: {best_single_mae:.4f}). "
                    f"Consider fallback to single best."
                )

            # Improvement ratio
            report["improvement_ratio"] = 1.0 - (ensemble_mae / max(best_single_mae, 1e-10))
        else:
            report["ensemble_mae"] = None
            report["improvement_ratio"] = None

        # Check 2: Diversity check
        diversity = board.read("composer/ensemble_diversity", 0.0)
        report["diversity"] = diversity
        if diversity < 0.05 and len(models) > 1:
            suggestions.append(
                f"Low ensemble diversity ({diversity:.3f}). "
                f"Models may be redundant."
            )

        # Check 3: Weight concentration
        if weights:
            max_weight = max(weights.values())
            report["max_weight"] = max_weight
            if max_weight > 0.80:
                suggestions.append(
                    f"Single model dominates with weight={max_weight:.2f}. "
                    f"Ensemble may be effectively single-model."
                )

        # Check 4: Spike sentinel (count lane)
        if lane == "count" and ensemble_oof is not None:
            safe_scale = max(np.median(np.abs(y_arr[y_arr > 0])), 1.0) if np.any(y_arr > 0) else 1.0
            if np.max(ensemble_oof) > 100 * safe_scale:
                guard_triggered = True
                suggestions.append(
                    f"Spike sentinel: max prediction ({np.max(ensemble_oof):.0f}) "
                    f"> 100× safe scale ({safe_scale:.0f})."
                )

        # Check 5: Tail quality (heavy-tail lane)
        if lane == "heavy_tail" and ensemble_oof is not None:
            q90_idx = y_arr > np.percentile(y_arr, 90)
            if q90_idx.sum() > 5:
                tail_mae = float(np.mean(np.abs(
                    y_arr[q90_idx][:len(ensemble_oof)] -
                    ensemble_oof[:len(y_arr)][q90_idx[:len(ensemble_oof)]]
                )))
                report["tail_q90_mae"] = tail_mae

        # Verdict
        if guard_triggered:
            verdict = "reject"
        elif suggestions:
            verdict = "warn"
        else:
            verdict = "accept"

        board.write("critic/verdict", verdict, self.name)
        board.write("critic/guard_triggered", guard_triggered, self.name)
        board.write("critic/quality_report", report, self.name)
        board.write("critic/improvement_suggestions", suggestions, self.name)

        self._log(f"Verdict: {verdict}, guard={guard_triggered}, suggestions={len(suggestions)}")
        return "ok"

    def _compute_ensemble_oof(
        self,
        models: List[str],
        weights: Dict[str, float],
        results: Dict[str, Dict],
        y_arr: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute weighted ensemble OOF predictions."""
        preds = []
        ws = []
        for name in models:
            oof = results.get(name, {}).get("oof_preds")
            w = weights.get(name, 0.0)
            if oof is not None and w > 0:
                preds.append(np.asarray(oof).ravel())
                ws.append(w)

        if not preds:
            return None

        min_len = min(len(p) for p in preds)
        min_len = min(min_len, len(y_arr))
        preds = [p[:min_len] for p in preds]
        ws_arr = np.array(ws)
        ws_arr /= ws_arr.sum()

        ensemble = np.zeros(min_len)
        for p, w in zip(preds, ws_arr):
            ensemble += w * p

        return ensemble


# ---------------------------------------------------------------------------
# Agent 5: ChampionTransferAgent — Domain Knowledge Transfer (V7.3)
# ---------------------------------------------------------------------------

class ChampionTransferAgent(AgentBase):
    """
    Encodes domain knowledge about which models win specific conditions,
    derived from the truth pack condition_leaderboard analysis.

    Reads from blackboard:
        - recon/lane, recon/meta_features
        - config/horizon_band, config/ablation
        - config/candidate_pool

    Writes to blackboard:
        - champion/force_include: list[str]  (models that MUST be evaluated)
        - champion/expected_winner: str  (predicted best model for this condition)
        - champion/fallback_strategy: str  (what to do if champion fails)
        - champion/gate_tolerance: float  (how strict the champion-first gate should be)
        - champion/knowledge_confidence: float  (0-1, how confident the transfer is)

    Knowledge base from condition_leaderboard.csv 104-key analysis:
        - NBEATS: 39 wins (count:all, heavy_tail:short)
        - PatchTST: 24 wins (binary:mid/short, heavy_tail:mid/long)
        - NHITS: 23 wins (binary:long, count:short, heavy_tail:short)
        - KAN: 7 wins (count:short only)
        - Chronos: 6 wins (heavy_tail:long/full ablation)
        - NBEATSx: 4 wins (heavy_tail:short/full, binary:long)
        - DLinear: 1 win (binary:short/full)
    """

    name = "ChampionTransferAgent"

    # Champion knowledge base: (lane, horizon_band) → {models, confidence}
    # Derived from truth pack condition_leaderboard.csv analysis
    _CHAMPION_KNOWLEDGE: Dict[Tuple[str, str], Dict[str, Any]] = {
        # Count lane: NBEATS dominates all horizons; KAN/NHITS competitive at short
        ("count", "short"): {
            "force_models": ["NBEATS", "NHITS", "KAN"],
            "expected_winner": "NBEATS",
            "gate_tol": 0.00,
            "confidence": 0.90,
        },
        ("count", "mid"): {
            "force_models": ["NBEATS", "NHITS"],
            "expected_winner": "NBEATS",
            "gate_tol": 0.00,
            "confidence": 0.92,
        },
        ("count", "long"): {
            "force_models": ["NBEATS", "NHITS"],
            "expected_winner": "NBEATS",
            "gate_tol": 0.00,
            "confidence": 0.95,
        },
        # Binary lane: PatchTST at mid/short, NHITS at long
        ("binary", "short"): {
            "force_models": ["PatchTST", "NHITS", "DLinear"],
            "expected_winner": "PatchTST",
            "gate_tol": 0.00,
            "confidence": 0.85,
        },
        ("binary", "mid"): {
            "force_models": ["PatchTST", "NHITS"],
            "expected_winner": "PatchTST",
            "gate_tol": 0.00,
            "confidence": 0.90,
        },
        ("binary", "long"): {
            "force_models": ["NHITS", "NBEATSx", "PatchTST"],
            "expected_winner": "NHITS",
            "gate_tol": 0.00,
            "confidence": 0.88,
        },
        # Heavy-tail lane: short=NBEATS/NHITS, mid=PatchTST, long=PatchTST/Chronos
        ("heavy_tail", "short"): {
            "force_models": ["NBEATS", "NHITS", "NBEATSx"],
            "expected_winner": "NBEATS",
            "gate_tol": 0.00,
            "confidence": 0.88,
        },
        ("heavy_tail", "mid"): {
            "force_models": ["PatchTST", "NHITS"],
            "expected_winner": "PatchTST",
            "gate_tol": 0.00,
            "confidence": 0.90,
        },
        ("heavy_tail", "long"): {
            "force_models": ["PatchTST", "Chronos", "NHITS"],
            "expected_winner": "PatchTST",
            "gate_tol": 0.00,
            "confidence": 0.85,
        },
    }

    def execute(self, board: Blackboard) -> str:
        lane = board.read("recon/lane", "general")
        horizon_band = board.read("config/horizon_band", "mid")
        candidates = board.read("config/candidate_pool", [])

        # Look up knowledge for this (lane, horizon_band)
        knowledge = self._CHAMPION_KNOWLEDGE.get((lane, horizon_band))

        if knowledge is None:
            # No specific knowledge — use general GPU-model priority
            board.write("champion/force_include", ["NBEATS", "PatchTST", "NHITS"], self.name)
            board.write("champion/expected_winner", "NBEATS", self.name)
            board.write("champion/fallback_strategy", "best_single_from_oof", self.name)
            board.write("champion/gate_tolerance", 0.00, self.name)
            board.write("champion/knowledge_confidence", 0.50, self.name)
            self._log(f"No specific knowledge for ({lane}, {horizon_band}), using defaults")
            return "ok"

        force_models = knowledge["force_models"]
        expected_winner = knowledge["expected_winner"]
        gate_tol = knowledge["gate_tol"]
        confidence = knowledge["confidence"]

        # Verify force models are in candidate pool
        available_forces = [m for m in force_models if m in candidates]
        if not available_forces:
            self._warn(
                f"None of force_models {force_models} in candidate pool. "
                f"Pool has {len(candidates)} candidates."
            )
            board.write("champion/force_include", force_models, self.name)
            board.write("champion/expected_winner", expected_winner, self.name)
            board.write("champion/fallback_strategy", "gpu_required", self.name)
            board.write("champion/gate_tolerance", gate_tol, self.name)
            board.write("champion/knowledge_confidence", confidence * 0.5, self.name)
            return "warn"

        board.write("champion/force_include", force_models, self.name)
        board.write("champion/expected_winner", expected_winner, self.name)
        board.write("champion/fallback_strategy", "best_single_from_oof", self.name)
        board.write("champion/gate_tolerance", gate_tol, self.name)
        board.write("champion/knowledge_confidence", confidence, self.name)

        self._log(
            f"({lane}, {horizon_band}): force={force_models}, "
            f"expected={expected_winner}, confidence={confidence:.2f}"
        )
        return "ok"


# ---------------------------------------------------------------------------
# Orchestrator: Multi-Agent Coordination
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    """Configuration for the multi-agent orchestrator."""
    max_restart_attempts: int = 3  # V7.3: increased from 2 for iterative protocol
    timeout_seconds: float = 3600.0
    enable_critic_restart: bool = True
    enable_champion_transfer: bool = True  # V7.3: activate ChampionTransferAgent
    verbose: bool = True


class MultiAgentOrchestrator:
    """
    Coordinates the multi-agent ensemble building pipeline.

    Pipeline:
        1. ReconAgent → analyzes data
        1b. ChampionTransferAgent → encodes champion domain knowledge (V7.3)
        2. ScoutAgent → screens candidates, allocates budget
        3. [External: V73 fit evaluates candidates]
        4. ComposerAgent → builds ensemble from results
        5. CriticAgent → validates quality

    Iterative Protocol (V7.3):
        Round 1: Standard composition
        Round 2 (if rejected): Relaxed params (+1 expert, +0.10 blend_alpha)
        Round 3 (if rejected): Champion-first fallback (use best single model)

    If CriticAgent rejects after all rounds, orchestrator forces best single model.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        recon: Optional[ReconAgent] = None,
        scout: Optional[ScoutAgent] = None,
        composer: Optional[ComposerAgent] = None,
        critic: Optional[CriticAgent] = None,
        champion_transfer: Optional[ChampionTransferAgent] = None,
    ):
        self._config = config or OrchestratorConfig()
        self._agents = {
            "recon": recon or ReconAgent(),
            "scout": scout or ScoutAgent(),
            "composer": composer or ComposerAgent(),
            "critic": critic or CriticAgent(),
            "champion_transfer": champion_transfer or ChampionTransferAgent(),
        }
        self._board = Blackboard()
        self._n_restarts = 0
        self._timing: Dict[str, float] = {}

    @property
    def blackboard(self) -> Blackboard:
        return self._board

    def run_recon_phase(
        self,
        X_train: Any,
        y_train: Any,
        candidate_pool: List[str],
        category_fn: Optional[Callable] = None,
        has_gpu: bool = False,
        top_k: int = 12,
        moe_max_experts: int = 5,
        moe_temperature: float = 0.40,
        blend_alpha: float = 0.25,
        time_budget: float = 3600.0,
        qs_threshold_mult: float = 0.90,
        horizon_band: str = "mid",
        ablation: str = "core_only",
    ) -> Dict[str, Any]:
        """
        Run Phase 1: Reconnaissance + Champion Transfer + Scouting.

        Call this BEFORE candidate evaluation.
        Returns scout's prioritized candidate list and budget plan,
        enriched with champion domain knowledge.
        """
        # Load inputs
        self._board.write("input/X_train", X_train, "orchestrator")
        self._board.write("input/y_train", y_train, "orchestrator")
        self._board.write("config/candidate_pool", candidate_pool, "orchestrator")
        self._board.write("config/category_fn", category_fn, "orchestrator")
        self._board.write("config/has_gpu", has_gpu, "orchestrator")
        self._board.write("config/top_k", top_k, "orchestrator")
        self._board.write("config/moe_max_experts", moe_max_experts, "orchestrator")
        self._board.write("config/moe_temperature", moe_temperature, "orchestrator")
        self._board.write("config/blend_alpha", blend_alpha, "orchestrator")
        self._board.write("config/time_budget", time_budget, "orchestrator")
        self._board.write("config/qs_threshold_mult", qs_threshold_mult, "orchestrator")
        self._board.write("config/horizon_band", horizon_band, "orchestrator")
        self._board.write("config/ablation", ablation, "orchestrator")

        # Phase 1a: Recon
        t0 = time.time()
        status = self._agents["recon"].execute(self._board)
        self._timing["recon"] = time.time() - t0

        if status == "fail":
            logger.warning("[Orchestrator] Recon failed, using defaults")

        # Phase 1b: Champion Transfer (V7.3)
        champion_info: Dict[str, Any] = {}
        if self._config.enable_champion_transfer:
            t0 = time.time()
            ct_status = self._agents["champion_transfer"].execute(self._board)
            self._timing["champion_transfer"] = time.time() - t0

            champion_info = {
                "force_include": self._board.read("champion/force_include", []),
                "expected_winner": self._board.read("champion/expected_winner", ""),
                "fallback_strategy": self._board.read("champion/fallback_strategy", ""),
                "gate_tolerance": self._board.read("champion/gate_tolerance", 0.00),
                "knowledge_confidence": self._board.read("champion/knowledge_confidence", 0.0),
            }

            # Inject force_include models into candidate pool if not present
            force_models = champion_info.get("force_include", [])
            current_pool = self._board.read("config/candidate_pool", [])
            for model_name in force_models:
                if model_name not in current_pool:
                    current_pool.append(model_name)
            self._board.write("config/candidate_pool", current_pool, "orchestrator")

        # Phase 1c: Scout
        t0 = time.time()
        status = self._agents["scout"].execute(self._board)
        self._timing["scout"] = time.time() - t0

        # Ensure force_include models are at the top of prioritized list
        if champion_info.get("force_include"):
            prioritized = self._board.read("scout/prioritized_candidates", candidate_pool)
            force_models = champion_info["force_include"]
            # Move force_include models to top while preserving order
            reordered = [m for m in force_models if m in prioritized]
            reordered += [m for m in prioritized if m not in force_models]
            self._board.write("scout/prioritized_candidates", reordered, "orchestrator")

        return {
            "lane": self._board.read("recon/lane", "general"),
            "meta_features": self._board.read("recon/meta_features", {}),
            "risks": self._board.read("recon/risk_flags", []),
            "prioritized_candidates": self._board.read("scout/prioritized_candidates", candidate_pool),
            "budget_plan": self._board.read("scout/budget_plan", {}),
            "time_allocation": self._board.read("scout/time_allocation", {}),
            "champion_transfer": champion_info,
        }

    def run_compose_phase(
        self,
        eval_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Run Phase 2: Composition + Critique with iterative protocol.

        Call this AFTER candidate evaluation with results dict.
        Each result should have: {mae, adj_mae, oof_preds, category, ...}

        V7.3 Iterative Protocol:
            Round 1: Standard composition (default params)
            Round 2: Relaxed params (+1 expert, +0.10 blend_alpha)
            Round 3: Champion-first fallback (force best single model)

        Returns composition decision and quality verdict.
        """
        self._board.write("eval/results", eval_results, "orchestrator")

        # Phase 2a: Compose
        t0 = time.time()
        comp_status = self._agents["composer"].execute(self._board)
        self._timing["composer"] = time.time() - t0

        if comp_status == "fail":
            logger.warning("[Orchestrator] Composer failed")
            return {"verdict": "fail", "selected_models": [], "weights": {}}

        # Phase 2b: Critique
        t0 = time.time()
        critic_status = self._agents["critic"].execute(self._board)
        self._timing["critic"] = time.time() - t0

        verdict = self._board.read("critic/verdict", "warn")
        guard_triggered = self._board.read("critic/guard_triggered", False)

        # V7.3 Iterative restart protocol
        while (
            verdict == "reject"
            and self._config.enable_critic_restart
            and self._n_restarts < self._config.max_restart_attempts
        ):
            self._n_restarts += 1
            logger.info(
                f"[Orchestrator] Critic rejected, restart #{self._n_restarts}/{self._config.max_restart_attempts}"
            )

            if self._n_restarts <= 2:
                # Rounds 1-2: progressively relax composition parameters
                current_moe = self._board.read("config/moe_max_experts", 5)
                self._board.write(
                    "config/moe_max_experts",
                    min(current_moe + 1, 8),
                    "orchestrator",
                )
                self._board.write(
                    "config/blend_alpha",
                    min(
                        self._board.read("config/blend_alpha", 0.25) + 0.10,
                        0.50,
                    ),
                    "orchestrator",
                )
                logger.info(
                    f"[Orchestrator] Round {self._n_restarts}: relaxed params "
                    f"(moe={self._board.read('config/moe_max_experts')}, "
                    f"blend_alpha={self._board.read('config/blend_alpha'):.2f})"
                )
            else:
                # Round 3: Champion-first fallback — force best single model
                best_name = None
                best_mae = float("inf")
                for name, info in eval_results.items():
                    mae = info.get("mae", float("inf"))
                    if mae < best_mae:
                        best_mae = mae
                        best_name = name

                if best_name:
                    logger.info(
                        f"[Orchestrator] Round {self._n_restarts}: "
                        f"champion-first fallback to {best_name} (MAE={best_mae:.4f})"
                    )
                    self._board.write(
                        "composer/selected_models", [best_name], "orchestrator"
                    )
                    self._board.write(
                        "composer/weights", {best_name: 1.0}, "orchestrator"
                    )
                    self._board.write(
                        "composer/composition_strategy",
                        "champion_first_fallback",
                        "orchestrator",
                    )
                    # Force accept on champion-first fallback
                    self._board.write("critic/verdict", "accept", "orchestrator")
                    self._board.write("critic/guard_triggered", False, "orchestrator")
                    verdict = "accept"
                    guard_triggered = False
                    break

            # Re-compose and re-critique with adjusted params
            self._agents["composer"].execute(self._board)
            self._agents["critic"].execute(self._board)
            verdict = self._board.read("critic/verdict", "warn")
            guard_triggered = self._board.read("critic/guard_triggered", False)

        result = {
            "verdict": verdict,
            "guard_triggered": guard_triggered,
            "selected_models": self._board.read("composer/selected_models", []),
            "weights": self._board.read("composer/weights", {}),
            "composition_strategy": self._board.read("composer/composition_strategy", "unknown"),
            "diversity": self._board.read("composer/ensemble_diversity", 0.0),
            "quality_report": self._board.read("critic/quality_report", {}),
            "suggestions": self._board.read("critic/improvement_suggestions", []),
            "n_restarts": self._n_restarts,
            "timing": dict(self._timing),
            # V7.3 champion transfer context
            "champion_expected_winner": self._board.read("champion/expected_winner", ""),
            "champion_force_include": self._board.read("champion/force_include", []),
            "champion_knowledge_confidence": self._board.read("champion/knowledge_confidence", 0.0),
        }

        logger.info(
            f"[Orchestrator] Final: verdict={verdict}, "
            f"models={len(result['selected_models'])}, "
            f"strategy={result['composition_strategy']}, "
            f"restarts={self._n_restarts}"
        )
        return result

    def get_full_telemetry(self) -> Dict[str, Any]:
        """Return complete orchestration telemetry."""
        return {
            "board_snapshot": {
                k: v for k, v in self._board.snapshot().items()
                if not k.startswith("input/")  # Skip large arrays
            },
            "audit_log": self._board.get_audit_log(),
            "timing": dict(self._timing),
            "n_restarts": self._n_restarts,
        }
