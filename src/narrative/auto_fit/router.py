#!/usr/bin/env python3
"""
AutoFit v2 — Mixture-of-Experts Router (Section B / D2).

Implements the gating network that routes instances to the correct expert
based on meta-features.

Architecture (paper contribution):
    ┌─────────────────┐
    │  Meta-features  │ (25+ dims from meta_features_v2)
    └───────┬─────────┘
            │
    ┌───────▼─────────┐
    │  Gating Network │  h = ReLU(W1 · mf + b1)  →  softmax(W2 · h + b2)
    └───────┬─────────┘
            │ ← K expert weights (sparse top-k)
    ┌───────┼─────────────────┐
    ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
    │Expert1│ │Expert2│ │ExpertK│
    │Tabular│ │ Deep  │ │Fndatn │
    └───┬───┘ └───┬───┘ └───┬───┘
        │         │         │
        └─────────┼─────────┘
           weighted combination
                  │
           ┌──────▼──────┐
           │ Final pred  │
           └─────────────┘

Expert categories:
    1. TabularExpert    — LightGBM / CatBoost with engineered features
    2. DeepExpert       — NHITS / PatchTST with learned features
    3. FoundationExpert — Chronos / Moirai zero-shot
    4. StatisticalExpert— AutoARIMA / ETS for strong periodicity

Gating modes:
    - "hard":   argmax → route to single expert (Stage A only)
    - "soft":   weighted combination of top-K experts
    - "sparse": top-K with renormalized weights (default, K=2)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .meta_features_v2 import MetaFeaturesV2

logger = logging.getLogger(__name__)


class GatingMode(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    SPARSE = "sparse"


@dataclass
class ExpertSpec:
    """Specification for one expert in the MoE."""
    name: str
    category: str          # matches MODEL_CATEGORIES keys
    models: List[str]      # candidate model names within this expert
    priority_boost: float = 0.0  # static bias from meta-features

    def __repr__(self):
        return f"ExpertSpec({self.name}, models={self.models})"


@dataclass
class GatingResult:
    """Output of the gating network."""
    expert_weights: Dict[str, float]     # expert_name → weight
    top_experts: List[str]               # ordered by weight
    gating_mode: GatingMode
    meta_feature_vector: List[float]     # raw input to gating
    rationale: List[str]                 # human-readable explanations


# ---------------------------------------------------------------------------
# Expert Catalogue
# ---------------------------------------------------------------------------

DEFAULT_EXPERTS = [
    ExpertSpec(
        name="tabular",
        category="ml_tabular",
        models=["LightGBM", "CatBoost", "XGBoost", "HistGradientBoosting", "RandomForest"],
    ),
    ExpertSpec(
        name="deep_ts",
        category="deep_classical",
        models=["NHITS", "NBEATS", "TFT", "DeepAR"],
    ),
    ExpertSpec(
        name="transformer",
        category="transformer_sota",
        models=["PatchTST", "iTransformer", "TimesNet", "TSMixer"],
    ),
    ExpertSpec(
        name="foundation",
        category="foundation",
        models=["Chronos", "Moirai"],  # TimesFM removed: Python 3.12 incompatible
    ),
    ExpertSpec(
        name="statistical",
        category="statistical",
        models=["AutoARIMA", "AutoETS", "AutoTheta", "MSTL"],
    ),
    ExpertSpec(
        name="irregular",
        category="irregular",
        models=["GRU-D", "SAITS"],
    ),
]


# ---------------------------------------------------------------------------
# Gating Network
# ---------------------------------------------------------------------------

class MetaFeatureRouter:
    """
    Rule-based + learnable gating over expert categories.

    Phase 1 (v2.0): Fully rule-based — deterministic routing from meta-features.
    Phase 2 (v2.1): Learnable weights via cross-validation.

    The router converts MetaFeaturesV2 into a weight vector over experts.
    """

    def __init__(
        self,
        experts: Optional[List[ExpertSpec]] = None,
        gating_mode: GatingMode = GatingMode.SPARSE,
        top_k: int = 2,
        temperature: float = 1.0,
    ):
        self.experts = experts or list(DEFAULT_EXPERTS)
        self.gating_mode = gating_mode
        self.top_k = min(top_k, len(self.experts))
        self.temperature = temperature
        self._expert_names = [e.name for e in self.experts]

    def route(self, mf: MetaFeaturesV2) -> GatingResult:
        """
        Route based on meta-features → expert weights.

        Returns GatingResult with expert weights and rationale.
        """
        # Compute raw scores per expert
        scores = {}
        rationale = []

        for expert in self.experts:
            score, reasons = self._score_expert(expert, mf)
            scores[expert.name] = score + expert.priority_boost
            rationale.extend(reasons)

        # Apply temperature + softmax
        names = list(scores.keys())
        raw = np.array([scores[n] for n in names])
        weights = self._softmax(raw / self.temperature)

        # Apply gating mode
        if self.gating_mode == GatingMode.HARD:
            idx = np.argmax(weights)
            final_weights = {n: (1.0 if i == idx else 0.0) for i, n in enumerate(names)}
        elif self.gating_mode == GatingMode.SPARSE:
            top_idx = np.argsort(weights)[::-1][:self.top_k]
            mask = np.zeros_like(weights)
            mask[top_idx] = weights[top_idx]
            mask = mask / mask.sum() if mask.sum() > 0 else mask
            final_weights = {n: float(mask[i]) for i, n in enumerate(names)}
        else:  # SOFT
            final_weights = {n: float(weights[i]) for i, n in enumerate(names)}

        # Order by weight
        sorted_experts = sorted(final_weights, key=lambda n: final_weights[n], reverse=True)

        return GatingResult(
            expert_weights=final_weights,
            top_experts=sorted_experts,
            gating_mode=self.gating_mode,
            meta_feature_vector=self._mf_to_vector(mf),
            rationale=rationale,
        )

    def _score_expert(
        self, expert: ExpertSpec, mf: MetaFeaturesV2
    ) -> Tuple[float, List[str]]:
        """
        Compute affinity score for an expert based on meta-features.

        Returns (score, list_of_rationale_strings).
        """
        score = 0.0
        reasons = []

        name = expert.name

        # --- Tabular expert ---
        if name == "tabular":
            # Always a strong baseline
            score += 5.0
            reasons.append(f"[tabular] baseline boost +5")
            # Better with heavy tails (tree-based robustness)
            if mf.kurtosis_mean > 5.0:
                score += 3.0
                reasons.append(f"[tabular] heavy-tail kurtosis={mf.kurtosis_mean:.1f} → +3")
            # Better with moderate missing (native handling)
            if 0.05 < mf.missing_rate_global < 0.4:
                score += 2.0
                reasons.append(f"[tabular] moderate missing={mf.missing_rate_global:.2f} → +2")
            # Penalise if strong periodicity (trees are weaker here)
            if mf.acf_lag7_mean > 0.5:
                score -= 2.0
                reasons.append(f"[tabular] strong periodicity ACF7={mf.acf_lag7_mean:.2f} → -2")

        # --- Deep TS expert ---
        elif name == "deep_ts":
            score += 3.0
            reasons.append(f"[deep_ts] baseline boost +3")
            if mf.nonstationarity_score > 0.5:
                score += 3.0
                reasons.append(f"[deep_ts] non-stationary={mf.nonstationarity_score:.2f} → +3")
            if mf.multiscale_score > 0.3:
                score += 2.0
                reasons.append(f"[deep_ts] multiscale={mf.multiscale_score:.2f} → +2")
            if mf.n_rows < 5000:
                score -= 3.0
                reasons.append(f"[deep_ts] small data n={mf.n_rows} → -3")

        # --- Transformer expert ---
        elif name == "transformer":
            score += 2.0
            reasons.append(f"[transformer] baseline boost +2")
            if mf.acf_lag30_mean > 0.3:
                score += 3.0
                reasons.append(f"[transformer] long-memory ACF30={mf.acf_lag30_mean:.2f} → +3")
            if mf.multiscale_score > 0.4:
                score += 2.0
                reasons.append(f"[transformer] multiscale={mf.multiscale_score:.2f} → +2")
            if mf.n_rows < 10000:
                score -= 2.0
                reasons.append(f"[transformer] limited data n={mf.n_rows} → -2")
            if mf.missing_rate_global > 0.3:
                score -= 2.0
                reasons.append(f"[transformer] high missing → -2")

        # --- Foundation expert ---
        elif name == "foundation":
            score += 1.0
            reasons.append(f"[foundation] baseline boost +1")
            # Foundation models: strong zero-shot but can't use exog
            if mf.exog_strength > 0.3:
                score -= 3.0
                reasons.append(f"[foundation] strong exog={mf.exog_strength:.2f} → -3 (can't use)")
            if mf.n_rows < 1000:
                score += 3.0
                reasons.append(f"[foundation] few-shot advantage n={mf.n_rows} → +3")
            if mf.acf_lag7_mean > 0.4:
                score += 2.0
                reasons.append(f"[foundation] periodic → pre-trained advantage +2")

        # --- Statistical expert ---
        elif name == "statistical":
            score += 2.0
            reasons.append(f"[statistical] baseline boost +2")
            if mf.acf_lag7_mean > 0.5:
                score += 4.0
                reasons.append(f"[statistical] strong periodicity ACF7={mf.acf_lag7_mean:.2f} → +4")
            if mf.nonstationarity_score < 0.3:
                score += 2.0
                reasons.append(f"[statistical] near-stationary → +2")
            if mf.missing_rate_global > 0.2:
                score -= 3.0
                reasons.append(f"[statistical] high missing → -3")

        # --- Irregular expert ---
        elif name == "irregular":
            score += 0.0
            if mf.sampling_interval_cv > 0.5:
                score += 5.0
                reasons.append(f"[irregular] high CV={mf.sampling_interval_cv:.2f} → +5")
            if mf.pct_gaps_gt_7d > 0.3:
                score += 3.0
                reasons.append(f"[irregular] many >7d gaps → +3")
            if mf.missing_rate_global > 0.4:
                score += 3.0
                reasons.append(f"[irregular] high missing → +3")
            if mf.sampling_interval_cv < 0.1:
                score -= 5.0
                reasons.append(f"[irregular] regular data → -5 (not needed)")

        return score, reasons

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _mf_to_vector(self, mf: MetaFeaturesV2) -> List[float]:
        """Convert MetaFeaturesV2 to fixed-order numeric vector for gating."""
        return [
            mf.missing_rate_global,
            mf.sampling_interval_cv,
            mf.acf_lag7_mean,
            mf.acf_lag30_mean,
            mf.acf_lag90_mean,
            mf.multiscale_score,
            mf.kurtosis_mean,
            mf.tail_index_proxy,
            mf.nonstationarity_score,
            mf.rolling_mean_drift,
            mf.exog_strength,
            mf.edgar_strength,
            mf.text_strength,
            float(mf.n_entities),
            float(mf.n_rows),
            mf.pct_gaps_gt_7d,
            mf.pct_outliers_3sigma,
        ]


# ---------------------------------------------------------------------------
# Expert Model Selector
# ---------------------------------------------------------------------------

def select_expert_models(
    gating: GatingResult,
    experts: List[ExpertSpec],
    max_models_per_expert: int = 2,
) -> List[Tuple[str, str, float]]:
    """
    Given gating result, select specific models from each active expert.

    Returns list of (model_name, expert_name, weight).
    """
    selected = []
    expert_map = {e.name: e for e in experts}

    for expert_name in gating.top_experts:
        w = gating.expert_weights.get(expert_name, 0.0)
        if w < 0.01:
            continue
        expert = expert_map.get(expert_name)
        if expert is None:
            continue
        for model in expert.models[:max_models_per_expert]:
            selected.append((model, expert_name, w))

    return selected
