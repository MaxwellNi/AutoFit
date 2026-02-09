#!/usr/bin/env python3
"""
AutoFit v1 / v2 Wrappers — First-Class Benchmark Competitors.

These wrappers implement ModelBase.fit() / predict() so that AutoFit
participates in the Block 3 benchmark harness exactly like any other model.

- AutoFitV1  : rule-based composer  (RuleBasedComposer)
- AutoFitV2  : MoE router + meta-feature gating (autofit_v2 pipeline)
- AutoFitV2E : v2 with soft-ensemble over top-K experts

The wrapper:
    1. In fit(): computes meta-features from train_raw panel data,
       routes through the gating network, selects the winning model,
       instantiates it from the registry, and calls model.fit().
    2. In predict(): delegates to the trained underlying model.

This means AutoFit's train_time includes the meta-feature + routing overhead,
giving a fair comparison.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import ModelBase, ModelConfig

logger = logging.getLogger(__name__)


# ============================================================================
# AutoFit v2 Wrapper
# ============================================================================

class AutoFitV2Wrapper(ModelBase):
    """
    AutoFit v2 as a first-class benchmark model.

    fit(X, y, **kw):
        1. Compute meta-features on train_raw panel.
        2. Run MoE gating → top model(s).
        3. Instantiate winning model from registry.
        4. Train it on (X, y).

    predict(X, **kw):
        Delegate to the trained underlying model.
    """

    def __init__(self, ensemble: bool = False, top_k: int = 2, **kwargs):
        name = "AutoFitV2E" if ensemble else "AutoFitV2"
        config = ModelConfig(
            name=name,
            model_type="regression",
            params={"ensemble": ensemble, "top_k": top_k},
        )
        super().__init__(config)
        self._ensemble = ensemble
        self._top_k = top_k
        self._underlying: Optional[ModelBase] = None
        self._underlying_models: List[Tuple[ModelBase, float]] = []  # (model, weight)
        self._routing_info: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AutoFitV2Wrapper":
        from src.narrative.auto_fit.meta_features_v2 import compute_meta_features
        from src.narrative.auto_fit.router import (
            DEFAULT_EXPERTS,
            GatingMode,
            MetaFeatureRouter,
            select_expert_models,
        )
        from src.narrative.block3.models.registry import get_model, check_model_available

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target", y.name or "funding_raised_usd")
        horizon = kwargs.get("horizon", 7)

        # --- Stage 1: Meta-features ---
        t0 = time.monotonic()
        if train_raw is not None and len(train_raw) > 0:
            try:
                mf = compute_meta_features(
                    train_raw,
                    entity_col="entity_id",
                    date_col="crawled_date_day",
                    target_col=target,
                    max_entities_sample=500,
                    seed=42,
                )
                logger.info(
                    f"[AutoFitV2] Meta-features computed: n_ent={mf.n_entities}, "
                    f"missing={mf.missing_rate_global:.3f}, "
                    f"nonstat={mf.nonstationarity_score:.3f}"
                )
            except Exception as e:
                logger.warning(f"[AutoFitV2] Meta-feature computation failed: {e}; using defaults")
                from src.narrative.auto_fit.meta_features_v2 import MetaFeaturesV2
                mf = MetaFeaturesV2()
        else:
            logger.warning("[AutoFitV2] No train_raw provided; using default meta-features")
            from src.narrative.auto_fit.meta_features_v2 import MetaFeaturesV2
            mf = MetaFeaturesV2()

        # --- Stage 2: Routing ---
        # Remove TimesFM from foundation expert (Python 3.12 incompatible)
        experts = []
        for exp in DEFAULT_EXPERTS:
            models = [m for m in exp.models if m != "TimesFM"]
            from src.narrative.auto_fit.router import ExpertSpec
            experts.append(ExpertSpec(
                name=exp.name,
                category=exp.category,
                models=models,
                priority_boost=exp.priority_boost,
            ))

        mode = GatingMode.SPARSE if self._ensemble else GatingMode.HARD
        router = MetaFeatureRouter(
            experts=experts,
            gating_mode=mode,
            top_k=self._top_k,
        )
        gating = router.route(mf)
        selected = select_expert_models(gating, experts, max_models_per_expert=2)

        routing_time = time.monotonic() - t0
        logger.info(
            f"[AutoFitV2] Routing done in {routing_time:.2f}s: "
            f"top_experts={gating.top_experts[:3]}, "
            f"selected={[m for m, _, _ in selected]}"
        )

        self._routing_info = {
            "meta_features": mf.to_dict() if hasattr(mf, "to_dict") else {},
            "expert_weights": gating.expert_weights,
            "top_experts": gating.top_experts,
            "selected_models": [(m, e, w) for m, e, w in selected],
            "routing_time_seconds": routing_time,
            "rationale": gating.rationale,
        }

        # --- Stage 3: Instantiate and train ---
        # Determine which category the winning model belongs to
        expert_map = {e.name: e for e in experts}
        panel_categories = {"deep_classical", "transformer_sota", "foundation",
                            "statistical", "irregular"}

        if self._ensemble and len(selected) >= 2:
            # Ensemble mode: train top-K models, weight predictions
            self._underlying_models = []
            for model_name, expert_name, weight in selected:
                if not check_model_available(model_name):
                    logger.warning(f"[AutoFitV2E] {model_name} not available, skipping")
                    continue
                try:
                    model = get_model(model_name)
                    fit_kw = {}
                    exp = expert_map.get(expert_name)
                    if exp and exp.category in panel_categories:
                        fit_kw["train_raw"] = train_raw
                        fit_kw["target"] = target
                        fit_kw["horizon"] = horizon
                    model.fit(X, y, **fit_kw)
                    self._underlying_models.append((model, weight))
                    logger.info(f"[AutoFitV2E] Trained {model_name} (w={weight:.3f})")
                except Exception as e:
                    logger.warning(f"[AutoFitV2E] Failed to train {model_name}: {e}")
            if not self._underlying_models:
                # Fallback to LightGBM
                logger.warning("[AutoFitV2E] All ensemble models failed, falling back to LightGBM")
                self._underlying = get_model("LightGBM")
                self._underlying.fit(X, y)
        else:
            # Single-model mode: train the top-1 model
            winner_name = selected[0][0] if selected else "LightGBM"
            winner_expert = selected[0][1] if selected else "tabular"

            # Check availability, fallback if needed
            if not check_model_available(winner_name):
                logger.warning(
                    f"[AutoFitV2] Winner {winner_name} not available, "
                    f"falling back through selected models..."
                )
                winner_name = "LightGBM"
                winner_expert = "tabular"
                for m, e, w in selected[1:]:
                    if check_model_available(m):
                        winner_name = m
                        winner_expert = e
                        break

            logger.info(f"[AutoFitV2] Training winner: {winner_name}")
            self._underlying = get_model(winner_name)
            fit_kw = {}
            exp = expert_map.get(winner_expert)
            if exp and exp.category in panel_categories:
                fit_kw["train_raw"] = train_raw
                fit_kw["target"] = target
                fit_kw["horizon"] = horizon
            self._underlying.fit(X, y, **fit_kw)

        self._routing_info["final_model"] = (
            self._underlying.name if self._underlying
            else [m.name for m, _ in self._underlying_models]
        )
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if self._underlying_models:
            # Ensemble prediction: weighted average
            preds_list = []
            weights = []
            for model, w in self._underlying_models:
                try:
                    pred_kw = {}
                    # Pass through panel-aware kwargs
                    for k in ("test_raw", "target", "horizon"):
                        if k in kwargs:
                            pred_kw[k] = kwargs[k]
                    p = model.predict(X, **pred_kw)
                    preds_list.append(p)
                    weights.append(w)
                except Exception as e:
                    logger.warning(f"[AutoFitV2E] Predict failed for {model.name}: {e}")
            if preds_list:
                weights = np.array(weights)
                weights = weights / weights.sum()
                preds = np.zeros(len(X))
                for p, w in zip(preds_list, weights):
                    preds += w * p
                return preds
            # Fallback
            return np.full(len(X), 0.0)

        if self._underlying is not None:
            pred_kw = {}
            for k in ("test_raw", "target", "horizon"):
                if k in kwargs:
                    pred_kw[k] = kwargs[k]
            return self._underlying.predict(X, **pred_kw)

        raise RuntimeError("AutoFitV2 not fitted")

    def get_routing_info(self) -> Dict[str, Any]:
        """Return routing metadata for audit/paper tables."""
        return self._routing_info


# ============================================================================
# AutoFit v1 Wrapper (RuleBasedComposer baseline)
# ============================================================================

class AutoFitV1Wrapper(ModelBase):
    """
    AutoFit v1 (rule-based composer) as a benchmark baseline.

    Uses the RuleBasedComposer to select a model family, then trains
    the corresponding model from the registry.
    """

    def __init__(self, **kwargs):
        config = ModelConfig(
            name="AutoFitV1",
            model_type="regression",
            params={},
        )
        super().__init__(config)
        self._underlying: Optional[ModelBase] = None
        self._selection_info: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AutoFitV1Wrapper":
        from src.narrative.auto_fit.rule_based_composer import compose_from_profile
        from src.narrative.block3.models.registry import get_model, check_model_available

        train_raw = kwargs.get("train_raw")
        target = kwargs.get("target", y.name or "funding_raised_usd")
        horizon = kwargs.get("horizon", 7)

        # Compute v1 profile and select model
        selected_model = "LightGBM"  # default
        try:
            if train_raw is not None:
                # v1 uses simpler meta-features from diagnose_dataset
                from src.narrative.auto_fit.diagnose_dataset import diagnose_dataset
                profile = diagnose_dataset(train_raw, target_col=target)
                result = compose_from_profile(profile)
                # v1 returns a backbone recommendation; map to registry model
                backbone = result.get("backbone", "lightgbm") if isinstance(result, dict) else "lightgbm"
                backbone_to_model = {
                    "lightgbm": "LightGBM",
                    "xgboost": "XGBoost",
                    "catboost": "CatBoost",
                    "random_forest": "RandomForest",
                    "ridge": "Ridge",
                    "nhits": "NHITS",
                    "nbeats": "NBEATS",
                    "patchtst": "PatchTST",
                    "autoarima": "AutoARIMA",
                    "autoets": "AutoETS",
                    "chronos": "Chronos",
                }
                selected_model = backbone_to_model.get(
                    backbone.lower().replace("-", "").replace("_", ""),
                    "LightGBM",
                )
                self._selection_info = {
                    "profile": profile if isinstance(profile, dict) else {},
                    "composer_result": result if isinstance(result, dict) else {},
                    "selected_model": selected_model,
                }
        except Exception as e:
            logger.warning(f"[AutoFitV1] Composer failed: {e}; defaulting to LightGBM")
            self._selection_info = {"error": str(e), "selected_model": "LightGBM"}

        if not check_model_available(selected_model):
            logger.warning(f"[AutoFitV1] {selected_model} unavailable, using LightGBM")
            selected_model = "LightGBM"

        logger.info(f"[AutoFitV1] Selected model: {selected_model}")
        self._underlying = get_model(selected_model)

        # Train with appropriate kwargs
        fit_kw = {}
        panel_categories = {"deep_classical", "transformer_sota", "foundation",
                            "statistical", "irregular"}
        # Determine category of selected model
        from src.narrative.block3.models.registry import MODEL_CATEGORIES
        model_cat = None
        for cat, models in MODEL_CATEGORIES.items():
            if selected_model in models:
                model_cat = cat
                break
        if model_cat in panel_categories:
            fit_kw["train_raw"] = train_raw
            fit_kw["target"] = target
            fit_kw["horizon"] = horizon

        self._underlying.fit(X, y, **fit_kw)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if self._underlying is None:
            raise RuntimeError("AutoFitV1 not fitted")
        pred_kw = {}
        for k in ("test_raw", "target", "horizon"):
            if k in kwargs:
                pred_kw[k] = kwargs[k]
        return self._underlying.predict(X, **pred_kw)


# ============================================================================
# Factory functions for the registry
# ============================================================================

def get_autofit_v1(**kwargs) -> AutoFitV1Wrapper:
    return AutoFitV1Wrapper(**kwargs)

def get_autofit_v2(**kwargs) -> AutoFitV2Wrapper:
    return AutoFitV2Wrapper(ensemble=False, **kwargs)

def get_autofit_v2e(**kwargs) -> AutoFitV2Wrapper:
    return AutoFitV2Wrapper(ensemble=True, top_k=2, **kwargs)


AUTOFIT_MODELS = {
    "AutoFitV1": get_autofit_v1,
    "AutoFitV2": get_autofit_v2,
    "AutoFitV2E": get_autofit_v2e,
}
