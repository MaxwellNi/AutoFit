#!/usr/bin/env python3
"""
Two-Stage AutoFit Model Selector for Block 3.

Stage A: Rule-based shortlisting
    - Uses meta-features to narrow down to ~10 candidate models
    - Based on rule_based_composer.py rules
    
Stage B: Budgeted selection (ASHA/Successive Halving)
    - Runs candidates on progressively larger data subsets
    - Eliminates poor performers early
    - Returns top-K models with confidence intervals

Outputs:
- Full audit trail with decisions at each stage
- Ranking with uncertainty estimates
- Recommended model ensemble (if beneficial)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .rule_based_composer import (
    RuleBasedComposer,
    ComposerConfig,
    ComposedConfig,
    get_profile_path_from_pointer,
)


@dataclass
class CandidateModel:
    """A candidate model for Stage B selection."""
    name: str
    category: str
    config: Dict[str, Any]
    priority: int = 0  # Higher = more likely to be good based on Stage A
    
    # Stage B results
    scores: List[float] = field(default_factory=list)  # Scores at each rung
    rungs_completed: int = 0
    eliminated_at_rung: int = -1
    total_train_time: float = 0.0
    
    def mean_score(self) -> float:
        return np.mean(self.scores) if self.scores else float('inf')
    
    def std_score(self) -> float:
        return np.std(self.scores) if len(self.scores) > 1 else 0.0


@dataclass 
class StageAResult:
    """Output of Stage A: shortlisted candidates."""
    candidates: List[CandidateModel]
    composed_config: ComposedConfig
    decisions: List[Dict[str, Any]]
    meta_features: Dict[str, float]


@dataclass
class StageBResult:
    """Output of Stage B: ranked models."""
    ranking: List[CandidateModel]
    best_model: CandidateModel
    best_score: float
    best_score_ci: Tuple[float, float]  # 95% CI
    rungs_completed: int
    total_budget_used: float  # In seconds
    audit_trail: List[Dict[str, Any]]


@dataclass
class AutoFitResult:
    """Complete AutoFit output."""
    stage_a: StageAResult
    stage_b: Optional[StageBResult]
    final_recommendation: str
    ensemble_recommendation: Optional[List[str]]
    audit_path: Optional[Path]


class TwoStageSelector:
    """
    Two-stage model selection with audit trail.
    
    Stage A: Rule-based shortlisting based on data profile
    Stage B: ASHA-style successive halving on shortlist
    """
    
    # Model pool organized by category
    MODEL_POOL = {
        "statistical": [
            {"name": "SeasonalNaive", "config": {}},
            {"name": "AutoARIMA", "config": {}},
            {"name": "ETS", "config": {}},
            {"name": "Theta", "config": {}},
            {"name": "MSTL", "config": {}},
        ],
        "ml_tabular": [
            {"name": "Ridge", "config": {"alpha": 1.0}},
            {"name": "LightGBM", "config": {"n_estimators": 100}},
            {"name": "XGBoost", "config": {"n_estimators": 100}},
            {"name": "CatBoost", "config": {"iterations": 100}},
            {"name": "HistGradientBoosting", "config": {"max_iter": 100}},
            {"name": "RandomForest", "config": {"n_estimators": 100}},
        ],
        "deep_classical": [
            {"name": "NBEATS", "config": {"max_steps": 50}},
            {"name": "NHITS", "config": {"max_steps": 50}},
            {"name": "TFT", "config": {"max_steps": 50}},
            {"name": "DeepAR", "config": {"max_steps": 50}},
        ],
        "transformer_sota": [
            {"name": "PatchTST", "config": {"max_steps": 50}},
            {"name": "iTransformer", "config": {"max_steps": 50}},
            {"name": "TimesNet", "config": {"max_steps": 50}},
            {"name": "TSMixer", "config": {"max_steps": 50}},
        ],
        "foundation": [
            {"name": "TimesFM", "config": {}},
            {"name": "Chronos", "config": {}},
        ],
        "irregular_aware": [
            {"name": "GRU-D", "config": {"hidden_size": 64}},
        ],
    }
    
    def __init__(
        self,
        composer_config: Optional[ComposerConfig] = None,
        budget_seconds: float = 3600.0,
        n_rungs: int = 4,
        reduction_factor: float = 3.0,
        min_candidates: int = 3,
    ):
        """
        Args:
            composer_config: Configuration for Stage A rule-based selection
            budget_seconds: Total compute budget for Stage B
            n_rungs: Number of successive halving rungs
            reduction_factor: How many candidates to eliminate per rung
            min_candidates: Minimum candidates to keep after Stage A
        """
        self.composer = RuleBasedComposer(composer_config)
        self.budget_seconds = budget_seconds
        self.n_rungs = n_rungs
        self.reduction_factor = reduction_factor
        self.min_candidates = min_candidates
    
    def run_stage_a(self, meta_features: Dict[str, float]) -> StageAResult:
        """
        Stage A: Rule-based shortlisting.
        
        Uses meta-features to assign priority scores to models,
        then returns top candidates.
        """
        # Get composed config for backbone recommendation
        composed = self.composer.compose(meta_features)
        
        candidates = []
        decisions = []
        
        # Score each model based on meta-features
        for category, models in self.MODEL_POOL.items():
            for model_spec in models:
                priority = self._compute_priority(
                    model_spec["name"], 
                    category, 
                    meta_features,
                    composed
                )
                
                candidate = CandidateModel(
                    name=model_spec["name"],
                    category=category,
                    config=model_spec["config"].copy(),
                    priority=priority,
                )
                candidates.append(candidate)
                
                decisions.append({
                    "model": model_spec["name"],
                    "category": category,
                    "priority": priority,
                    "reason": self._get_priority_reason(model_spec["name"], category, meta_features),
                })
        
        # Sort by priority and take top candidates
        candidates.sort(key=lambda c: c.priority, reverse=True)
        
        # Keep at least min_candidates, more if within priority threshold
        top_priority = candidates[0].priority if candidates else 0
        threshold = top_priority * 0.5
        
        shortlisted = [c for c in candidates if c.priority >= threshold]
        if len(shortlisted) < self.min_candidates:
            shortlisted = candidates[:self.min_candidates]
        
        return StageAResult(
            candidates=shortlisted,
            composed_config=composed,
            decisions=decisions,
            meta_features=meta_features,
        )
    
    def _compute_priority(
        self, 
        model_name: str, 
        category: str, 
        meta: Dict[str, float],
        composed: ComposedConfig
    ) -> int:
        """Compute priority score for a model."""
        priority = 50  # Base priority
        
        # Boost if matches composed backbone recommendation
        if model_name.lower() == composed.backbone.lower():
            priority += 30
        
        # Category-specific boosts based on meta-features
        nonstationarity = meta.get("nonstationarity_score", 0.0)
        periodicity = meta.get("periodicity_score", 0.0)
        irregular = meta.get("irregular_score", 0.0)
        heavy_tail = meta.get("heavy_tail_score", 0.0)
        exog = meta.get("exog_strength", 0.0)
        missing = meta.get("missing_rate", 0.0)
        
        # Statistical models: good for simple patterns
        if category == "statistical":
            if periodicity > 0.3:
                priority += 20  # Good at periodicity
            if nonstationarity < 0.3:
                priority += 10  # Assumes stationarity
            if missing > 0.3:
                priority -= 20  # Don't handle missing well
        
        # ML tabular: robust, handles many scenarios
        if category == "ml_tabular":
            priority += 10  # Always a solid baseline
            if heavy_tail > 0.3:
                priority += 10  # Tree models handle outliers
        
        # Deep classical: good for complex patterns
        if category == "deep_classical":
            if nonstationarity > 0.3:
                priority += 15  # Can learn non-stationary
            if periodicity > 0.4:
                priority += 10
        
        # Transformer SOTA: best for complex long-range
        if category == "transformer_sota":
            if meta.get("long_memory_score", 0.0) > 0.3:
                priority += 20
            if periodicity > 0.3:
                priority += 15
        
        # Foundation: good zero-shot, expensive
        if category == "foundation":
            priority += 5  # Bonus for zero-shot capability
            if exog > 0.2:
                priority -= 10  # Can't easily use exogenous
        
        # Irregular-aware: specifically for irregular data
        if category == "irregular_aware":
            if irregular > 0.3:
                priority += 30
            if missing > 0.3:
                priority += 20
            else:
                priority -= 20  # Overkill for regular data
        
        return priority
    
    def _get_priority_reason(self, model_name: str, category: str, meta: Dict[str, float]) -> str:
        """Generate human-readable reason for priority."""
        reasons = []
        
        if category == "statistical":
            reasons.append("Statistical baseline")
        elif category == "ml_tabular":
            reasons.append("Robust ML baseline")
        elif category == "deep_classical":
            reasons.append("Deep learning approach")
        elif category == "transformer_sota":
            reasons.append("SOTA transformer architecture")
        elif category == "foundation":
            reasons.append("Pre-trained foundation model")
        elif category == "irregular_aware":
            reasons.append("Designed for irregular data")
        
        if meta.get("irregular_score", 0.0) > 0.3 and category == "irregular_aware":
            reasons.append("matches high irregularity")
        if meta.get("periodicity_score", 0.0) > 0.3 and category in ("statistical", "transformer_sota"):
            reasons.append("matches periodic patterns")
        
        return "; ".join(reasons)
    
    def run_stage_b(
        self,
        candidates: List[CandidateModel],
        eval_fn,  # Callable[[str, Dict, float], float] - returns validation score
        progress_callback=None,
    ) -> StageBResult:
        """
        Stage B: ASHA-style successive halving.
        
        Args:
            candidates: Shortlisted candidates from Stage A
            eval_fn: Function (model_name, config, data_fraction) -> score
            progress_callback: Optional callback for progress updates
        
        Returns:
            StageBResult with ranking and audit trail
        """
        audit_trail = []
        total_time_used = 0.0
        
        # Compute data fractions for each rung
        # Rung 0: small fraction, Rung n-1: full data
        fractions = [1.0 / (self.reduction_factor ** (self.n_rungs - 1 - i)) for i in range(self.n_rungs)]
        
        active_candidates = list(candidates)
        
        for rung in range(self.n_rungs):
            if len(active_candidates) == 0:
                break
            
            frac = fractions[rung]
            rung_start = time.time()
            
            audit_trail.append({
                "rung": rung,
                "data_fraction": frac,
                "n_candidates": len(active_candidates),
                "candidates": [c.name for c in active_candidates],
            })
            
            if progress_callback:
                progress_callback(f"Rung {rung + 1}/{self.n_rungs}: {len(active_candidates)} candidates, {frac:.1%} data")
            
            # Evaluate each candidate
            for candidate in active_candidates:
                try:
                    start_time = time.time()
                    score = eval_fn(candidate.name, candidate.config, frac)
                    elapsed = time.time() - start_time
                    
                    candidate.scores.append(score)
                    candidate.rungs_completed = rung + 1
                    candidate.total_train_time += elapsed
                    total_time_used += elapsed
                    
                except Exception as e:
                    # Failed evaluation = worst score
                    candidate.scores.append(float('inf'))
                    audit_trail.append({
                        "event": "evaluation_error",
                        "rung": rung,
                        "model": candidate.name,
                        "error": str(e),
                    })
            
            # Sort by mean score (lower is better)
            active_candidates.sort(key=lambda c: c.mean_score())
            
            # Record rung results
            audit_trail.append({
                "rung_complete": rung,
                "rankings": [
                    {"name": c.name, "mean_score": c.mean_score(), "std": c.std_score()}
                    for c in active_candidates
                ],
            })
            
            # Eliminate bottom candidates (keep top 1/reduction_factor)
            if rung < self.n_rungs - 1:
                n_keep = max(1, int(len(active_candidates) / self.reduction_factor))
                eliminated = active_candidates[n_keep:]
                for c in eliminated:
                    c.eliminated_at_rung = rung
                active_candidates = active_candidates[:n_keep]
            
            # Check budget
            if total_time_used > self.budget_seconds:
                audit_trail.append({
                    "event": "budget_exceeded",
                    "time_used": total_time_used,
                    "budget": self.budget_seconds,
                })
                break
        
        # Final ranking
        all_candidates = list(candidates)
        all_candidates.sort(key=lambda c: (c.eliminated_at_rung if c.eliminated_at_rung >= 0 else self.n_rungs, c.mean_score()))
        
        best = all_candidates[0]
        
        # Compute confidence interval for best model
        if len(best.scores) > 1:
            se = best.std_score() / np.sqrt(len(best.scores))
            ci = (best.mean_score() - 1.96 * se, best.mean_score() + 1.96 * se)
        else:
            ci = (best.mean_score(), best.mean_score())
        
        return StageBResult(
            ranking=all_candidates,
            best_model=best,
            best_score=best.mean_score(),
            best_score_ci=ci,
            rungs_completed=max(c.rungs_completed for c in candidates),
            total_budget_used=total_time_used,
            audit_trail=audit_trail,
        )
    
    def select(
        self,
        meta_features: Dict[str, float],
        eval_fn=None,
        skip_stage_b: bool = False,
        output_dir: Optional[Path] = None,
    ) -> AutoFitResult:
        """
        Run full two-stage selection.
        
        Args:
            meta_features: Data profile meta-features
            eval_fn: Evaluation function for Stage B
            skip_stage_b: If True, only run Stage A (for quick selection)
            output_dir: Optional directory for audit artifacts
        
        Returns:
            AutoFitResult with full audit trail
        """
        # Stage A
        stage_a = self.run_stage_a(meta_features)
        
        # Stage B (if not skipped and eval_fn provided)
        stage_b = None
        if not skip_stage_b and eval_fn is not None:
            stage_b = self.run_stage_b(stage_a.candidates, eval_fn)
        
        # Final recommendation
        if stage_b is not None:
            final = stage_b.best_model.name
            
            # Check if ensemble would help
            # (if top 3 have similar scores, ensemble may be beneficial)
            top_3 = stage_b.ranking[:3]
            if len(top_3) >= 3:
                score_range = top_3[-1].mean_score() - top_3[0].mean_score()
                if score_range < 0.1 * top_3[0].mean_score():
                    ensemble = [c.name for c in top_3]
                else:
                    ensemble = None
            else:
                ensemble = None
        else:
            # Stage A only: use highest priority
            final = stage_a.candidates[0].name if stage_a.candidates else "LightGBM"
            ensemble = None
        
        result = AutoFitResult(
            stage_a=stage_a,
            stage_b=stage_b,
            final_recommendation=final,
            ensemble_recommendation=ensemble,
            audit_path=None,
        )
        
        # Write audit trail
        if output_dir is not None:
            result.audit_path = self._write_audit(result, output_dir)
        
        return result
    
    def _write_audit(self, result: AutoFitResult, output_dir: Path) -> Path:
        """Write audit trail to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audit = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "final_recommendation": result.final_recommendation,
            "ensemble_recommendation": result.ensemble_recommendation,
            "stage_a": {
                "meta_features": result.stage_a.meta_features,
                "composed_config": result.stage_a.composed_config.to_dict(),
                "decisions": result.stage_a.decisions,
                "shortlisted": [
                    {"name": c.name, "category": c.category, "priority": c.priority}
                    for c in result.stage_a.candidates
                ],
            },
        }
        
        if result.stage_b is not None:
            audit["stage_b"] = {
                "best_model": result.stage_b.best_model.name,
                "best_score": result.stage_b.best_score,
                "best_score_ci": list(result.stage_b.best_score_ci),
                "rungs_completed": result.stage_b.rungs_completed,
                "total_budget_used_seconds": result.stage_b.total_budget_used,
                "final_ranking": [
                    {
                        "name": c.name,
                        "mean_score": c.mean_score(),
                        "std_score": c.std_score(),
                        "rungs_completed": c.rungs_completed,
                        "eliminated_at_rung": c.eliminated_at_rung,
                    }
                    for c in result.stage_b.ranking
                ],
                "audit_trail": result.stage_b.audit_trail,
            }
        
        audit_path = output_dir / "autofit_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        
        return audit_path


def select_from_profile(
    profile_path: Path,
    eval_fn=None,
    skip_stage_b: bool = True,
    output_dir: Optional[Path] = None,
) -> AutoFitResult:
    """
    Convenience function for two-stage selection from profile.
    
    Args:
        profile_path: Path to profile.json
        eval_fn: Optional evaluation function for Stage B
        skip_stage_b: If True, only run Stage A
        output_dir: Optional output directory for audit
    
    Returns:
        AutoFitResult
    """
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    meta_features = profile.get("meta_features", {})
    
    selector = TwoStageSelector()
    return selector.select(
        meta_features,
        eval_fn=eval_fn,
        skip_stage_b=skip_stage_b,
        output_dir=output_dir,
    )
