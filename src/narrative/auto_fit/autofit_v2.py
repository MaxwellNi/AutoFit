#!/usr/bin/env python3
"""
AutoFit v2 — Main Entry Point (Section B / D2).

The core method for the KDD'26 full paper:

    AutoFit v2: Data-Adaptive Model Selection via
    Meta-Feature–Gated Mixture of Experts for Financial Time Series

Pipeline:
    1. Compute meta-features (meta_features_v2.py)
    2. Route to expert categories (router.py)
    3. Within each active expert, select best model via ASHA
       (search_budget.py)
    4. Produce final prediction = Σ wₖ · expertₖ(x)
    5. Write full audit trail

Usage:
    # From Python
    result = autofit_v2(
        df=panel_data,
        target_col="funding_raised_usd",
        task="task1_outcome",
    )

    # From CLI
    python -m narrative.auto_fit.autofit_v2 \\
        --pointer docs/audits/FULL_SCALE_POINTER.yaml \\
        --task task1_outcome --target funding_raised_usd
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .meta_features_v2 import (
    MetaFeaturesV2,
    compute_meta_features,
    save_meta_features_report,
)
from .router import (
    DEFAULT_EXPERTS,
    ExpertSpec,
    GatingMode,
    GatingResult,
    MetaFeatureRouter,
    select_expert_models,
)
from .search_budget import (
    ASHACandidate,
    ASHAResult,
    run_asha,
    save_asha_result,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AutoFitV2Result:
    """Complete result of AutoFit v2 pipeline."""

    # Stage 1: Meta-features
    meta_features: MetaFeaturesV2 = field(default_factory=MetaFeaturesV2)

    # Stage 2: Routing
    gating: Optional[GatingResult] = None
    selected_models: List[Tuple[str, str, float]] = field(default_factory=list)
    # (model_name, expert_name, weight)

    # Stage 3: ASHA search (optional)
    asha_result: Optional[ASHAResult] = None

    # Final recommendation
    final_model: str = ""
    final_expert: str = ""
    ensemble: Optional[List[Tuple[str, float]]] = None  # (model, weight)

    # Audit
    total_time_seconds: float = 0.0
    audit_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "final_model": self.final_model,
            "final_expert": self.final_expert,
            "total_time_seconds": self.total_time_seconds,
            "meta_features": self.meta_features.to_dict(),
        }
        if self.gating:
            d["gating"] = {
                "expert_weights": self.gating.expert_weights,
                "top_experts": self.gating.top_experts,
                "mode": self.gating.gating_mode.value,
                "rationale": self.gating.rationale,
            }
        if self.selected_models:
            d["selected_models"] = [
                {"model": m, "expert": e, "weight": w}
                for m, e, w in self.selected_models
            ]
        if self.ensemble:
            d["ensemble"] = [
                {"model": m, "weight": w} for m, w in self.ensemble
            ]
        if self.asha_result:
            d["asha"] = self.asha_result.to_dict()
        return d


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def autofit_v2(
    df: pd.DataFrame,
    *,
    target_col: str = "funding_raised_usd",
    task: str = "task1_outcome",
    entity_col: str = "entity_id",
    date_col: str = "crawled_date_day",
    gating_mode: GatingMode = GatingMode.SPARSE,
    top_k: int = 2,
    experts: Optional[List[ExpertSpec]] = None,
    max_entities_sample: int = 500,
    run_asha_search: bool = False,
    eval_fn: Optional[Callable] = None,
    asha_budget_seconds: float = 1800.0,
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> AutoFitV2Result:
    """
    Run the full AutoFit v2 pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw panel data.
    target_col : str
        Prediction target column.
    task : str
        Block 3 task name (task1_outcome, task2_forecast, task3_risk_adjust).
    gating_mode : GatingMode
        How to combine expert outputs.
    top_k : int
        Number of active experts in sparse mode.
    experts : list of ExpertSpec
        Expert catalogue (defaults to DEFAULT_EXPERTS).
    run_asha_search : bool
        Whether to run ASHA within selected experts.
    eval_fn : callable
        ``eval_fn(model_name, config, entity_fraction) -> float``
        Required if run_asha_search=True.
    asha_budget_seconds : float
        Wall-clock budget for ASHA.
    output_dir : Path
        Where to save audit artifacts.
    seed : int
        Reproducibility seed.

    Returns
    -------
    AutoFitV2Result
    """
    t0 = time.monotonic()
    result = AutoFitV2Result()
    experts = experts or list(DEFAULT_EXPERTS)

    # ---- Stage 1: Meta-features ----
    logger.info("Stage 1: Computing meta-features ...")
    mf = compute_meta_features(
        df,
        entity_col=entity_col,
        date_col=date_col,
        target_col=target_col,
        max_entities_sample=max_entities_sample,
        seed=seed,
    )
    result.meta_features = mf
    logger.info(
        f"  n_entities={mf.n_entities}, n_rows={mf.n_rows}, "
        f"missing={mf.missing_rate_global:.3f}, "
        f"nonstat={mf.nonstationarity_score:.3f}, "
        f"acf7={mf.acf_lag7_mean:.3f}"
    )

    # Warn about leakage suspects
    if mf.leakage_suspects:
        logger.warning(
            f"  LEAKAGE SUSPECTS: {mf.leakage_suspects} "
            f"(max_corr={mf.leakage_max_corr:.4f})"
        )

    # ---- Stage 2: Routing ----
    logger.info("Stage 2: Routing to experts ...")
    router = MetaFeatureRouter(
        experts=experts,
        gating_mode=gating_mode,
        top_k=top_k,
    )
    gating = router.route(mf)
    result.gating = gating

    # Log expert weights
    for name in gating.top_experts[:5]:
        w = gating.expert_weights[name]
        logger.info(f"  {name}: weight={w:.3f}")

    # Select specific models from top experts
    selected = select_expert_models(gating, experts, max_models_per_expert=2)
    result.selected_models = selected
    logger.info(f"  Selected {len(selected)} models: {[m for m, _, _ in selected]}")

    # ---- Stage 3 (optional): ASHA search ----
    if run_asha_search and eval_fn is not None:
        logger.info("Stage 3: ASHA search within selected experts ...")
        asha_candidates = [
            ASHACandidate(
                model_name=model_name,
                expert_name=expert_name,
                weight=weight,
            )
            for model_name, expert_name, weight in selected
        ]
        asha_result = run_asha(
            asha_candidates,
            eval_fn,
            budget_seconds=asha_budget_seconds,
            seed=seed,
        )
        result.asha_result = asha_result
        result.final_model = asha_result.winner.model_name
        result.final_expert = asha_result.winner.expert_name

        # Build ensemble from top-3 if scores are close
        top3 = asha_result.ranking[:3]
        if len(top3) >= 2:
            scores = [c.best_score() for c in top3]
            if scores[0] > 0 and (scores[-1] - scores[0]) / max(abs(scores[0]), 1e-12) < 0.1:
                # Close scores → ensemble
                total_inv = sum(1.0 / max(s, 1e-12) for s in scores)
                result.ensemble = [
                    (c.model_name, (1.0 / max(c.best_score(), 1e-12)) / total_inv)
                    for c in top3
                ]
                logger.info(f"  Ensemble recommendation: {result.ensemble}")

        logger.info(
            f"  ASHA winner: {result.final_model} "
            f"(score={asha_result.winner.best_score():.4f})"
        )
    else:
        # Without ASHA, pick the top-weighted model
        if selected:
            result.final_model = selected[0][0]
            result.final_expert = selected[0][1]
        else:
            result.final_model = "LightGBM"
            result.final_expert = "tabular"

    result.total_time_seconds = time.monotonic() - t0

    # ---- Save audit artifacts ----
    if output_dir is not None:
        result.audit_path = _save_audit(result, Path(output_dir), mf)

    logger.info(f"AutoFit v2 complete in {result.total_time_seconds:.1f}s → {result.final_model}")
    return result


# ---------------------------------------------------------------------------
# Audit output
# ---------------------------------------------------------------------------

def _save_audit(
    result: AutoFitV2Result,
    output_dir: Path,
    mf: MetaFeaturesV2,
) -> Path:
    """Save full audit trail."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Meta-features report
    save_meta_features_report(mf, output_dir)

    # Full result JSON
    audit_path = output_dir / "autofit_v2_audit.json"
    audit_data = result.to_dict()
    audit_data["timestamp"] = datetime.now(timezone.utc).isoformat()
    audit_path.write_text(
        json.dumps(audit_data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # ASHA result
    if result.asha_result:
        save_asha_result(result.asha_result, output_dir)

    logger.info(f"Audit saved to {output_dir}")
    return audit_path


# ---------------------------------------------------------------------------
# Convenience: from FreezePointer
# ---------------------------------------------------------------------------

def autofit_v2_from_pointer(
    pointer_path: str = "docs/audits/FULL_SCALE_POINTER.yaml",
    target_col: str = "funding_raised_usd",
    task: str = "task1_outcome",
    output_dir: Optional[str] = None,
    **kwargs,
) -> AutoFitV2Result:
    """Load data via FreezePointer and run AutoFit v2."""
    import yaml
    import pyarrow.parquet as pq

    pointer = yaml.safe_load(Path(pointer_path).read_text(encoding="utf-8"))
    stamp = pointer["stamp"]
    core_dir = pointer["paths"]["offers_core_daily"]["dir"]

    logger.info(f"Loading panel from {core_dir} (stamp={stamp}) ...")
    df = pq.read_table(core_dir).to_pandas()
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} cols")

    out = Path(output_dir) if output_dir else None
    return autofit_v2(df, target_col=target_col, task=task, output_dir=out, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="AutoFit v2 pipeline")
    parser.add_argument("--pointer", default="docs/audits/FULL_SCALE_POINTER.yaml")
    parser.add_argument("--target", default="funding_raised_usd")
    parser.add_argument("--task", default="task1_outcome",
                        choices=["task1_outcome", "task2_forecast", "task3_risk_adjust"])
    parser.add_argument("--gating-mode", default="sparse",
                        choices=["hard", "soft", "sparse"])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-entities", type=int, default=500)
    args = parser.parse_args()

    result = autofit_v2_from_pointer(
        pointer_path=args.pointer,
        target_col=args.target,
        task=args.task,
        output_dir=args.output_dir,
        gating_mode=GatingMode(args.gating_mode),
        top_k=args.top_k,
        max_entities_sample=args.max_entities,
    )

    if args.output_dir is None:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(f"Done. Audit at {result.audit_path}")
