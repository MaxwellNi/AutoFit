#!/usr/bin/env python3
"""
AutoFit v2 — ASHA Budget Search (Section B / D6).

Implements Asynchronous Successive Halving (ASHA) for model selection
within each expert category.

Key differences from v1 (budget_search.py):
    1. Operates on *real* Block3Dataset splits (not random train/val).
    2. Uses the unified temporal split from unified_protocol.py.
    3. Rung = fraction of *entities* (not epochs) for cross-entity stability.
    4. Budget is measured in wall-clock seconds.
    5. Full audit trail with JSON output.

Protocol:
    Rung 0: 10% of entities → train each candidate, score on val subset
    Rung 1: 30% of entities → top ⌈N/η⌉ survive
    Rung 2: 100% of entities → top ⌈N/η²⌉ survive
    → Final winner (or top-K for ensemble).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ASHACandidate:
    """A candidate model being evaluated in ASHA."""
    model_name: str
    expert_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0       # expert routing weight

    # Results per rung
    rung_scores: List[float] = field(default_factory=list)
    rung_times: List[float] = field(default_factory=list)
    rungs_completed: int = 0
    eliminated_at_rung: int = -1
    error: Optional[str] = None

    def best_score(self) -> float:
        """Best (lowest) validation score across rungs."""
        return min(self.rung_scores) if self.rung_scores else float("inf")

    def latest_score(self) -> float:
        return self.rung_scores[-1] if self.rung_scores else float("inf")


@dataclass
class ASHAResult:
    """Complete ASHA search result with audit trail."""
    winner: ASHACandidate
    ranking: List[ASHACandidate]
    total_time_seconds: float
    n_rungs_completed: int
    n_candidates_initial: int
    n_candidates_final: int
    audit: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner.model_name,
            "winner_expert": self.winner.expert_name,
            "winner_score": self.winner.best_score(),
            "total_time_seconds": self.total_time_seconds,
            "n_rungs_completed": self.n_rungs_completed,
            "n_candidates_initial": self.n_candidates_initial,
            "n_candidates_final": self.n_candidates_final,
            "ranking": [
                {
                    "model": c.model_name,
                    "expert": c.expert_name,
                    "best_score": c.best_score(),
                    "rungs": c.rungs_completed,
                    "eliminated_at": c.eliminated_at_rung,
                    "error": c.error,
                }
                for c in self.ranking
            ],
            "audit": self.audit,
        }


# ---------------------------------------------------------------------------
# Rung configuration
# ---------------------------------------------------------------------------

DEFAULT_RUNGS = [0.10, 0.30, 1.0]    # entity fractions
DEFAULT_ETA = 3                       # halving factor


def run_asha(
    candidates: List[ASHACandidate],
    eval_fn: Callable[[str, Dict[str, Any], float], float],
    *,
    rungs: Optional[List[float]] = None,
    eta: int = DEFAULT_ETA,
    budget_seconds: float = 3600.0,
    seed: int = 42,
) -> ASHAResult:
    """
    Run ASHA over a list of candidates.

    Parameters
    ----------
    candidates : list of ASHACandidate
        Models to evaluate.
    eval_fn : callable
        ``eval_fn(model_name, config, entity_fraction) -> float``
        Returns validation metric (lower = better).
    rungs : list of float
        Entity fraction at each rung (ascending).
    eta : int
        Halving factor.
    budget_seconds : float
        Wall-clock budget.
    seed : int
        Reproducibility seed.

    Returns
    -------
    ASHAResult
    """
    rungs = rungs or list(DEFAULT_RUNGS)
    audit: List[Dict[str, Any]] = []
    active = list(candidates)
    total_time = 0.0
    t0 = time.monotonic()

    rng = np.random.RandomState(seed)

    for rung_idx, frac in enumerate(rungs):
        if not active:
            break

        rung_entry = {
            "rung": rung_idx,
            "entity_fraction": frac,
            "n_active": len(active),
            "candidates": [c.model_name for c in active],
            "results": [],
        }

        for cand in active:
            if time.monotonic() - t0 > budget_seconds:
                audit.append({"event": "budget_exceeded", "rung": rung_idx})
                break

            try:
                t_start = time.monotonic()
                score = eval_fn(cand.model_name, cand.config, frac)
                elapsed = time.monotonic() - t_start

                cand.rung_scores.append(score)
                cand.rung_times.append(elapsed)
                cand.rungs_completed = rung_idx + 1
                total_time += elapsed

                rung_entry["results"].append({
                    "model": cand.model_name,
                    "score": score,
                    "time": elapsed,
                })

            except Exception as e:
                cand.rung_scores.append(float("inf"))
                cand.error = str(e)
                rung_entry["results"].append({
                    "model": cand.model_name,
                    "error": str(e),
                })

        audit.append(rung_entry)

        # Halve: keep top ⌈N/η⌉
        active.sort(key=lambda c: c.latest_score())
        n_keep = max(1, int(np.ceil(len(active) / eta)))

        if rung_idx < len(rungs) - 1:
            eliminated = active[n_keep:]
            for c in eliminated:
                c.eliminated_at_rung = rung_idx
            active = active[:n_keep]

            audit.append({
                "event": "halving",
                "rung": rung_idx,
                "kept": [c.model_name for c in active],
                "eliminated": [c.model_name for c in eliminated],
            })

    # Final ranking
    all_cands = list(candidates)
    all_cands.sort(key=lambda c: c.best_score())
    winner = all_cands[0]

    return ASHAResult(
        winner=winner,
        ranking=all_cands,
        total_time_seconds=time.monotonic() - t0,
        n_rungs_completed=max((c.rungs_completed for c in candidates), default=0),
        n_candidates_initial=len(candidates),
        n_candidates_final=len(active),
        audit=audit,
    )


def save_asha_result(result: ASHAResult, output_dir: Path) -> Path:
    """Save ASHA result as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "asha_result.json"
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path
