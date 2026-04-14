#!/usr/bin/env python3
"""
Run a resumable local champion loop for V740 on the shared 112-cell main-result surface.

This script is designed for fast V740 iteration, validation, and error correction.
It does NOT touch the canonical benchmark outputs. Instead, it:

1. derives the current shared non-seed 112-cell surface from all_results.csv,
2. identifies the incumbent benchmark champion model for each cell,
3. rebuilds the same freeze-backed local slice for that cell,
4. runs V740 and the incumbent champion on that identical local slice,
5. reports where V740 wins, ties, loses, fails, or degenerates.

The local numbers are only comparable within the same local slice. The purpose is
to create a fast, honest closed loop for V740 engineering, not to overwrite the
canonical benchmark line.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v740_alpha_minibenchmark import _build_case_frame, _make_temporal_config
from scripts.run_v740_alpha_smoke_slice import (
    _compute_metrics,
    _prepare_features,
    add_v740_funding_regime_args,
    add_v740_target_regime_args,
    v740_funding_regime_kwargs_from_args,
    v740_target_regime_kwargs_from_args,
)
from src.narrative.block3.models.nf_adaptive_champion import NFAdaptiveChampionV739
from src.narrative.block3.models.registry import check_model_available, get_model
from src.narrative.block3.models.single_model_mainline import SingleModelMainlineWrapper
from src.narrative.block3.models.v740_alpha import V740AlphaPrototypeWrapper
from src.narrative.block3.models.v740_variant_profiles import (
    apply_v740_variant_profile,
    is_local_v740_variant,
    resolve_requested_v740_variant,
)


NONSEED_ABLATIONS = ["core_only", "core_edgar", "core_text", "full"]
CELL_COLS = ["task", "ablation", "target", "horizon"]
TASK_ORDER = {"task1_outcome": 0, "task2_forecast": 1, "task3_risk_adjust": 2}
ABLATION_ORDER = {"core_only": 0, "core_edgar": 1, "core_text": 2, "full": 3}
TARGET_ORDER = {"is_funded": 0, "funding_raised_usd": 1, "investors_count": 2}
LOCAL_MAINLINE_ALIASES = {
    "single_model_mainline": {
        "variant": "mainline_alpha",
        "use_delegate": False,
        "wrapper_overrides": {},
    },
    "single_model_mainline_delegate": {
        "variant": "mainline_delegate_alpha",
        "use_delegate": True,
        "wrapper_overrides": {},
    },
    "single_model_mainline_track_legacy_baseline": {
        "variant": "mainline_alpha",
        "use_delegate": False,
        "wrapper_overrides": {
            "enable_investors_horizon_contract": False,
            "enable_count_hurdle_head": False,
            "enable_count_jump": False,
            "enable_count_sparsity_gate": False,
        },
    },
    "single_model_mainline_track_guarded_jump": {
        "variant": "mainline_alpha",
        "use_delegate": False,
        "wrapper_overrides": {
            "enable_count_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
            "enable_count_sparsity_gate": False,
        },
    },
    "single_model_mainline_track_guarded_jump_plus_sparsity": {
        "variant": "mainline_alpha",
        "use_delegate": False,
        "wrapper_overrides": {
            "enable_count_hurdle_head": True,
            "enable_count_jump": True,
            "count_jump_strength": 0.30,
            "enable_count_sparsity_gate": True,
            "count_sparsity_gate_strength": 0.75,
        },
    },
    "single_model_mainline_track_source_policy_transition_guard": {
        "variant": "mainline_alpha",
        "use_delegate": False,
        "wrapper_overrides": {
            "enable_investors_source_read_policy": True,
            "enable_investors_source_guard": True,
            "enable_investors_transition_correction": True,
        },
    },
}


def _positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return out


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--all-results-csv",
        type=Path,
        default=REPO_ROOT / "runs" / "benchmarks" / "block3_phase9_fair" / "all_results.csv",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "references" / "v740_shared112_champion_loop_20260401",
    )
    ap.add_argument(
        "--summary-md",
        type=Path,
        default=REPO_ROOT / "docs" / "references" / "V740_SHARED112_CHAMPION_LOOP_20260401.md",
    )
    ap.add_argument(
        "--surface-json",
        type=Path,
        default=REPO_ROOT / "docs" / "references" / "v740_shared112_surface_20260401.json",
    )
    ap.add_argument(
        "--models",
        default="v740_alpha,incumbent",
        help=(
            "Comma-separated models. Special tokens: v740_alpha, v741_lite, v742_unified, "
            "v743_factorized, v744_guarded_phase, v745_evidence_residual, single_model_mainline, "
            "single_model_mainline_delegate, single_model_mainline_track_legacy_baseline, "
            "single_model_mainline_track_guarded_jump, "
            "single_model_mainline_track_guarded_jump_plus_sparsity, "
            "single_model_mainline_track_source_policy_transition_guard, incumbent, v739"
        ),
    )
    ap.add_argument(
        "--profile",
        choices=["quick", "audit", "hard"],
        default="quick",
        help="Controls local slice size for faster iteration vs stricter audits.",
    )
    ap.add_argument("--task", choices=list(TASK_ORDER.keys()), default="")
    ap.add_argument("--ablation", choices=NONSEED_ABLATIONS, default="")
    ap.add_argument("--target", choices=list(TARGET_ORDER.keys()), default="")
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--case-substr", default="")
    ap.add_argument("--max-cells", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--dry-run-surface",
        action="store_true",
        help="Only materialize the shared-112 surface manifest; do not run any models.",
    )
    ap.add_argument("--tie-tolerance-pct", type=float, default=0.5)
    ap.add_argument(
        "--skip-scorecard",
        action="store_true",
        help="Do not generate the serious-branch scorecard draft after the loop finishes.",
    )
    ap.add_argument(
        "--scorecard-md",
        type=Path,
        default=None,
        help="Optional path for the auto-generated serious-branch scorecard markdown.",
    )
    ap.add_argument(
        "--scorecard-json",
        type=Path,
        default=None,
        help="Optional path for the auto-generated serious-branch scorecard JSON summary.",
    )
    ap.add_argument(
        "--scorecard-branch-name",
        default="",
        help="Optional branch name label passed through to the scorecard generator.",
    )
    ap.add_argument(
        "--scorecard-branch-type",
        default="cross_task",
        help="Branch type label passed through to the scorecard generator.",
    )

    ap.add_argument("--input-size", type=_positive_int, default=60)
    ap.add_argument("--hidden-dim", type=_positive_int, default=64)
    ap.add_argument("--max-epochs", type=_positive_int, default=3)
    ap.add_argument("--batch-size", type=_positive_int, default=128)
    ap.add_argument("--max-covariates", type=_positive_int, default=15)
    ap.add_argument("--max-windows", type=_positive_int, default=50000)
    ap.add_argument("--patience", type=_positive_int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable-teacher-distill", action="store_true")
    ap.add_argument("--disable-event-head", action="store_true")
    ap.add_argument("--disable-task-modulation", action="store_true")
    add_v740_funding_regime_args(ap)
    add_v740_target_regime_args(ap)
    return ap.parse_args()


def _default_scorecard_md(summary_md: Path) -> Path:
    if "CHAMPION_LOOP" in summary_md.name:
        return summary_md.with_name(summary_md.name.replace("CHAMPION_LOOP", "SCORECARD"))
    if "champion_loop" in summary_md.name:
        return summary_md.with_name(summary_md.name.replace("champion_loop", "scorecard"))
    return summary_md.with_name(summary_md.stem + "_scorecard" + summary_md.suffix)


def _normalize_local_model_token(token: str, args: argparse.Namespace) -> str:
    return resolve_requested_v740_variant(token.strip(), vars(args))


def _task_budget(task: str, target: str, horizon: int, profile: str) -> Tuple[int, int]:
    if profile == "quick":
        max_entities = 8 if target == "is_funded" else 12
        max_rows = 800 if target == "is_funded" else 1200
    elif profile == "audit":
        max_entities = 16 if target == "is_funded" else 24
        max_rows = 1600 if target == "is_funded" else 3000
    else:
        max_entities = 24 if target == "is_funded" else 32
        max_rows = 3200 if target == "is_funded" else 6000

    if task != "task1_outcome":
        max_entities = int(round(max_entities * 1.5))
        max_rows = int(round(max_rows * 1.5))
    if horizon >= 30:
        max_rows = int(round(max_rows * 1.35))
    if horizon >= 60:
        max_rows = int(round(max_rows * 1.25))
    return max_entities, max_rows


def _cell_name(row: Dict[str, Any]) -> str:
    return f"{row['task']}__{row['ablation']}__{row['target']}__h{int(row['horizon'])}"


def _sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        TASK_ORDER.get(str(row["task"]), 99),
        TARGET_ORDER.get(str(row["target"]), 99),
        int(row["horizon"]),
        ABLATION_ORDER.get(str(row["ablation"]), 99),
    )


def _load_shared112_surface(csv_path: Path, profile: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    df = df[df["ablation"].isin(NONSEED_ABLATIONS)].copy()
    df = df[df["mae"].notna()].copy()
    df["horizon"] = df["horizon"].astype(int)

    model_cell_counts = (
        df[CELL_COLS + ["model_name"]]
        .drop_duplicates()
        .groupby("model_name")
        .size()
        .sort_values(ascending=False)
    )
    max_cells = int(model_cell_counts.max())
    complete_models = sorted(model_cell_counts[model_cell_counts == max_cells].index.tolist())
    comparable = df[df["model_name"].isin(complete_models)].copy()

    shared_counts = (
        comparable[CELL_COLS + ["model_name"]]
        .drop_duplicates()
        .groupby(CELL_COLS)["model_name"]
        .nunique()
        .reset_index(name="n_models")
    )
    shared_cells = shared_counts[shared_counts["n_models"] == len(complete_models)].copy()
    comparable = comparable.merge(shared_cells[CELL_COLS], on=CELL_COLS, how="inner")

    cells: List[Dict[str, Any]] = []
    champion_dist = Counter()
    for cell_key, grp in comparable.groupby(CELL_COLS, sort=False):
        task, ablation, target, horizon = cell_key
        grp = grp.sort_values(["mae", "model_name"], ascending=[True, True]).reset_index(drop=True)
        champion_row = grp.iloc[0]
        runner_row = grp.iloc[1] if len(grp) > 1 else None
        max_entities, max_rows = _task_budget(str(task), str(target), int(horizon), profile)
        row = {
            "name": _cell_name({"task": task, "ablation": ablation, "target": target, "horizon": horizon}),
            "task": str(task),
            "ablation": str(ablation),
            "target": str(target),
            "horizon": int(horizon),
            "max_entities": int(max_entities),
            "max_rows": int(max_rows),
            "incumbent_model": str(champion_row["model_name"]),
            "incumbent_benchmark_mae": float(champion_row["mae"]),
            "runner_up_model": str(runner_row["model_name"]) if runner_row is not None else None,
            "runner_up_benchmark_mae": float(runner_row["mae"]) if runner_row is not None else None,
        }
        cells.append(row)
        champion_dist[row["incumbent_model"]] += 1

    cells.sort(key=_sort_key)
    return {
        "surface": "shared112_nonseed_main_result",
        "models": complete_models,
        "n_models": len(complete_models),
        "n_cells": len(cells),
        "champion_distribution": dict(sorted(champion_dist.items(), key=lambda kv: (-kv[1], kv[0]))),
        "cells": cells,
    }


def _instantiate_model(model_token: str, resolved_name: str, args: argparse.Namespace):
    if is_local_v740_variant(model_token):
        target_kwargs = apply_v740_variant_profile(model_token, v740_target_regime_kwargs_from_args(args))
        return V740AlphaPrototypeWrapper(
            input_size=args.input_size,
            hidden_dim=args.hidden_dim,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            max_covariates=args.max_covariates,
            max_entities=3000,
            max_windows=args.max_windows,
            patience=args.patience,
            enable_teacher_distill=not args.disable_teacher_distill,
            enable_event_head=not args.disable_event_head,
            enable_task_modulation=not args.disable_task_modulation,
            **v740_funding_regime_kwargs_from_args(args),
            **target_kwargs,
            seed=args.seed,
        )
    if model_token in LOCAL_MAINLINE_ALIASES:
        alias_cfg = LOCAL_MAINLINE_ALIASES[model_token]
        wrapper_kwargs = {
            "variant": alias_cfg["variant"],
            "use_delegate": alias_cfg["use_delegate"],
            "input_size": args.input_size,
            "hidden_dim": args.hidden_dim,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "max_covariates": args.max_covariates,
            "max_entities": 3000,
            "max_windows": args.max_windows,
            "patience": args.patience,
            "enable_teacher_distill": not args.disable_teacher_distill,
            "enable_event_head": not args.disable_event_head,
            "enable_task_modulation": not args.disable_task_modulation,
            "seed": args.seed,
        }
        wrapper_kwargs.update(dict(alias_cfg.get("wrapper_overrides", {})))
        return SingleModelMainlineWrapper(
            **wrapper_kwargs,
        )
    if model_token == "v739":
        return NFAdaptiveChampionV739(model_timeout=90)
    return get_model(resolved_name)


def _resolve_model_specs(raw_models: List[str], case: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for token in raw_models:
        token = token.strip()
        if not token:
            continue
        if token == "incumbent":
            resolved = case["incumbent_model"]
            label = f"incumbent__{resolved}"
        elif is_local_v740_variant(token):
            token = _normalize_local_model_token(token, args)
            resolved = token
            label = token
        elif token == "v739":
            resolved = "AutoFitV739"
            label = "v739"
        else:
            resolved = token
            label = token
        key = (token, resolved)
        if key in seen:
            continue
        seen.add(key)
        specs.append({"token": token, "resolved": resolved, "label": label})
    return specs


def _run_local_model(
    model_token: str,
    resolved_name: str,
    case: Dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    X_train, y_train = _prepare_features(train, case["target"])
    X_test, y_test = _prepare_features(test, case["target"])
    if len(X_train) < 10 or len(X_test) < 10:
        raise RuntimeError(
            f"Insufficient rows for {case['name']} / {resolved_name}: train={len(X_train)} test={len(X_test)}"
        )
    if (
        model_token != "v739"
        and not is_local_v740_variant(model_token)
        and model_token not in LOCAL_MAINLINE_ALIASES
        and not check_model_available(resolved_name)
    ):
        raise RuntimeError(f"Model {resolved_name} is not available in this environment")

    model = _instantiate_model(model_token, resolved_name, args)
    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        train_raw=train,
        val_raw=val,
        target=case["target"],
        task=case["task"],
        ablation=case["ablation"],
        horizon=case["horizon"],
    )
    fit_seconds = time.time() - t0

    t1 = time.time()
    preds = model.predict(
        X_test,
        test_raw=test,
        target=case["target"],
        task=case["task"],
        ablation=case["ablation"],
        horizon=case["horizon"],
    )
    pred_seconds = time.time() - t1

    preds = np.asarray(preds, dtype=np.float64)
    metrics = _compute_metrics(y_test.to_numpy(dtype=np.float64), preds)
    pred_std = float(np.nanstd(preds)) if len(preds) else 0.0
    constant_prediction = bool(len(preds) > 1 and pred_std < 1e-8)
    selected_model = None
    routing = None
    if hasattr(model, "_routing_info"):
        routing = getattr(model, "_routing_info")
        if isinstance(routing, dict):
            selected_model = routing.get("selected_model")
    regime_info = model.get_regime_info() if hasattr(model, "get_regime_info") else None

    return {
        "metrics": metrics,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(pred_seconds),
        "prediction_std": pred_std,
        "constant_prediction": constant_prediction,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_matrix_rows": int(len(X_train)),
        "test_matrix_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "selected_model": selected_model,
        "routing_info": routing,
        "funding_log_domain": bool(getattr(model, "_funding_log_domain", False)),
        "funding_source_scaling": bool(getattr(model, "_funding_source_scaling", False)),
        "funding_anchor_enabled": bool(getattr(model, "_funding_anchor_enabled", False)),
        "funding_anchor_strength": float(getattr(model, "_effective_funding_anchor_strength", 0.0)),
        "target_routing_enabled": bool(getattr(model, "enable_target_routing", False)),
        "target_route_experts": int(getattr(model, "target_route_experts", 0)),
        "count_anchor_enabled": bool(getattr(model, "enable_count_anchor", False)),
        "count_anchor_strength": float(getattr(model, "count_anchor_strength", 0.0)),
        "count_jump_enabled": bool(getattr(model, "enable_count_jump", False)),
        "count_jump_strength": float(getattr(model, "count_jump_strength", 0.0)),
        "count_sparsity_gate_enabled": bool(getattr(model, "enable_count_sparsity_gate", False)),
        "count_sparsity_gate_strength": float(getattr(model, "count_sparsity_gate_strength", 0.0)),
        "count_source_routing_enabled": bool(getattr(model, "enable_count_source_routing", False)),
        "count_route_experts": int(getattr(model, "count_route_experts", 0)),
        "count_route_floor": float(getattr(model, "count_route_floor", 0.0)),
        "count_route_entropy_strength": float(getattr(model, "count_route_entropy_strength", 0.0)),
        "count_active_loss_strength": float(getattr(model, "count_active_loss_strength", 0.0)),
        "regime_info": regime_info,
    }


def _pick_primary_candidate_label(results: List[Dict[str, Any]]) -> str:
    labels: List[str] = []
    seen = set()
    for row in results:
        label = str(row.get("model_label", "")).strip()
        if not label or label.startswith("incumbent__") or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    for preferred in (
        "single_model_mainline",
        "single_model_mainline_delegate",
        "v745_evidence_residual",
        "v744_guarded_phase",
        "v743_factorized",
        "v742_unified",
        "v741_lite",
        "v740_alpha",
        "v739",
    ):
        if preferred in seen:
            return preferred
    return labels[0] if labels else "v740_alpha"


def _classify_case(case_rows: Dict[str, Dict[str, Any]], tie_tol_pct: float, candidate_label: str) -> str:
    candidate = case_rows.get(candidate_label)
    incumbent_key = next((k for k in case_rows if k.startswith("incumbent__")), None)
    incumbent = case_rows.get(incumbent_key) if incumbent_key else None
    if not candidate or not incumbent:
        return "incomplete"
    if candidate.get("error") or incumbent.get("error"):
        return "error"
    candidate_mae = candidate.get("metrics", {}).get("mae")
    inc_mae = incumbent.get("metrics", {}).get("mae")
    if candidate_mae is None or inc_mae is None:
        return "error"
    denom = max(abs(float(inc_mae)), 1e-9)
    rel = abs(float(candidate_mae) - float(inc_mae)) / denom * 100.0
    if rel <= tie_tol_pct:
        return "tie"
    return "win" if float(candidate_mae) < float(inc_mae) else "loss"


def _display_candidate_name(candidate_label: str) -> str:
    if candidate_label == "single_model_mainline":
        return "SingleModelMainline"
    if candidate_label == "single_model_mainline_delegate":
        return "SingleModelMainline-Delegate"
    if candidate_label == "v745_evidence_residual":
        return "V745-EvidenceResidual"
    if candidate_label == "v744_guarded_phase":
        return "V744-GuardedPhase"
    if candidate_label == "v743_factorized":
        return "V743-Factorized"
    if candidate_label == "v742_unified":
        return "V742-Unified"
    if candidate_label == "v741_lite":
        return "V741-Lite"
    if candidate_label == "v740_alpha":
        return "V740-alpha"
    if candidate_label == "v739":
        return "V739"
    return candidate_label


def _write_summary(path: Path, manifest: Dict[str, Any], results: List[Dict[str, Any]], tie_tol_pct: float) -> None:
    by_case: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in results:
        by_case[row["case_name"]][row["model_label"]] = row
    candidate_label = _pick_primary_candidate_label(results)
    candidate_display = _display_candidate_name(candidate_label)

    outcome_counts = Counter()
    gap_rows: List[Dict[str, Any]] = []
    task_summary: Dict[str, Counter] = defaultdict(Counter)
    target_summary: Dict[str, Counter] = defaultdict(Counter)
    ablation_summary: Dict[str, Counter] = defaultdict(Counter)
    failures = 0
    constants = 0
    total_v740 = 0

    for case in manifest["cells"]:
        case_rows = by_case.get(case["name"], {})
        outcome = _classify_case(case_rows, tie_tol_pct, candidate_label)
        outcome_counts[outcome] += 1
        task_summary[case["task"]][outcome] += 1
        target_summary[case["target"]][outcome] += 1
        ablation_summary[case["ablation"]][outcome] += 1

        candidate = case_rows.get(candidate_label)
        inc_key = f"incumbent__{case['incumbent_model']}"
        incumbent = case_rows.get(inc_key)
        if candidate:
            total_v740 += 1
            if candidate.get("error"):
                failures += 1
            if candidate.get("constant_prediction"):
                constants += 1
        if candidate and incumbent and not candidate.get("error") and not incumbent.get("error"):
            candidate_mae = candidate.get("metrics", {}).get("mae")
            inc_mae = incumbent.get("metrics", {}).get("mae")
            if candidate_mae is not None and inc_mae is not None:
                gap_pct = (float(candidate_mae) / max(float(inc_mae), 1e-9) - 1.0) * 100.0
                gap_rows.append({
                    "case": case["name"],
                    "task": case["task"],
                    "ablation": case["ablation"],
                    "target": case["target"],
                    "horizon": case["horizon"],
                    "incumbent_model": case["incumbent_model"],
                    "candidate_mae": float(candidate_mae),
                    "incumbent_mae": float(inc_mae),
                    "gap_pct": float(gap_pct),
                    "outcome": outcome,
                })

    worst_losses = sorted(
        [r for r in gap_rows if r["outcome"] == "loss"],
        key=lambda r: (-r["gap_pct"], r["case"]),
    )[:20]

    lines = [
        "# Shared-112 Champion Loop",
        "",
        "Generated by `scripts/run_v740_shared112_champion_loop.py`.",
        "",
        "This report is a fast local engineering loop, not a canonical benchmark landing.",
        f"Each row compares {candidate_display} and the incumbent benchmark champion on the **same local freeze-backed slice**.",
        "",
        "## Surface",
        "",
        f"- Shared surface: `{manifest['n_models']} models / {manifest['n_cells']} cells`",
        f"- Primary candidate: `{candidate_label}`",
        f"- Tie tolerance: `{tie_tol_pct:.3f}%` relative MAE difference",
        f"- {candidate_display} rows attempted: `{total_v740}`",
        f"- {candidate_display} failures: `{failures}`",
        f"- {candidate_display} constant predictions: `{constants}`",
        "",
        f"## {candidate_display} vs Incumbent",
        "",
        f"- Wins: `{outcome_counts.get('win', 0)}`",
        f"- Ties: `{outcome_counts.get('tie', 0)}`",
        f"- Losses: `{outcome_counts.get('loss', 0)}`",
        f"- Incomplete: `{outcome_counts.get('incomplete', 0)}`",
        f"- Errors: `{outcome_counts.get('error', 0)}`",
        "",
        "## Champion Distribution On Shared-112 Surface",
        "",
    ]
    for model_name, count in manifest["champion_distribution"].items():
        lines.append(f"- `{model_name}`: `{count}` cells")

    lines.extend(["", "## Outcome By Task", ""])
    for key in sorted(task_summary, key=lambda x: TASK_ORDER.get(x, 99)):
        c = task_summary[key]
        lines.append(
            f"- `{key}`: win={c.get('win',0)} tie={c.get('tie',0)} loss={c.get('loss',0)} error={c.get('error',0)} incomplete={c.get('incomplete',0)}"
        )

    lines.extend(["", "## Outcome By Target", ""])
    for key in sorted(target_summary, key=lambda x: TARGET_ORDER.get(x, 99)):
        c = target_summary[key]
        lines.append(
            f"- `{key}`: win={c.get('win',0)} tie={c.get('tie',0)} loss={c.get('loss',0)} error={c.get('error',0)} incomplete={c.get('incomplete',0)}"
        )

    lines.extend(["", "## Outcome By Ablation", ""])
    for key in sorted(ablation_summary, key=lambda x: ABLATION_ORDER.get(x, 99)):
        c = ablation_summary[key]
        lines.append(
            f"- `{key}`: win={c.get('win',0)} tie={c.get('tie',0)} loss={c.get('loss',0)} error={c.get('error',0)} incomplete={c.get('incomplete',0)}"
        )

    lines.extend([
        "",
        "## Worst Local Losses",
        "",
        f"| Case | Incumbent | {candidate_display} MAE | Incumbent MAE | Gap % |",
        "|---|---|---:|---:|---:|",
    ])
    for row in worst_losses:
        lines.append(
            f"| {row['case']} | {row['incumbent_model']} | {row['candidate_mae']:.4f} | {row['incumbent_mae']:.4f} | {row['gap_pct']:.2f} |"
        )

    lines.extend([
        "",
        "## Per-Cell Summary",
        "",
        f"| Case | Incumbent | Benchmark MAE | {candidate_display} Local MAE | Incumbent Local MAE | Outcome | Const | Error |",
        "|---|---|---:|---:|---:|---|---|---|",
    ])
    for case in manifest["cells"]:
        case_rows = by_case.get(case["name"], {})
        candidate = case_rows.get(candidate_label, {})
        incumbent = case_rows.get(f"incumbent__{case['incumbent_model']}", {})
        lines.append(
            "| {case_name} | {incumbent_name} | {bench_mae} | {v740_mae} | {inc_mae} | {outcome} | {const} | {error} |".format(
                case_name=case["name"],
                incumbent_name=case["incumbent_model"],
                bench_mae=_fmt(case.get("incumbent_benchmark_mae")),
                v740_mae=_fmt(candidate.get("metrics", {}).get("mae") if isinstance(candidate.get("metrics"), dict) else None),
                inc_mae=_fmt(incumbent.get("metrics", {}).get("mae") if isinstance(incumbent.get("metrics"), dict) else None),
                outcome=_classify_case(case_rows, tie_tol_pct, candidate_label),
                const=candidate.get("constant_prediction", "-"),
                error=(candidate.get("error") or incumbent.get("error") or "-")[:120],
            )
        )

    lines.extend(["", "## Raw JSON Artifacts", ""])
    for row in results:
        lines.append(f"- `{row['case_name']} / {row['model_label']}`: `{row['json_path']}`")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_scorecard_builder(
    args: argparse.Namespace,
    results_dir: Path,
    surface_json: Path,
    candidate_label: str,
) -> Path:
    scorecard_md = args.scorecard_md or _default_scorecard_md(args.summary_md)
    scorecard_json = args.scorecard_json or scorecard_md.with_suffix(".json")
    scorecard_md.parent.mkdir(parents=True, exist_ok=True)
    scorecard_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_shared112_scorecard.py"),
        "--results-dir",
        str(results_dir),
        "--surface-json",
        str(surface_json),
        "--output-md",
        str(scorecard_md),
        "--output-json",
        str(scorecard_json),
        "--candidate-label",
        candidate_label,
        "--branch-type",
        args.scorecard_branch_type,
        "--tie-tolerance-pct",
        str(args.tie_tolerance_pct),
    ]
    branch_name = args.scorecard_branch_name.strip() or args.output_dir.name
    if branch_name:
        cmd.extend(["--branch-name", branch_name])

    subprocess.run(cmd, check=True)
    return scorecard_md


def main() -> int:
    args = _parse_args()
    raw_models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.surface_json.parent.mkdir(parents=True, exist_ok=True)

    manifest = _load_shared112_surface(args.all_results_csv, args.profile)
    surface_payload = json.dumps(manifest, indent=2, sort_keys=True)
    args.surface_json.write_text(surface_payload, encoding="utf-8")
    local_surface_json = args.output_dir / "surface.json"
    if local_surface_json != args.surface_json:
        local_surface_json.write_text(surface_payload, encoding="utf-8")
    print(
        f"[v740-shared112] surface ready: {manifest['n_models']} models / {manifest['n_cells']} cells",
        flush=True,
    )
    if args.dry_run_surface:
        print(f"[v740-shared112] wrote surface manifest to {args.surface_json}", flush=True)
        return 0

    cases = list(manifest["cells"])
    if args.task:
        cases = [c for c in cases if c["task"] == args.task]
    if args.ablation:
        cases = [c for c in cases if c["ablation"] == args.ablation]
    if args.target:
        cases = [c for c in cases if c["target"] == args.target]
    if args.horizon:
        cases = [c for c in cases if int(c["horizon"]) == int(args.horizon)]
    if args.case_substr:
        cases = [c for c in cases if args.case_substr in c["name"]]
    if args.max_cells and args.max_cells > 0:
        cases = cases[: args.max_cells]
    if not cases:
        raise SystemExit("No shared-112 cells selected.")

    temporal_config = _make_temporal_config()
    results: List[Dict[str, Any]] = []
    for case in cases:
        print(
            f"[v740-shared112] preparing {case['name']} ({case['incumbent_model']} is incumbent)",
            flush=True,
        )
        train, val, test = _build_case_frame(case, temporal_config)
        model_specs = _resolve_model_specs(raw_models, case, args)
        for spec in model_specs:
            model_token = spec["token"]
            resolved = spec["resolved"]
            label = spec["label"]
            out_path = args.output_dir / f"{_slugify(case['name'])}__{_slugify(label)}.json"
            if args.skip_existing and out_path.exists():
                try:
                    row = json.loads(out_path.read_text(encoding="utf-8"))
                    row["json_path"] = str(out_path)
                    out_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
                    results.append(row)
                    print(f"[v740-shared112] skip-existing {case['name']} / {label}", flush=True)
                    continue
                except Exception as exc:
                    print(
                        f"[v740-shared112] skip-existing read failed for {case['name']} / {label}: {exc}; recomputing",
                        flush=True,
                    )
            row: Dict[str, Any] = {
                "case_name": case["name"],
                "task": case["task"],
                "ablation": case["ablation"],
                "target": case["target"],
                "horizon": int(case["horizon"]),
                "incumbent_model": case["incumbent_model"],
                "incumbent_benchmark_mae": float(case["incumbent_benchmark_mae"]),
                "runner_up_model": case.get("runner_up_model"),
                "runner_up_benchmark_mae": case.get("runner_up_benchmark_mae"),
                "model_token": model_token,
                "model_label": label,
                "resolved_model_name": resolved,
                "profile": args.profile,
                "json_path": str(out_path),
            }
            try:
                print(f"[v740-shared112] running {case['name']} / {label}", flush=True)
                summary = _run_local_model(model_token, resolved, case, train, val, test, args)
                row.update(summary)
                print(
                    f"[v740-shared112] finished {case['name']} / {label} mae={summary.get('metrics', {}).get('mae')}",
                    flush=True,
                )
            except Exception as exc:
                row["error"] = str(exc)
                print(f"[v740-shared112] failed {case['name']} / {label}: {exc}", flush=True)
            out_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
            results.append(row)

    _write_summary(args.summary_md, manifest, results, args.tie_tolerance_pct)
    print(f"[v740-shared112] wrote summary to {args.summary_md}", flush=True)
    if not args.skip_scorecard:
        candidate_label = _pick_primary_candidate_label(results)
        scorecard_md = _run_scorecard_builder(args, args.output_dir, local_surface_json, candidate_label)
        print(f"[v740-shared112] wrote scorecard to {scorecard_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())