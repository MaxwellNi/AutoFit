from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from narrative.explainability.shap_utils import explain_with_shap
from narrative.explainability.tcav_utils import tcav_scores
from narrative.explainability.integrated_gradients import integrated_gradients
from narrative.explainability.lime_utils import explain_with_lime
from narrative.explainability.llm_explainer import build_llm_report

DEFAULT_CONCEPTS = [
    "tone_optimism",
    "risk_disclosure",
    "hype_vs_fundamentals",
    "social_proof_fomo",
    "numbers_vs_story",
    "professionalization",
    "esg_impact_focus",
    "genai_likelihood",
]


def _rank_features(df: pd.DataFrame, feature_cols: List[str], target_col: Optional[str]) -> List[Tuple[str, float]]:
    feats = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if target_col and target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors="coerce")
        scores = []
        for col in feature_cols:
            x = pd.to_numeric(feats[col], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 5:
                scores.append(0.0)
            else:
                scores.append(abs(np.corrcoef(x[valid], y[valid])[0, 1]))
        ranked = sorted(zip(feature_cols, scores), key=lambda x: -x[1])
    else:
        scores = feats.abs().mean(axis=0).tolist()
        ranked = sorted(zip(feature_cols, scores), key=lambda x: -x[1])
    return ranked


def _proxy_predict(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    feats = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats.sum(axis=1).to_numpy(dtype=float)


def _faithfulness_tests(
    df: pd.DataFrame,
    ranked_features: List[Tuple[str, float]],
    *,
    top_k: int = 10,
) -> Dict[str, float]:
    feature_cols = [f for f, _ in ranked_features[:top_k]]
    if not feature_cols:
        return {"deletion_drop": 0.0, "insertion_gain": 0.0, "counterfactual_shift": 0.0}

    base = _proxy_predict(df, feature_cols)
    base_mean = float(np.mean(np.abs(base)))

    deleted_df = df.copy()
    deleted_df[feature_cols] = 0.0
    deleted = _proxy_predict(deleted_df, feature_cols)
    deletion_drop = base_mean - float(np.mean(np.abs(deleted)))

    inserted_df = df.copy()
    for col in df.columns:
        if col not in feature_cols:
            inserted_df[col] = 0.0
    inserted = _proxy_predict(inserted_df, feature_cols)
    insertion_gain = float(np.mean(np.abs(inserted))) - 0.0

    counter_df = df.copy()
    counter_df[feature_cols] = -counter_df[feature_cols]
    counter = _proxy_predict(counter_df, feature_cols)
    counterfactual_shift = float(np.mean(np.abs(counter - base)))

    return {
        "deletion_drop": float(deletion_drop),
        "insertion_gain": float(insertion_gain),
        "counterfactual_shift": float(counterfactual_shift),
        "top_k": int(top_k),
    }


def _rank_features(df: pd.DataFrame, feature_cols: List[str], target_col: Optional[str]) -> List[Tuple[str, float]]:
    feats = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if target_col and target_col in df.columns:
        y = pd.to_numeric(df[target_col], errors="coerce")
        scores = []
        for col in feature_cols:
            x = pd.to_numeric(feats[col], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 5:
                scores.append(0.0)
            else:
                scores.append(abs(np.corrcoef(x[valid], y[valid])[0, 1]))
        ranked = sorted(zip(feature_cols, scores), key=lambda x: -x[1])
    else:
        scores = feats.abs().mean(axis=0).tolist()
        ranked = sorted(zip(feature_cols, scores), key=lambda x: -x[1])
    return ranked


def _proxy_predict(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    feats = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats.sum(axis=1).to_numpy(dtype=float)


def _faithfulness_tests(
    df: pd.DataFrame,
    ranked_features: List[Tuple[str, float]],
    *,
    top_k: int = 10,
) -> Dict[str, float]:
    feature_cols = [f for f, _ in ranked_features[:top_k]]
    if not feature_cols:
        return {"deletion_drop": 0.0, "insertion_gain": 0.0, "counterfactual_shift": 0.0}

    base = _proxy_predict(df, feature_cols)
    base_mean = float(np.mean(np.abs(base)))

    deleted_df = df.copy()
    deleted_df[feature_cols] = 0.0
    deleted = _proxy_predict(deleted_df, feature_cols)
    deletion_drop = base_mean - float(np.mean(np.abs(deleted)))

    inserted_df = df.copy()
    for col in df.columns:
        if col not in feature_cols:
            inserted_df[col] = 0.0
    inserted = _proxy_predict(inserted_df, feature_cols)
    insertion_gain = float(np.mean(np.abs(inserted))) - 0.0

    counter_df = df.copy()
    counter_df[feature_cols] = -counter_df[feature_cols]
    counter = _proxy_predict(counter_df, feature_cols)
    counterfactual_shift = float(np.mean(np.abs(counter - base)))

    return {
        "deletion_drop": float(deletion_drop),
        "insertion_gain": float(insertion_gain),
        "counterfactual_shift": float(counterfactual_shift),
        "top_k": int(top_k),
    }


def export_explainability(
    run_dir: Path,
    *,
    df: Optional[pd.DataFrame] = None,
    target_col: Optional[str] = None,
) -> Path:
    """
    Export explainability artifacts to runs/<exp>/explain/.
    """
    explain_dir = Path(run_dir) / "explain"
    explain_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        offers_path = Path(run_dir) / "offers_model.parquet"
        if offers_path.exists():
            df = pd.read_parquet(offers_path)
        else:
            df = pd.DataFrame()

    feature_cols = []
    if not df.empty:
        drop_cols = {"platform_name", "offer_id", "hash_id", "link"}
        if target_col:
            drop_cols.add(target_col)
        numeric = df.select_dtypes(include=["number", "bool"]).columns
        feature_cols = [c for c in numeric if c not in drop_cols]

    shap_out = explain_with_shap(df, feature_cols, target_col=target_col)
    tcav_out = {"concept_scores": tcav_scores(DEFAULT_CONCEPTS), "method": "proxy_tcav"}
    ig_out = integrated_gradients(df, feature_cols)
    lime_out = explain_with_lime(df, feature_cols)

    ranked = _rank_features(df, feature_cols, target_col)
    faithfulness = _faithfulness_tests(df, ranked, top_k=10)

    concept_cols = [c for c in feature_cols if c in DEFAULT_CONCEPTS]
    time_cols = [c for c in feature_cols if "time" in c or "date" in c or "day" in c or "week" in c or "month" in c or "year" in c]
    exog_cols = [c for c in feature_cols if c.startswith("edgar_") or c.startswith("text_")]

    attributions = {
        "concept": _rank_features(df, concept_cols, target_col)[:10],
        "time": _rank_features(df, time_cols, target_col)[:10],
        "exogenous": _rank_features(df, exog_cols, target_col)[:10],
    }

    ranked = _rank_features(df, feature_cols, target_col)
    faithfulness = _faithfulness_tests(df, ranked, top_k=10)

    concept_cols = [c for c in feature_cols if c in DEFAULT_CONCEPTS]
    time_cols = [c for c in feature_cols if "time" in c or "date" in c or "day" in c or "week" in c or "month" in c or "year" in c]
    exog_cols = [c for c in feature_cols if c.startswith("edgar_") or c.startswith("text_")]

    attributions = {
        "concept": _rank_features(df, concept_cols, target_col)[:10],
        "time": _rank_features(df, time_cols, target_col)[:10],
        "exogenous": _rank_features(df, exog_cols, target_col)[:10],
    }

    shap_path = explain_dir / "shap_global.json"
    tcav_path = explain_dir / "tcav_concepts.json"
    ig_path = explain_dir / "ig_highlights.json"
    lime_path = explain_dir / "lime_cases.json"
    llm_path = explain_dir / "llm_report.md"
    faith_path = explain_dir / "faithfulness.json"
    attr_path = explain_dir / "attributions.json"
    faith_path = explain_dir / "faithfulness.json"
    attr_path = explain_dir / "attributions.json"

    shap_path.write_text(json.dumps(shap_out, indent=2), encoding="utf-8")
    tcav_path.write_text(json.dumps(tcav_out, indent=2), encoding="utf-8")
    ig_path.write_text(json.dumps(ig_out, indent=2), encoding="utf-8")
    lime_path.write_text(json.dumps(lime_out, indent=2), encoding="utf-8")
    faith_path.write_text(json.dumps(faithfulness, indent=2), encoding="utf-8")
    attr_path.write_text(json.dumps(attributions, indent=2), encoding="utf-8")
    faith_path.write_text(json.dumps(faithfulness, indent=2), encoding="utf-8")
    attr_path.write_text(json.dumps(attributions, indent=2), encoding="utf-8")

    shap_summary = [(d["feature"], d["importance"]) for d in shap_out.get("feature_importance", [])]
    ig_summary = [(d["feature"], d["score"]) for d in ig_out.get("attributions", [])]
    llm_report = build_llm_report(
        concepts=tcav_out["concept_scores"],
        shap_summary=shap_summary[:5],
        ig_summary=ig_summary[:5],
    )
    llm_path.write_text(llm_report, encoding="utf-8")

    return explain_dir


__all__ = ["export_explainability"]
