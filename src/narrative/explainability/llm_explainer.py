"""
LLM-based explainer for generating natural language reports.

This module provides utilities for generating LLM-based explanation reports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union


def build_llm_report(
    predictions: Optional[Dict[str, Any]] = None,
    explanations: Optional[Dict[str, Any]] = None,
    concepts: Optional[Union[List[str], Dict[str, float]]] = None,
    template: Optional[str] = None,
    shap_summary: Optional[List[Tuple[str, float]]] = None,
    ig_summary: Optional[List[Tuple[str, float]]] = None,
) -> str:
    """
    Build an LLM-ready report from predictions and explanations.
    
    Args:
        predictions: Model predictions
        explanations: Explanation data (SHAP, LIME, etc.)
        concepts: Optional concept names/scores to focus on
        template: Optional report template
        shap_summary: Top SHAP features [(name, importance), ...]
        ig_summary: Top IG features [(name, score), ...]
        
    Returns:
        Natural language report string
    """
    report_parts = []
    
    report_parts.append("# Prediction Explanation Report\n")
    
    if predictions and "prediction" in predictions:
        report_parts.append(f"**Predicted Value**: {predictions['prediction']:.4f}\n")
    
    if predictions and "confidence" in predictions:
        report_parts.append(f"**Confidence**: {predictions['confidence']:.2%}\n")
    
    # SHAP summary
    if shap_summary:
        report_parts.append("\n## SHAP Feature Importance\n")
        for i, (feat, importance) in enumerate(shap_summary[:5], 1):
            report_parts.append(f"{i}. **{feat}**: {importance:.4f}\n")
    
    # IG summary
    if ig_summary:
        report_parts.append("\n## Integrated Gradients Attribution\n")
        for i, (feat, score) in enumerate(ig_summary[:5], 1):
            report_parts.append(f"{i}. **{feat}**: {score:.4f}\n")
    
    # Legacy format
    if explanations and "top_features" in explanations:
        report_parts.append("\n## Key Factors\n")
        for i, (feat, importance) in enumerate(explanations["top_features"][:5], 1):
            report_parts.append(f"{i}. **{feat}**: {importance:.4f}\n")
    
    if concepts:
        report_parts.append("\n## Concept Analysis\n")
        if isinstance(concepts, dict):
            for concept, score in concepts.items():
                report_parts.append(f"- **{concept}**: {score:.4f}\n")
        else:
            for concept in concepts:
                report_parts.append(f"- {concept}\n")
    
    return "".join(report_parts)


__all__ = [
    "build_llm_report",
]
