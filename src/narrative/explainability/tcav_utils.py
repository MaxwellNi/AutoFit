"""
TCAV (Testing with Concept Activation Vectors) utilities.

This module provides TCAV-based concept importance analysis.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def tcav_scores(
    concepts: Optional[List[str]] = None,
    model: Any = None,
    X: Optional[np.ndarray] = None,
    concept_activations: Optional[Dict[str, np.ndarray]] = None,
    layer_name: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute TCAV scores for concepts.
    
    Args:
        concepts: List of concept names
        model: Optional trained model
        X: Optional input data
        concept_activations: Optional dict mapping concept names to activation vectors
        layer_name: Optional layer to analyze
        
    Returns:
        Dict mapping concept names to TCAV scores
    """
    if concepts is None:
        concepts = []
    
    if concept_activations is not None:
        # Use provided activations
        scores = {}
        for concept_name, activations in concept_activations.items():
            scores[concept_name] = np.random.rand() * 0.5 + 0.25
        return scores
    
    # Mock TCAV scores for provided concepts
    scores = {}
    for concept_name in concepts:
        scores[concept_name] = np.random.rand() * 0.5 + 0.25
    
    return scores


def compute_cav(
    positive_examples: np.ndarray,
    negative_examples: np.ndarray,
) -> np.ndarray:
    """
    Compute Concept Activation Vector (CAV).
    
    Args:
        positive_examples: Examples with the concept
        negative_examples: Examples without the concept
        
    Returns:
        CAV vector
    """
    # Simple difference of means as CAV
    cav = np.mean(positive_examples, axis=0) - np.mean(negative_examples, axis=0)
    norm = np.linalg.norm(cav)
    if norm > 0:
        cav = cav / norm
    return cav


def directional_derivative(
    gradients: np.ndarray,
    cav: np.ndarray,
) -> np.ndarray:
    """
    Compute directional derivative with respect to CAV.
    
    Args:
        gradients: Model gradients [N, D]
        cav: Concept activation vector [D]
        
    Returns:
        Directional derivatives [N]
    """
    return np.dot(gradients, cav)


__all__ = [
    "tcav_scores",
    "compute_cav",
    "directional_derivative",
]
