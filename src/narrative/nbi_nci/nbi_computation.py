"""
Narrative Behavior Index (NBI) Computation Module.

This module implements the computation of NBI across 15 bias dimensions
for narrative-based model selection and interpretability.

NBI Dimensions (15 total):
1. Optimism Bias - Overconfidence in positive outcomes
2. Anchoring Bias - Over-reliance on initial information
3. Confirmation Bias - Selective information processing
4. Recency Bias - Overweighting recent events
5. Availability Bias - Salience-based judgments
6. Herding Bias - Following crowd behavior
7. Overconfidence Bias - Excessive certainty
8. Loss Aversion - Asymmetric loss/gain sensitivity
9. Framing Bias - Presentation-dependent decisions
10. Sunk Cost Bias - Honoring past investments
11. Hindsight Bias - Post-hoc rationalization
12. Self-Attribution Bias - Internal success/external failure attribution
13. Status Quo Bias - Preference for current state
14. Projection Bias - Assuming others share views
15. Affect Bias - Emotion-driven decisions

Each dimension is computed from text features and can be used for:
- Gating model selection (MoE router)
- Concept bottleneck interpretability
- DiD mediator analysis
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dimension_moe import DimensionMoE, DimensionMoERouter


class NBIDimension(Enum):
    """The 15 NBI dimensions for narrative bias analysis."""
    OPTIMISM = 0
    ANCHORING = 1
    CONFIRMATION = 2
    RECENCY = 3
    AVAILABILITY = 4
    HERDING = 5
    OVERCONFIDENCE = 6
    LOSS_AVERSION = 7
    FRAMING = 8
    SUNK_COST = 9
    HINDSIGHT = 10
    SELF_ATTRIBUTION = 11
    STATUS_QUO = 12
    PROJECTION = 13
    AFFECT = 14


NBI_DIMENSION_NAMES = [
    "optimism",
    "anchoring",
    "confirmation",
    "recency",
    "availability",
    "herding",
    "overconfidence",
    "loss_aversion",
    "framing",
    "sunk_cost",
    "hindsight",
    "self_attribution",
    "status_quo",
    "projection",
    "affect",
]

# Keywords for keyword-based extraction (fallback when embeddings unavailable)
NBI_KEYWORDS: Dict[str, List[str]] = {
    "optimism": ["opportunity", "growth", "potential", "success", "promising", "confident", "positive"],
    "anchoring": ["initial", "original", "first", "baseline", "starting", "reference"],
    "confirmation": ["confirm", "support", "evidence", "validate", "consistent", "align"],
    "recency": ["recent", "latest", "current", "new", "update", "just"],
    "availability": ["notable", "memorable", "significant", "major", "prominent", "visible"],
    "herding": ["others", "market", "industry", "peers", "trend", "popular", "following"],
    "overconfidence": ["certain", "definite", "guaranteed", "assured", "undoubtedly", "clearly"],
    "loss_aversion": ["risk", "protect", "preserve", "avoid", "loss", "downside", "safe"],
    "framing": ["perspective", "view", "consider", "present", "frame", "position"],
    "sunk_cost": ["invested", "committed", "already", "spent", "past", "previous"],
    "hindsight": ["expected", "predicted", "foresaw", "obvious", "clear", "evident"],
    "self_attribution": ["achieved", "accomplished", "success", "credit", "result", "effort"],
    "status_quo": ["maintain", "continue", "stable", "current", "existing", "unchanged"],
    "projection": ["believe", "think", "assume", "expect", "similar", "share"],
    "affect": ["feel", "emotion", "excited", "worried", "enthusiastic", "concerned"],
}


@dataclass
class NBIOutput:
    """Output from NBI computation."""
    dimension_scores: np.ndarray  # [batch, 15]
    dominant_dimensions: List[str]  # Top-k dimension names
    confidence: float
    gating_weights: Optional[np.ndarray] = None  # MoE router weights
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension_scores": self.dimension_scores.tolist() if isinstance(self.dimension_scores, np.ndarray) else self.dimension_scores,
            "dominant_dimensions": self.dominant_dimensions,
            "confidence": self.confidence,
            "gating_weights": self.gating_weights.tolist() if self.gating_weights is not None else None,
        }


class KeywordNBIExtractor:
    """
    Keyword-based NBI dimension extraction.
    
    Simple baseline when embeddings are not available.
    """
    
    def __init__(self, keywords: Optional[Dict[str, List[str]]] = None):
        self.keywords = keywords or NBI_KEYWORDS
    
    def extract(self, text: str) -> np.ndarray:
        """
        Extract NBI dimension scores from text.
        
        Args:
            text: Input text
            
        Returns:
            Array of shape [15] with dimension scores
        """
        text_lower = text.lower()
        scores = np.zeros(len(NBI_DIMENSION_NAMES))
        
        for i, dim_name in enumerate(NBI_DIMENSION_NAMES):
            keywords = self.keywords.get(dim_name, [])
            count = sum(1 for kw in keywords if kw in text_lower)
            # Normalize by number of keywords
            scores[i] = min(count / max(len(keywords), 1), 1.0)
        
        return scores
    
    def batch_extract(self, texts: List[str]) -> np.ndarray:
        """Extract NBI scores for a batch of texts."""
        return np.stack([self.extract(t) for t in texts])


class NBIComputationModel(nn.Module):
    """
    Neural NBI computation model with optional MoE router.
    
    Can be used in two modes:
    1. Direct embedding -> NBI dimension projection
    2. MoE-gated model selection based on NBI dimensions
    """
    
    def __init__(
        self,
        emb_dim: int = 768,
        n_bias_dims: int = 15,
        hidden_dim: int = 256,
        n_concepts: int = 32,
        use_moe_router: bool = False,
        moe_top_k: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize NBI computation model.
        
        Args:
            emb_dim: Input embedding dimension
            n_bias_dims: Number of NBI dimensions (default 15)
            hidden_dim: Hidden layer dimension
            n_concepts: Number of intermediate concepts
            use_moe_router: Whether to use MoE routing
            moe_top_k: Top-k experts for MoE
            dropout: Dropout rate
        """
        super().__init__()
        
        self.emb_dim = emb_dim
        self.n_bias_dims = n_bias_dims
        self.n_concepts = n_concepts
        self.use_moe_router = use_moe_router
        
        # Text encoder projection
        self.text_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Concept extraction layer
        self.concept_layer = nn.Sequential(
            nn.Linear(hidden_dim, n_concepts),
            nn.Tanh(),
        )
        
        # NBI dimension projection (from concepts)
        self.dim_proj = nn.Sequential(
            nn.Linear(n_concepts, n_bias_dims),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )
        
        # MoE router (optional)
        if use_moe_router:
            self.moe = DimensionMoE(
                input_dim=hidden_dim,
                num_dims=n_bias_dims,
                hidden_dim=hidden_dim // 2,
                top_k=moe_top_k,
            )
        else:
            self.moe = None
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_concepts: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input embeddings [B, T, D] or [B, D]
            return_concepts: Whether to return intermediate concepts
            
        Returns:
            Dict with:
                - dim_scores: [B, 15] NBI dimension scores
                - confidence: [B, 1] prediction confidence
                - concepts: [B, n_concepts] (if return_concepts)
                - gating_diagnostics: MoE routing info (if use_moe_router)
                - moe_losses: dict with load_balance_loss, sparsity_loss
        """
        # Handle input shape
        if x.dim() == 3:
            # [B, T, D] -> [B, D] via mean pooling
            x_pooled = x.mean(dim=1)
        elif x.dim() == 2:
            x_pooled = x
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
        
        # Project to hidden space
        h = self.text_proj(x_pooled)  # [B, hidden_dim]
        
        # Extract concepts
        concepts = self.concept_layer(h)  # [B, n_concepts]
        
        # Project to NBI dimensions
        dim_scores = self.dim_proj(concepts)  # [B, n_bias_dims]
        
        # Confidence
        confidence = self.confidence_head(h)  # [B, 1]
        
        output = {
            "dim_scores": dim_scores,
            "confidence": confidence,
        }
        
        if return_concepts:
            output["concepts"] = concepts
        
        # MoE routing (if enabled)
        if self.moe is not None:
            moe_out = self.moe(h.unsqueeze(1) if h.dim() == 2 else h)
            output["gating_diagnostics"] = moe_out["gating_diagnostics"]
            output["moe_losses"] = {
                "load_balance_loss": moe_out["gating_diagnostics"]["load_balance_loss"],
                "sparsity_loss": moe_out["gating_diagnostics"]["sparsity_loss"],
            }
        
        return output
    
    def get_dominant_dimensions(
        self,
        dim_scores: torch.Tensor,
        top_k: int = 3,
    ) -> List[List[str]]:
        """
        Get top-k dominant NBI dimensions.
        
        Args:
            dim_scores: [B, 15] dimension scores
            top_k: Number of top dimensions to return
            
        Returns:
            List of lists with dimension names
        """
        _, indices = torch.topk(dim_scores, k=min(top_k, dim_scores.size(-1)), dim=-1)
        indices = indices.cpu().numpy()
        
        result = []
        for batch_idx in range(indices.shape[0]):
            dims = [NBI_DIMENSION_NAMES[i] for i in indices[batch_idx]]
            result.append(dims)
        
        return result


class NBIAggregator:
    """
    Aggregates NBI scores across multiple samples or time steps.
    
    Useful for entity-level NBI profiles.
    """
    
    def __init__(self, aggregation: str = "mean"):
        """
        Args:
            aggregation: Aggregation method (mean, max, last, weighted)
        """
        self.aggregation = aggregation
    
    def aggregate(
        self,
        scores: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Aggregate NBI scores.
        
        Args:
            scores: [N, 15] scores to aggregate
            weights: Optional [N] weights for weighted aggregation
            
        Returns:
            [15] aggregated scores
        """
        if len(scores) == 0:
            return np.zeros(len(NBI_DIMENSION_NAMES))
        
        if self.aggregation == "mean":
            return np.mean(scores, axis=0)
        elif self.aggregation == "max":
            return np.max(scores, axis=0)
        elif self.aggregation == "last":
            return scores[-1]
        elif self.aggregation == "weighted":
            if weights is None:
                weights = np.ones(len(scores)) / len(scores)
            weights = weights / weights.sum()
            return np.average(scores, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


def compute_nbi_from_text(
    texts: List[str],
    model: Optional[NBIComputationModel] = None,
    embeddings: Optional[np.ndarray] = None,
    use_keywords: bool = True,
) -> NBIOutput:
    """
    Convenience function to compute NBI from text.
    
    Args:
        texts: List of text strings
        model: Optional neural model
        embeddings: Pre-computed embeddings [B, D]
        use_keywords: Whether to use keyword extraction as fallback
        
    Returns:
        NBIOutput with dimension scores
    """
    if model is not None and embeddings is not None:
        # Use neural model
        with torch.no_grad():
            x = torch.tensor(embeddings, dtype=torch.float32)
            out = model(x)
            dim_scores = out["dim_scores"].numpy()
            confidence = out["confidence"].mean().item()
            
            # Get dominant dimensions
            dominant = model.get_dominant_dimensions(out["dim_scores"], top_k=3)
            
            return NBIOutput(
                dimension_scores=dim_scores,
                dominant_dimensions=dominant[0] if len(dominant) == 1 else dominant,
                confidence=confidence,
                gating_weights=out.get("gating_diagnostics", {}).get("gate_weights", None),
            )
    
    elif use_keywords:
        # Fallback to keyword extraction
        extractor = KeywordNBIExtractor()
        dim_scores = extractor.batch_extract(texts)
        
        # Find dominant dimensions
        mean_scores = dim_scores.mean(axis=0)
        top_indices = np.argsort(mean_scores)[-3:][::-1]
        dominant = [NBI_DIMENSION_NAMES[i] for i in top_indices]
        
        return NBIOutput(
            dimension_scores=dim_scores,
            dominant_dimensions=dominant,
            confidence=0.5,  # Lower confidence for keyword-based
        )
    
    else:
        raise ValueError("Either model+embeddings or use_keywords must be provided")


# For backward compatibility and testing
def create_nbi_model(
    emb_dim: int = 768,
    use_moe: bool = True,
    **kwargs,
) -> NBIComputationModel:
    """Create an NBI computation model with default settings."""
    return NBIComputationModel(
        emb_dim=emb_dim,
        n_bias_dims=15,
        use_moe_router=use_moe,
        **kwargs,
    )


__all__ = [
    "NBIDimension",
    "NBI_DIMENSION_NAMES",
    "NBI_KEYWORDS",
    "NBIOutput",
    "KeywordNBIExtractor",
    "NBIComputationModel",
    "NBIAggregator",
    "compute_nbi_from_text",
    "create_nbi_model",
]
