"""
Auditable Concept Bottleneck for Block 3.

This module implements concept-based interpretability for time series forecasting.
It extracts interpretable concepts from text and EDGAR features, then uses these
as an intermediate representation for prediction.

Design Principles:
1. Every concept has an explicit definition
2. Concept activations are logged per prediction
3. Marginal contribution of each concept is measurable
4. Supports TCAV-style concept importance analysis

Concept Categories:
- Risk concepts: financial_risk, market_risk, operational_risk
- Sentiment concepts: optimism, pessimism, uncertainty
- Disclosure concepts: transparency, completeness, timeliness
- Financial concepts: revenue_growth, profitability, funding_progress
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ConceptDefinition:
    """Definition of a single concept."""
    name: str
    category: str
    description: str
    extraction_method: str  # keyword | embedding | model
    keywords: List[str] = field(default_factory=list)
    embedding_model: Optional[str] = None
    threshold: float = 0.5


@dataclass
class ConceptActivation:
    """Activation of a concept for a single sample."""
    concept_name: str
    activation: float  # 0-1 score
    confidence: float  # 0-1 confidence
    evidence: Optional[str] = None  # Text snippet or feature names


@dataclass
class ConceptBottleneckOutput:
    """Output from the concept bottleneck layer."""
    sample_id: str
    activations: List[ConceptActivation]
    prediction: Optional[float] = None
    concept_contributions: Optional[Dict[str, float]] = None  # Marginal contribution per concept
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "activations": [
                {
                    "concept_name": a.concept_name,
                    "activation": a.activation,
                    "confidence": a.confidence,
                    "evidence": a.evidence,
                }
                for a in self.activations
            ],
            "prediction": self.prediction,
            "concept_contributions": self.concept_contributions,
        }


class ConceptBank:
    """
    Bank of concept definitions.
    
    Defines all concepts used in the bottleneck layer.
    """
    
    def __init__(self):
        self.concepts: Dict[str, ConceptDefinition] = {}
        self._init_default_concepts()
    
    def _init_default_concepts(self):
        """Initialize default concept definitions."""
        
        # Risk concepts
        self.add_concept(ConceptDefinition(
            name="financial_risk",
            category="risk",
            description="Indicators of financial risk or instability",
            extraction_method="keyword",
            keywords=["risk", "loss", "liability", "debt", "default", "bankruptcy", "uncertain"],
        ))
        
        self.add_concept(ConceptDefinition(
            name="market_risk",
            category="risk",
            description="Market-related risk factors",
            extraction_method="keyword",
            keywords=["market", "competition", "volatile", "economic", "downturn", "recession"],
        ))
        
        self.add_concept(ConceptDefinition(
            name="operational_risk",
            category="risk",
            description="Operational and execution risks",
            extraction_method="keyword",
            keywords=["operational", "execution", "management", "scalability", "capacity"],
        ))
        
        # Sentiment concepts
        self.add_concept(ConceptDefinition(
            name="optimism",
            category="sentiment",
            description="Positive outlook and confidence",
            extraction_method="keyword",
            keywords=["growth", "opportunity", "success", "confident", "positive", "strong"],
        ))
        
        self.add_concept(ConceptDefinition(
            name="pessimism",
            category="sentiment",
            description="Negative outlook or concerns",
            extraction_method="keyword",
            keywords=["concern", "challenge", "difficult", "negative", "decline", "weak"],
        ))
        
        self.add_concept(ConceptDefinition(
            name="uncertainty",
            category="sentiment",
            description="Uncertainty and ambiguity",
            extraction_method="keyword",
            keywords=["uncertain", "may", "might", "could", "possible", "unknown"],
        ))
        
        # Disclosure concepts
        self.add_concept(ConceptDefinition(
            name="transparency",
            category="disclosure",
            description="Level of disclosure detail",
            extraction_method="keyword",
            keywords=["detail", "specific", "comprehensive", "complete", "full disclosure"],
        ))
        
        self.add_concept(ConceptDefinition(
            name="regulatory_compliance",
            category="disclosure",
            description="Regulatory and compliance mentions",
            extraction_method="keyword",
            keywords=["SEC", "compliance", "regulatory", "filing", "legal", "regulation"],
        ))
        
        # Financial concepts (from EDGAR features)
        self.add_concept(ConceptDefinition(
            name="funding_progress",
            category="financial",
            description="Progress toward funding goal",
            extraction_method="model",  # Computed from EDGAR ratios
            keywords=[],
        ))
        
        self.add_concept(ConceptDefinition(
            name="investor_interest",
            category="financial",
            description="Level of investor engagement",
            extraction_method="model",
            keywords=[],
        ))
    
    def add_concept(self, concept: ConceptDefinition):
        """Add a concept to the bank."""
        self.concepts[concept.name] = concept
    
    def get_concept(self, name: str) -> Optional[ConceptDefinition]:
        """Get a concept by name."""
        return self.concepts.get(name)
    
    def list_concepts(self) -> List[str]:
        """List all concept names."""
        return list(self.concepts.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Export concept bank as dictionary."""
        return {
            name: {
                "name": c.name,
                "category": c.category,
                "description": c.description,
                "extraction_method": c.extraction_method,
                "keywords": c.keywords,
            }
            for name, c in self.concepts.items()
        }


class ConceptExtractor:
    """
    Extracts concept activations from features.
    
    Supports multiple extraction methods:
    - Keyword matching for text features
    - Threshold-based for numerical features
    - Model-based for complex concepts
    """
    
    def __init__(self, concept_bank: ConceptBank):
        self.concept_bank = concept_bank
    
    def extract_from_text(
        self,
        text: str,
        concept_name: str,
    ) -> ConceptActivation:
        """Extract concept activation from text."""
        concept = self.concept_bank.get_concept(concept_name)
        if concept is None:
            return ConceptActivation(concept_name, 0.0, 0.0)
        
        if concept.extraction_method == "keyword":
            # Simple keyword matching
            text_lower = text.lower() if text else ""
            matches = [kw for kw in concept.keywords if kw.lower() in text_lower]
            
            # Activation = proportion of keywords found
            if concept.keywords:
                activation = len(matches) / len(concept.keywords)
            else:
                activation = 0.0
            
            # Confidence based on text length
            confidence = min(1.0, len(text_lower) / 1000) if text_lower else 0.0
            
            evidence = ", ".join(matches[:5]) if matches else None
            
            return ConceptActivation(
                concept_name=concept_name,
                activation=activation,
                confidence=confidence,
                evidence=evidence,
            )
        
        else:
            # Default: zero activation
            return ConceptActivation(concept_name, 0.0, 0.0)
    
    def extract_from_edgar(
        self,
        edgar_features: Dict[str, float],
        concept_name: str,
    ) -> ConceptActivation:
        """Extract concept activation from EDGAR features."""
        concept = self.concept_bank.get_concept(concept_name)
        if concept is None:
            return ConceptActivation(concept_name, 0.0, 0.0)
        
        if concept_name == "funding_progress":
            # Compute from EDGAR ratios
            total_sold = edgar_features.get("last_total_amount_sold", 0)
            total_offering = edgar_features.get("last_total_offering_amount", 1)
            
            if total_offering > 0:
                activation = min(1.0, total_sold / total_offering)
            else:
                activation = 0.0
            
            return ConceptActivation(
                concept_name=concept_name,
                activation=activation,
                confidence=0.8 if total_offering > 0 else 0.0,
                evidence=f"sold={total_sold:.0f}, offering={total_offering:.0f}",
            )
        
        elif concept_name == "investor_interest":
            # Based on number of investors
            n_investors = edgar_features.get("last_total_number_already_invested", 0)
            
            # Normalize to 0-1 (assuming max ~1000 investors)
            activation = min(1.0, n_investors / 1000)
            
            return ConceptActivation(
                concept_name=concept_name,
                activation=activation,
                confidence=0.8 if n_investors > 0 else 0.0,
                evidence=f"investors={n_investors}",
            )
        
        else:
            return ConceptActivation(concept_name, 0.0, 0.0)
    
    def extract_all(
        self,
        text_features: Optional[Dict[str, str]] = None,
        edgar_features: Optional[Dict[str, float]] = None,
    ) -> List[ConceptActivation]:
        """Extract all concepts from available features."""
        activations = []
        
        for concept_name, concept in self.concept_bank.concepts.items():
            if concept.extraction_method == "keyword" and text_features:
                # Combine all text fields
                combined_text = " ".join(str(v) for v in text_features.values() if v)
                activation = self.extract_from_text(combined_text, concept_name)
            elif concept.extraction_method == "model" and edgar_features:
                activation = self.extract_from_edgar(edgar_features, concept_name)
            else:
                activation = ConceptActivation(concept_name, 0.0, 0.0)
            
            activations.append(activation)
        
        return activations


class ConceptBottleneck:
    """
    Concept Bottleneck layer for interpretable predictions.
    
    Architecture:
    1. Extract concept activations from raw features
    2. Use concept activations as intermediate representation
    3. Predict target from concept activations
    4. Log concept contributions for auditability
    """
    
    def __init__(self, concept_bank: Optional[ConceptBank] = None):
        self.concept_bank = concept_bank or ConceptBank()
        self.extractor = ConceptExtractor(self.concept_bank)
        self.concept_weights: Optional[Dict[str, float]] = None
    
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        text_cols: List[str],
        edgar_cols: List[str],
    ):
        """
        Fit concept weights using simple linear regression.
        
        This learns the marginal contribution of each concept to the target.
        """
        # Extract concepts for all samples
        concept_matrix = []
        targets = []
        
        for _, row in df.iterrows():
            text_features = {col: row.get(col, "") for col in text_cols}
            edgar_features = {col: row.get(col, 0) for col in edgar_cols}
            
            activations = self.extractor.extract_all(text_features, edgar_features)
            concept_vector = [a.activation for a in activations]
            concept_matrix.append(concept_vector)
            
            target = row.get(target_col)
            if pd.notna(target):
                targets.append(target)
            else:
                targets.append(0)
        
        concept_matrix = np.array(concept_matrix)
        targets = np.array(targets)
        
        # Simple linear regression
        # Add bias term
        X = np.column_stack([concept_matrix, np.ones(len(concept_matrix))])
        
        # Solve least squares
        try:
            weights, _, _, _ = np.linalg.lstsq(X, targets, rcond=None)
            self.concept_weights = {
                name: float(weights[i])
                for i, name in enumerate(self.concept_bank.list_concepts())
            }
            self.concept_weights["_bias"] = float(weights[-1])
        except Exception:
            self.concept_weights = {name: 0.0 for name in self.concept_bank.list_concepts()}
            self.concept_weights["_bias"] = float(np.mean(targets))
    
    def predict(
        self,
        text_features: Optional[Dict[str, str]] = None,
        edgar_features: Optional[Dict[str, float]] = None,
        sample_id: str = "unknown",
    ) -> ConceptBottleneckOutput:
        """
        Make a prediction with full concept explanation.
        """
        # Extract concepts
        activations = self.extractor.extract_all(text_features, edgar_features)
        
        # Compute prediction
        prediction = self.concept_weights.get("_bias", 0.0) if self.concept_weights else 0.0
        contributions = {}
        
        for activation in activations:
            weight = self.concept_weights.get(activation.concept_name, 0.0) if self.concept_weights else 0.0
            contribution = activation.activation * weight
            prediction += contribution
            contributions[activation.concept_name] = contribution
        
        return ConceptBottleneckOutput(
            sample_id=sample_id,
            activations=activations,
            prediction=prediction,
            concept_contributions=contributions,
        )
    
    def explain(self, output: ConceptBottleneckOutput) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Prediction for sample {output.sample_id}: {output.prediction:.2f}",
            "",
            "Top contributing concepts:",
        ]
        
        if output.concept_contributions:
            sorted_concepts = sorted(
                output.concept_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            
            for name, contrib in sorted_concepts[:5]:
                activation = next((a for a in output.activations if a.concept_name == name), None)
                if activation:
                    lines.append(
                        f"  - {name}: contribution={contrib:.3f} "
                        f"(activation={activation.activation:.3f}, "
                        f"evidence={activation.evidence or 'N/A'})"
                    )
        
        return "\n".join(lines)
    
    def save_audit_log(self, outputs: List[ConceptBottleneckOutput], path: Path):
        """Save audit log for all predictions."""
        log = {
            "concept_bank": self.concept_bank.to_dict(),
            "concept_weights": self.concept_weights,
            "predictions": [o.to_dict() for o in outputs],
        }
        path.write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # Example usage
    bank = ConceptBank()
    print("Concept Bank:")
    for name in bank.list_concepts():
        concept = bank.get_concept(name)
        print(f"  {name}: {concept.description}")
    
    # Example extraction
    extractor = ConceptExtractor(bank)
    
    sample_text = {
        "description": "We face significant market risk and competition. Our growth opportunity is strong.",
        "financial_risks": "The company has substantial debt and may face bankruptcy.",
    }
    
    sample_edgar = {
        "last_total_amount_sold": 500000,
        "last_total_offering_amount": 1000000,
        "last_total_number_already_invested": 150,
    }
    
    activations = extractor.extract_all(sample_text, sample_edgar)
    
    print("\nConcept Activations:")
    for a in activations:
        if a.activation > 0:
            print(f"  {a.concept_name}: {a.activation:.3f} (evidence: {a.evidence})")
