"""
AutoFit Rule-Based Composer for Block 3.

This module implements an auditable, rule-based composer that selects model
configurations based on data profile meta-features.

Design Principles:
1. All rules are explicit and documented
2. Selection logic is deterministic given the same meta-features
3. Every decision is logged with rationale
4. Supports override for ablation studies

Meta-Features Used:
- nonstationarity_score: Indicates distribution shift over time
- periodicity_score: Strength of periodic patterns
- multiscale_score: Presence of multiple temporal scales
- long_memory_score: Long-range dependencies
- irregular_score: Irregularity in sampling intervals
- heavy_tail_score: Heavy-tailed distributions
- exog_strength: Combined exogenous feature strength
- edgar_strength: EDGAR feature informativeness
- text_strength: Text feature informativeness
- missing_rate: Overall missing data rate
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class ComposerConfig:
    """Configuration for the rule-based composer."""
    
    # Backbone selection thresholds
    ssm_threshold: float = 0.2  # long_memory_score > this -> prefer SSM
    transformer_threshold: float = 0.3  # periodicity_score > this -> prefer Transformer
    decomp_threshold: float = 0.4  # nonstationarity_score > this -> prefer Decomp
    
    # Fusion selection thresholds
    exog_threshold: float = 0.15  # exog_strength > this -> enable fusion
    film_threshold: float = 0.2  # prefer FiLM for moderate exog
    cross_attn_threshold: float = 0.3  # prefer cross-attention for strong exog
    bridge_token_threshold: float = 0.4  # prefer bridge tokens for very strong exog
    
    # Loss selection thresholds
    heavy_tail_threshold: float = 0.5  # heavy_tail_score > this -> prefer robust loss
    
    # Regularization thresholds
    high_missing_threshold: float = 0.5  # missing_rate > this -> stronger dropout
    irregular_threshold: float = 0.3  # irregular_score > this -> time masking
    
    @classmethod
    def from_yaml(cls, path: Path) -> "ComposerConfig":
        """Load config from block3.yaml autofit section."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        rules = data.get("autofit", {}).get("rules", {})
        
        return cls(
            ssm_threshold=rules.get("backbone_selection", {}).get("ssm_threshold", 0.2),
            transformer_threshold=rules.get("backbone_selection", {}).get("transformer_threshold", 0.3),
            decomp_threshold=rules.get("backbone_selection", {}).get("decomp_threshold", 0.4),
            exog_threshold=rules.get("fusion_selection", {}).get("exog_threshold", 0.15),
            film_threshold=rules.get("fusion_selection", {}).get("film_threshold", 0.2),
            cross_attn_threshold=rules.get("fusion_selection", {}).get("cross_attn_threshold", 0.3),
            bridge_token_threshold=rules.get("fusion_selection", {}).get("bridge_token_threshold", 0.4),
            heavy_tail_threshold=rules.get("loss_selection", {}).get("heavy_tail_threshold", 0.5),
            high_missing_threshold=rules.get("regularization", {}).get("high_missing_threshold", 0.5),
            irregular_threshold=rules.get("regularization", {}).get("irregular_threshold", 0.3),
        )


@dataclass
class ComposerDecision:
    """A single composition decision with rationale."""
    component: str
    choice: str
    rationale: str
    meta_features_used: Dict[str, float] = field(default_factory=dict)
    threshold_applied: Optional[float] = None


@dataclass
class ComposedConfig:
    """Output of the composer: a complete model configuration."""
    backbone: str
    backbone_variant: Optional[str] = None
    fusion_type: str = "none"
    loss_type: str = "mse"
    use_ssm: bool = False
    use_decomp: bool = False
    use_irregular_patch: bool = False
    use_time_masking: bool = False
    dropout: float = 0.1
    use_edgar: bool = False
    use_text: bool = False
    use_multiscale: bool = False
    
    decisions: List[ComposerDecision] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backbone": self.backbone,
            "backbone_variant": self.backbone_variant,
            "fusion_type": self.fusion_type,
            "loss_type": self.loss_type,
            "use_ssm": self.use_ssm,
            "use_decomp": self.use_decomp,
            "use_irregular_patch": self.use_irregular_patch,
            "use_time_masking": self.use_time_masking,
            "dropout": self.dropout,
            "use_edgar": self.use_edgar,
            "use_text": self.use_text,
            "use_multiscale": self.use_multiscale,
            "decisions": [
                {
                    "component": d.component,
                    "choice": d.choice,
                    "rationale": d.rationale,
                    "meta_features_used": d.meta_features_used,
                    "threshold_applied": d.threshold_applied,
                }
                for d in self.decisions
            ],
        }


class RuleBasedComposer:
    """
    Auditable rule-based model composer.
    
    Selects model configurations based on data profile meta-features.
    All decisions are logged with rationale for auditability.
    """
    
    def __init__(self, config: Optional[ComposerConfig] = None):
        self.config = config or ComposerConfig()
    
    def compose(self, meta_features: Dict[str, float]) -> ComposedConfig:
        """
        Compose a model configuration based on meta-features.
        
        Args:
            meta_features: Dictionary of meta-features from data profile
        
        Returns:
            ComposedConfig with all decisions and rationale
        """
        result = ComposedConfig(backbone="patchtst")  # Default backbone
        
        # 1. Select backbone
        self._select_backbone(meta_features, result)
        
        # 2. Select fusion type
        self._select_fusion(meta_features, result)
        
        # 3. Select loss function
        self._select_loss(meta_features, result)
        
        # 4. Select regularization
        self._select_regularization(meta_features, result)
        
        # 5. Select auxiliary modules
        self._select_modules(meta_features, result)
        
        return result
    
    def _select_backbone(self, meta: Dict[str, float], result: ComposedConfig):
        """Select backbone architecture."""
        long_memory = meta.get("long_memory_score", 0.0)
        periodicity = meta.get("periodicity_score", 0.0)
        nonstationarity = meta.get("nonstationarity_score", 0.0)
        multiscale = meta.get("multiscale_score", 0.0)
        
        # Decision logic
        if nonstationarity > self.config.decomp_threshold:
            # High non-stationarity: prefer decomposition-based models
            backbone = "timemixer"
            variant = "pp"  # TimeMixer++
            rationale = f"High nonstationarity ({nonstationarity:.3f} > {self.config.decomp_threshold}) suggests decomposition-based backbone"
            result.use_decomp = True
        elif long_memory > self.config.ssm_threshold:
            # Long memory: prefer SSM-based models
            backbone = "itransformer"
            variant = "ssm"
            rationale = f"High long_memory ({long_memory:.3f} > {self.config.ssm_threshold}) suggests SSM augmentation"
            result.use_ssm = True
        elif periodicity > self.config.transformer_threshold:
            # Strong periodicity: prefer attention-based models
            backbone = "patchtst"
            variant = None
            rationale = f"Strong periodicity ({periodicity:.3f} > {self.config.transformer_threshold}) suggests Transformer backbone"
        elif multiscale > 0.5:
            # Multi-scale patterns: prefer TimesNet
            backbone = "timesnet"
            variant = None
            rationale = f"Multi-scale patterns ({multiscale:.3f}) suggest TimesNet"
        else:
            # Default: PatchTST as strong general-purpose baseline
            backbone = "patchtst"
            variant = None
            rationale = "Default: PatchTST as strong general-purpose baseline"
        
        result.backbone = backbone
        result.backbone_variant = variant
        result.decisions.append(ComposerDecision(
            component="backbone",
            choice=f"{backbone}" + (f"-{variant}" if variant else ""),
            rationale=rationale,
            meta_features_used={
                "long_memory_score": long_memory,
                "periodicity_score": periodicity,
                "nonstationarity_score": nonstationarity,
                "multiscale_score": multiscale,
            },
        ))
    
    def _select_fusion(self, meta: Dict[str, float], result: ComposedConfig):
        """Select fusion type for exogenous features."""
        exog = meta.get("exog_strength", 0.0)
        edgar = meta.get("edgar_strength", 0.0)
        text = meta.get("text_strength", 0.0)
        
        if exog < self.config.exog_threshold:
            # Weak exogenous signal: no fusion needed
            fusion = "none"
            rationale = f"Weak exog signal ({exog:.3f} < {self.config.exog_threshold}): no fusion"
        elif exog < self.config.film_threshold:
            # Moderate: FiLM conditioning
            fusion = "film"
            rationale = f"Moderate exog ({exog:.3f}): FiLM conditioning"
        elif exog < self.config.cross_attn_threshold:
            # Stronger: cross-attention
            fusion = "cross_attention"
            rationale = f"Strong exog ({exog:.3f}): cross-attention fusion"
        else:
            # Very strong: bridge tokens
            fusion = "bridge_token"
            rationale = f"Very strong exog ({exog:.3f}): bridge token fusion"
        
        result.fusion_type = fusion
        result.use_edgar = edgar > 0.05
        result.use_text = text > 0.05
        
        result.decisions.append(ComposerDecision(
            component="fusion",
            choice=fusion,
            rationale=rationale,
            meta_features_used={
                "exog_strength": exog,
                "edgar_strength": edgar,
                "text_strength": text,
            },
            threshold_applied=self.config.exog_threshold,
        ))
    
    def _select_loss(self, meta: Dict[str, float], result: ComposedConfig):
        """Select loss function."""
        heavy_tail = meta.get("heavy_tail_score", 0.0)
        
        if heavy_tail > self.config.heavy_tail_threshold:
            # Heavy tails: robust loss
            loss = "huber"
            rationale = f"Heavy tails ({heavy_tail:.3f} > {self.config.heavy_tail_threshold}): Huber loss for robustness"
        else:
            # Standard MSE
            loss = "mse"
            rationale = "Standard MSE loss"
        
        result.loss_type = loss
        result.decisions.append(ComposerDecision(
            component="loss",
            choice=loss,
            rationale=rationale,
            meta_features_used={"heavy_tail_score": heavy_tail},
            threshold_applied=self.config.heavy_tail_threshold,
        ))
    
    def _select_regularization(self, meta: Dict[str, float], result: ComposedConfig):
        """Select regularization strategy."""
        missing = meta.get("missing_rate", 0.0)
        irregular = meta.get("irregular_score", 0.0)
        
        # Dropout based on missing rate
        if missing > self.config.high_missing_threshold:
            dropout = 0.3
            rationale = f"High missing rate ({missing:.3f}): stronger dropout"
        else:
            dropout = 0.1
            rationale = "Standard dropout"
        
        result.dropout = dropout
        result.decisions.append(ComposerDecision(
            component="dropout",
            choice=str(dropout),
            rationale=rationale,
            meta_features_used={"missing_rate": missing},
            threshold_applied=self.config.high_missing_threshold,
        ))
        
        # Time masking for irregular data
        if irregular > self.config.irregular_threshold:
            result.use_time_masking = True
            result.use_irregular_patch = True
            rationale = f"High irregularity ({irregular:.3f}): enable time masking and irregular patching"
        else:
            result.use_time_masking = False
            result.use_irregular_patch = False
            rationale = "Regular sampling: no time masking"
        
        result.decisions.append(ComposerDecision(
            component="time_masking",
            choice=str(result.use_time_masking),
            rationale=rationale,
            meta_features_used={"irregular_score": irregular},
            threshold_applied=self.config.irregular_threshold,
        ))
    
    def _select_modules(self, meta: Dict[str, float], result: ComposedConfig):
        """Select auxiliary modules."""
        multiscale = meta.get("multiscale_score", 0.0)
        
        if multiscale > 0.3:
            result.use_multiscale = True
            rationale = f"Multi-scale patterns ({multiscale:.3f}): enable multi-scale module"
        else:
            result.use_multiscale = False
            rationale = "No strong multi-scale patterns"
        
        result.decisions.append(ComposerDecision(
            component="multiscale",
            choice=str(result.use_multiscale),
            rationale=rationale,
            meta_features_used={"multiscale_score": multiscale},
        ))


def compose_from_profile(profile_path: Path, config_path: Optional[Path] = None) -> ComposedConfig:
    """
    Convenience function to compose from a profile JSON file.
    
    Args:
        profile_path: Path to profile.json from block3_profile_data.py
        config_path: Optional path to block3.yaml for composer config
    
    Returns:
        ComposedConfig with all decisions
    """
    # Load profile
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    meta_features = profile.get("meta_features", {})
    
    # Load composer config
    if config_path is not None:
        composer_config = ComposerConfig.from_yaml(config_path)
    else:
        composer_config = ComposerConfig()
    
    # Compose
    composer = RuleBasedComposer(composer_config)
    return composer.compose(meta_features)


def get_profile_path_from_pointer(pointer_path: Path = Path("docs/audits/FULL_SCALE_POINTER.yaml")) -> Path:
    """Resolve profile path from FreezePointer."""
    import yaml
    data = yaml.safe_load(pointer_path.read_text(encoding="utf-8"))
    stamp = data["stamp"]
    # Profile is stored under orchestrator analysis
    return Path(f"runs/orchestrator/20260129_073037/block3_{stamp}/profile/profile.json")


if __name__ == "__main__":
    # Example usage - resolve path via pointer
    import sys
    
    profile_path = get_profile_path_from_pointer()
    
    if profile_path.exists():
        result = compose_from_profile(profile_path)
        print("Composed Configuration:")
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Profile not found: {profile_path}")
        print("Run: python scripts/block3_profile_data.py first")
