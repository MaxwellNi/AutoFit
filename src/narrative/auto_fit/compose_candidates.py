from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def compose_candidates(
    meta: Dict[str, float],
    *,
    available_backbones: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """
    Heuristic candidate composer based on meta-profile.
    Expands search space with SOTA + irregular patch + SSM v2 + fusion options.
    """
    nonstat = meta.get("nonstationarity_score", 0.0) > 0.3
    periodic = meta.get("periodicity_score", 0.0) > 0.2
    long_memory = meta.get("long_memory_score", 0.0) > 0.2
    exog = meta.get("exog_strength", 0.0) > 0.15

    fusion_types = ["none", "concat", "film", "cross_attention", "bridge_token"] if exog else ["none"]
    edgar_on = [True, False] if exog else [False]
    explain_on = [True, False]

    base_backbones = [
        "patchtst",
        "itransformer",
        "timemixer",
        "timesnet",
        "cats",
    ]
    if available_backbones is not None:
        base_backbones = [b for b in base_backbones if b in set(available_backbones)]
        if "timemixer" in set(available_backbones) and "timemixer" not in base_backbones:
            base_backbones.append("timemixer")
    if "dlinear" in (available_backbones or []):
        base_backbones.append("dlinear")

    candidates = []

    for backbone in base_backbones:
        for fusion_type in fusion_types:
            for use_edgar in edgar_on:
                for expl in explain_on:
                    flags = {
                        "nonstat": nonstat,
                        "multiscale": periodic,
                        "multiscale_fft": periodic and meta.get("multiscale_score", 0.0) > 0.35,
                        "ssm": long_memory,
                    }
                    module_cfg = {
                        "ssm_variant": "v2",
                        "ssm_chunk_size": 256,
                        "ssm_causal": True,
                        "ssm_kernel_size": 5,
                    }
                    candidates.append(
                        {
                            "backbone": backbone,
                            "backbone_tag": "timemixer++" if backbone == "timemixer" else backbone,
                            "fusion_type": fusion_type if use_edgar else "none",
                            "module_flags": flags,
                            "module_cfg": module_cfg,
                            "use_irregular_patch": True,
                            "edgar_on": use_edgar,
                            "explainability_on": expl,
                            "train": {"epochs": 3, "lr": 1e-3},
                        }
                    )

    # Conservative baseline
    candidates.append(
        {
            "backbone": "dlinear",
            "fusion_type": "none",
            "module_flags": {"nonstat": False, "multiscale": False, "ssm": False},
            "module_cfg": {"ssm_variant": "mamba"},
            "use_irregular_patch": False,
            "edgar_on": False,
            "explainability_on": False,
            "train": {"epochs": 3, "lr": 1e-3},
        }
    )

    return candidates
