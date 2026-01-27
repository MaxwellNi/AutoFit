import pytest

from narrative.auto_fit.compose_candidates import compose_candidates


def test_compose_candidates_expands_space():
    meta = {
        "nonstationarity_score": 0.5,
        "periodicity_score": 0.4,
        "long_memory_score": 0.3,
        "multiscale_score": 0.4,
        "exog_strength": 0.2,
    }
    candidates = compose_candidates(meta, available_backbones=["patchtst", "itransformer", "timemixer", "timesnet", "dlinear"])
    assert len(candidates) > 0
    sample = candidates[0]
    assert "fusion_type" in sample
    assert "module_flags" in sample
    assert "use_irregular_patch" in sample
    assert "edgar_on" in sample
    assert "explainability_on" in sample
