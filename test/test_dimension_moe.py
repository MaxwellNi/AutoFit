import pytest
torch = pytest.importorskip("torch")

from narrative.nbi_nci.dimension_moe import DimensionMoE
from narrative.nbi_nci.nbi_computation import NBIComputationModel


def test_dimension_moe_shapes():
    import torch
    x = torch.randn(2, 5, 8)
    moe = DimensionMoE(input_dim=8, num_dims=4, hidden_dim=16, top_k=2)
    out = moe(x)
    assert out["dim_scores"].shape == (2, 4)
    diag = out["gating_diagnostics"]
    assert diag["gate_weights"].shape == (2, 4)


def test_nbi_computation_with_moe_router():
    import torch
    model = NBIComputationModel(emb_dim=8, use_moe_router=True, n_concepts=8, n_bias_dims=4)
    x = torch.randn(2, 3, 8)
    out = model(x)
    assert "dim_scores" in out
    assert "gating_diagnostics" in out
    assert "moe_losses" in out
