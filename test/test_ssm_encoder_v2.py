import pytest
torch = pytest.importorskip("torch")

from narrative.models.ssm_encoder_v2 import SSMEncoderV2


def test_ssm_encoder_v2_causal_shape():
    import torch
    x = torch.randn(2, 8, 4)
    model = SSMEncoderV2(input_dim=4, d_model=8, n_layers=2, causal=True, chunk_size=3)
    out = model(x)
    assert out.shape == (2, 8)


def test_ssm_encoder_v2_noncausal_shape():
    import torch
    x = torch.randn(2, 8, 4)
    model = SSMEncoderV2(input_dim=4, d_model=8, n_layers=2, causal=False, chunk_size=3)
    out = model(x)
    assert out.shape == (2, 8)


def test_ssm_encoder_v2_chunk_consistency():
    import torch
    torch.manual_seed(0)
    x = torch.randn(2, 10, 4)
    base = SSMEncoderV2(input_dim=4, d_model=8, n_layers=2, causal=True, chunk_size=None, dropout=0.0)
    chunked = SSMEncoderV2(input_dim=4, d_model=8, n_layers=2, causal=True, chunk_size=4, dropout=0.0)
    chunked.load_state_dict(base.state_dict())
    base.eval()
    chunked.eval()
    out_full = base(x)
    out_chunk = chunked(x)
    assert torch.allclose(out_full, out_chunk, atol=1e-6)


def test_ssm_encoder_v2_return_sequence():
    import torch
    x = torch.randn(1, 6, 4)
    model = SSMEncoderV2(input_dim=4, d_model=8, n_layers=1, return_sequence=True)
    out = model(x)
    assert out.shape == (1, 6, 8)
