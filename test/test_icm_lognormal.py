#!/usr/bin/env python3
import numpy as np
import torch

from src.narrative.block3.models.single_model_mainline.causal_decoders import ICMLogNormalDecoder


def _mock_batch(n: int = 256, d: int = 32):
    rng = np.random.default_rng(7)
    h = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    pos_mask = rng.random(n) > 0.70
    base = np.exp(1.0 + 0.8 * h[pos_mask, 0] + 0.5 * rng.normal(size=pos_mask.sum()))
    y[pos_mask] = base.astype(np.float32)
    # Inject a few extreme outliers.
    out_idx = np.where(pos_mask)[0][:5]
    y[out_idx] *= 20.0
    return torch.from_numpy(h), torch.from_numpy(y)


def test_lognormal_branch_zero_mask_and_finite_loss():
    torch.manual_seed(0)
    h, y = _mock_batch()
    dec = ICMLogNormalDecoder(hidden_dim=h.shape[1])

    out = dec.compute_loss(h, y)
    assert torch.isfinite(out.total_loss)
    assert out.stats["positive_fraction"] > 0.0

    y_all_zero = torch.zeros_like(y)
    out_zero = dec.compute_loss(h, y_all_zero)
    assert torch.isfinite(out_zero.total_loss)
    assert abs(out_zero.stats["main_loss"]) < 1e-8


def test_lognormal_branch_lateral_detach_blocks_downstream_grad():
    torch.manual_seed(0)
    h, y = _mock_batch(n=128)
    h = h.requires_grad_(True)
    dec = ICMLogNormalDecoder(hidden_dim=h.shape[1])

    dec.zero_grad(set_to_none=True)
    out = dec.compute_loss(h, y)
    out.main_loss.backward(retain_graph=True)

    # Downstream main loss must not update S/C heads due to detach.
    s_grad = dec.chain.head_s.weight.grad
    c_grad = dec.chain.head_c.weight.grad
    assert s_grad is None or torch.allclose(s_grad, torch.zeros_like(s_grad), atol=1e-12)
    assert c_grad is None or torch.allclose(c_grad, torch.zeros_like(c_grad), atol=1e-12)

    dec.zero_grad(set_to_none=True)
    out.total_loss.backward()
    assert dec.chain.head_s.weight.grad is not None
    assert dec.chain.head_c.weight.grad is not None


def test_lognormal_branch_nll_decreases_on_mock_data():
    torch.manual_seed(0)
    h, y = _mock_batch(n=512)
    dec = ICMLogNormalDecoder(hidden_dim=h.shape[1])
    opt = torch.optim.Adam(dec.parameters(), lr=5e-3)

    first = None
    last = None
    for _ in range(25):
        opt.zero_grad()
        out = dec.compute_loss(h, y)
        if first is None:
            first = float(out.main_loss.detach().cpu().item())
        out.total_loss.backward()
        opt.step()
        last = float(out.main_loss.detach().cpu().item())

    assert np.isfinite(first)
    assert np.isfinite(last)
    assert last < first


def test_lognormal_funding_head_has_nonzero_grad_after_first_backward():
    torch.manual_seed(0)
    h, y = _mock_batch(n=256)
    dec = ICMLogNormalDecoder(hidden_dim=h.shape[1])

    dec.zero_grad(set_to_none=True)
    out = dec.compute_loss(h, y)
    out.total_loss.backward()

    grad = dec.head_f.weight.grad
    assert grad is not None
    assert float(grad.abs().sum().detach().cpu().item()) > 0.0
