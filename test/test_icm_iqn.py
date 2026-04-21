#!/usr/bin/env python3
import numpy as np
import torch

from src.narrative.block3.models.single_model_mainline.causal_decoders import ICMIQNDecoder


def _mock_batch(n: int = 384, d: int = 32):
    rng = np.random.default_rng(11)
    h = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    pos_mask = rng.random(n) > 0.65
    loc = 1.2 + 0.6 * h[pos_mask, 0] + 0.4 * h[pos_mask, 1]
    y[pos_mask] = np.exp(loc + 0.6 * rng.normal(size=pos_mask.sum())).astype(np.float32)
    y[np.where(pos_mask)[0][:8]] *= 15.0
    return torch.from_numpy(h), torch.from_numpy(y)


def test_iqn_branch_zero_mask_and_finite_loss():
    torch.manual_seed(1)
    h, y = _mock_batch()
    dec = ICMIQNDecoder(hidden_dim=h.shape[1], n_tau=12)

    out = dec.compute_loss(h, y)
    assert torch.isfinite(out.total_loss)
    assert out.stats["positive_fraction"] > 0.0

    y_all_zero = torch.zeros_like(y)
    out_zero = dec.compute_loss(h, y_all_zero)
    assert torch.isfinite(out_zero.total_loss)
    assert abs(out_zero.stats["main_loss"]) < 1e-8


def test_iqn_branch_lateral_detach_blocks_downstream_grad():
    torch.manual_seed(1)
    h, y = _mock_batch(n=128)
    h = h.requires_grad_(True)
    dec = ICMIQNDecoder(hidden_dim=h.shape[1], n_tau=8)

    dec.zero_grad(set_to_none=True)
    out = dec.compute_loss(h, y)
    out.main_loss.backward(retain_graph=True)

    s_grad = dec.chain.head_s.weight.grad
    c_grad = dec.chain.head_c.weight.grad
    assert s_grad is None or torch.allclose(s_grad, torch.zeros_like(s_grad), atol=1e-12)
    assert c_grad is None or torch.allclose(c_grad, torch.zeros_like(c_grad), atol=1e-12)


def test_iqn_branch_var_quantiles_are_finite():
    torch.manual_seed(1)
    h, y = _mock_batch(n=512)
    dec = ICMIQNDecoder(hidden_dim=h.shape[1], n_tau=16)
    opt = torch.optim.Adam(dec.parameters(), lr=5e-3)

    for _ in range(20):
        opt.zero_grad()
        out = dec.compute_loss(h, y)
        out.total_loss.backward()
        opt.step()

    out = dec.compute_loss(h, y)
    q90 = out.stats["q90_mean"]
    q95 = out.stats["q95_mean"]
    q99 = out.stats["q99_mean"]
    assert np.isfinite(q90)
    assert np.isfinite(q95)
    assert np.isfinite(q99)


def test_iqn_funding_head_has_nonzero_grad_after_first_backward():
    torch.manual_seed(1)
    h, y = _mock_batch(n=256)
    dec = ICMIQNDecoder(hidden_dim=h.shape[1], n_tau=8)

    dec.zero_grad(set_to_none=True)
    out = dec.compute_loss(h, y)
    out.total_loss.backward()

    grad_sum = 0.0
    for p in dec.q_head.parameters():
        if p.grad is not None:
            grad_sum += float(p.grad.abs().sum().detach().cpu().item())
    assert grad_sum > 0.0
