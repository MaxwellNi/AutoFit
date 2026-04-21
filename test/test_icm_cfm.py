#!/usr/bin/env python3
import numpy as np
import torch

from src.narrative.block3.models.single_model_mainline.causal_decoders import ICMCFMDecoder


def _mock_batch(n: int = 320, d: int = 32):
    rng = np.random.default_rng(23)
    h = rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    pos_mask = rng.random(n) > 0.68
    loc = 1.0 + 0.7 * h[pos_mask, 0] + 0.2 * h[pos_mask, 2]
    y[pos_mask] = np.exp(loc + 0.7 * rng.normal(size=pos_mask.sum())).astype(np.float32)
    y[np.where(pos_mask)[0][:10]] *= 12.0
    return torch.from_numpy(h), torch.from_numpy(y)


def test_cfm_branch_zero_mask_and_finite_loss():
    torch.manual_seed(2)
    h, y = _mock_batch()
    dec = ICMCFMDecoder(hidden_dim=h.shape[1])

    out = dec.compute_loss(h, y)
    assert torch.isfinite(out.total_loss)
    assert out.stats["positive_fraction"] > 0.0

    y_all_zero = torch.zeros_like(y)
    out_zero = dec.compute_loss(h, y_all_zero)
    assert torch.isfinite(out_zero.total_loss)
    assert abs(out_zero.stats["main_loss"]) < 1e-8


def test_cfm_branch_lateral_detach_blocks_downstream_grad():
    torch.manual_seed(2)
    h, y = _mock_batch(n=128)
    h = h.requires_grad_(True)
    dec = ICMCFMDecoder(hidden_dim=h.shape[1])

    dec.zero_grad(set_to_none=True)
    out = dec.compute_loss(h, y)
    out.main_loss.backward(retain_graph=True)

    s_grad = dec.chain.head_s.weight.grad
    c_grad = dec.chain.head_c.weight.grad
    assert s_grad is None or torch.allclose(s_grad, torch.zeros_like(s_grad), atol=1e-12)
    assert c_grad is None or torch.allclose(c_grad, torch.zeros_like(c_grad), atol=1e-12)


def test_cfm_branch_training_and_euler_sampling_are_finite():
    torch.manual_seed(2)
    h, y = _mock_batch(n=512)
    dec = ICMCFMDecoder(hidden_dim=h.shape[1])
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

    samples = dec.sample_euler(h[:128], steps=24)
    assert np.isfinite(first)
    assert np.isfinite(last)
    assert last < first
    assert torch.isfinite(samples).all()
    assert torch.all(samples >= 0.0)


def test_cfm_funding_head_has_nonzero_grad_after_first_backward():
    torch.manual_seed(2)
    h, y = _mock_batch(n=256)
    dec = ICMCFMDecoder(hidden_dim=h.shape[1])

    dec.zero_grad(set_to_none=True)
    out = dec.compute_loss(h, y)
    out.total_loss.backward()

    grad_sum = 0.0
    for p in dec.vfield.parameters():
        if p.grad is not None:
            grad_sum += float(p.grad.abs().sum().detach().cpu().item())
    assert grad_sum > 0.0
