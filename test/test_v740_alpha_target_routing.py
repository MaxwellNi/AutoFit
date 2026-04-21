import pytest

from narrative.block3.models.v740_alpha import _torch_imports


@pytest.fixture(scope="module")
def torch_and_net():
    torch = pytest.importorskip("torch")
    _, _, net_cls = _torch_imports()
    return torch, net_cls


def _tokens(torch, batch_size: int, target_idx: int):
    task_idx = torch.zeros((batch_size,), dtype=torch.long)
    target_ids = torch.full((batch_size,), target_idx, dtype=torch.long)
    horizon_value = torch.full((batch_size,), 4.0, dtype=torch.float32)
    ablation_idx = torch.zeros((batch_size,), dtype=torch.long)
    return task_idx, target_ids, horizon_value, ablation_idx


def test_target_routing_bias_selects_distinct_experts(torch_and_net):
    torch, net_cls = torch_and_net
    torch.manual_seed(0)
    net = net_cls(
        in_channels=3,
        seq_len=12,
        horizon=4,
        hidden_dim=16,
        enable_target_routing=True,
        target_route_experts=3,
    )
    net.eval()

    with torch.no_grad():
        for param in net.target_decoder.route_logits.parameters():
            param.zero_()
        net.target_decoder.target_route_bias.weight.zero_()
        net.target_decoder.target_route_bias.weight[0, 0] = 8.0
        net.target_decoder.target_route_bias.weight[1, 1] = 8.0
        net.target_decoder.target_route_bias.weight[2, 2] = 8.0

    x = torch.randn(1, 3, 12)
    out_funding = net(x, *_tokens(torch, 1, 0))
    out_investors = net(x, *_tokens(torch, 1, 1))
    out_binary = net(x, *_tokens(torch, 1, 2))

    assert out_funding["target_route_weights"].shape == (1, 3)
    assert int(torch.argmax(out_funding["target_route_weights"], dim=1).item()) == 0
    assert int(torch.argmax(out_investors["target_route_weights"], dim=1).item()) == 1
    assert int(torch.argmax(out_binary["target_route_weights"], dim=1).item()) == 2


def test_count_anchor_head_tracks_history_level(torch_and_net):
    torch, net_cls = torch_and_net
    torch.manual_seed(0)
    net = net_cls(
        in_channels=3,
        seq_len=8,
        horizon=4,
        hidden_dim=16,
        enable_count_anchor=True,
        count_anchor_strength=0.7,
    )

    with torch.no_grad():
        for param in net.count_structure_head.parameters():
            param.zero_()

    hidden = torch.zeros((1, 16), dtype=torch.float32)
    cond = torch.zeros((1, 16), dtype=torch.float32)
    low_hist = torch.tensor([[1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]], dtype=torch.float32)
    high_hist = low_hist + 10.0

    low_out = net.count_structure_head(hidden, cond, low_hist)
    high_out = net.count_structure_head(hidden, cond, high_hist)

    assert low_out.shape == (1, 4)
    assert float(high_out.mean().detach()) > float(low_out.mean().detach())
    assert torch.all(low_out >= 0.0)


def test_count_jump_head_responds_to_recent_entry_shift(torch_and_net):
    torch, net_cls = torch_and_net
    torch.manual_seed(0)
    net = net_cls(
        in_channels=3,
        seq_len=8,
        horizon=4,
        hidden_dim=16,
        enable_count_anchor=True,
        count_anchor_strength=0.0,
        enable_count_jump=True,
        count_jump_strength=0.3,
    )

    head = net.count_structure_head
    with torch.no_grad():
        for param in head.parameters():
            param.zero_()
        jump_last_idx = 16 + 16 + 3
        head.jump_basis[2].bias.fill_(1.0)
        head.jump_gate[0].weight[0, jump_last_idx] = 1.0
        head.jump_gate[2].weight[:, 0] = 1.0
        head.jump_scale[0].weight[0, jump_last_idx] = 1.0
        head.jump_scale[2].weight[0, 0] = 1.0

    hidden = torch.zeros((1, 16), dtype=torch.float32)
    cond = torch.zeros((1, 16), dtype=torch.float32)
    flat_hist = torch.tensor([[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]], dtype=torch.float32)
    jump_hist = torch.tensor([[10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 0.0, 10.0]], dtype=torch.float32)

    flat_out = head(hidden, cond, flat_hist)
    jump_out = head(hidden, cond, jump_hist)

    assert float(jump_out.mean().detach()) > float(flat_out.mean().detach())
    assert torch.allclose(flat_out[:, :1], torch.tensor([[10.0]], dtype=torch.float32), atol=1e-4)


def test_count_sparsity_gate_suppresses_quiet_histories(torch_and_net):
    torch, net_cls = torch_and_net
    torch.manual_seed(0)
    net = net_cls(
        in_channels=3,
        seq_len=8,
        horizon=4,
        hidden_dim=16,
        enable_target_routing=False,
        enable_count_anchor=False,
        enable_count_jump=False,
        enable_count_sparsity_gate=True,
        count_sparsity_gate_strength=0.85,
    )
    net.eval()

    quiet_x = torch.zeros((1, 3, 8), dtype=torch.float32)
    active_x = quiet_x.clone()
    active_x[:, 0, :] = torch.tensor([[0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]], dtype=torch.float32)

    with torch.no_grad():
        net(quiet_x, *_tokens(torch, 1, 1))

    with torch.no_grad():
        for param in net.parameters():
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                continue
            param.zero_()
        net.count_head[2].bias.fill_(10.0)

    quiet_out = net(quiet_x, *_tokens(torch, 1, 1))
    active_out = net(active_x, *_tokens(torch, 1, 1))

    assert "count_sparsity_gate" in quiet_out
    assert float(active_out["count_sparsity_gate"].mean().detach()) > float(quiet_out["count_sparsity_gate"].mean().detach())
    assert float(active_out["count"].mean().detach()) > float(quiet_out["count"].mean().detach())