import pytest

from narrative.block3.models.v740_alpha import V740AlphaPrototypeWrapper, _torch_imports


@pytest.fixture(scope="module")
def torch_and_net():
    torch = pytest.importorskip("torch")
    _, _, net_cls = _torch_imports()
    return torch, net_cls


def _tokens(torch, batch_size: int, target_idx: int):
    task_idx = torch.full((batch_size,), 1, dtype=torch.long)
    target_ids = torch.full((batch_size,), target_idx, dtype=torch.long)
    horizon_value = torch.full((batch_size,), 1.0, dtype=torch.float32)
    ablation_idx = torch.zeros((batch_size,), dtype=torch.long)
    return task_idx, target_ids, horizon_value, ablation_idx


def test_v741_lite_investors_forward_emits_typed_state_outputs(torch_and_net):
    torch, net_cls = torch_and_net
    torch.manual_seed(0)
    net = net_cls(
        in_channels=3,
        seq_len=12,
        horizon=4,
        hidden_dim=16,
        enable_v741_lite=True,
    )
    net.eval()

    x = torch.randn(2, 3, 12)
    out = net(x, *_tokens(torch, 2, 1))

    assert out["count"].shape == (2, 4)
    assert out["count_occurrence_logits"].shape == (2, 4)
    assert out["count_state_logits"].shape == (2, 4, 11)
    assert out["count_anchor_path"].shape == (2, 4)
    assert tuple(out["count_state_bucket_values"].shape) == (11,)


def test_v741_lite_wrapper_disables_routed_investors_regime():
    wrapper = V740AlphaPrototypeWrapper(
        enable_v741_lite=True,
        enable_target_routing=True,
        enable_count_source_routing=True,
        enable_count_hurdle_head=True,
        enable_count_source_specialists=True,
    )
    wrapper._task_name = "task2_forecast"
    wrapper._target_name = "investors_count"
    wrapper._ablation_name = "full"
    wrapper._horizon = 1

    wrapper._refresh_target_route_regime()
    wrapper._refresh_count_gate_regime()
    wrapper._refresh_count_route_regime()
    wrapper._refresh_count_specialist_regime()

    regime = wrapper.get_regime_info()

    assert regime["single_model_lite"]["enabled"] is True
    assert regime["target_routing"]["enabled"] is True
    assert regime["target_routing"]["effective_enabled"] is False
    assert regime["target_routing"]["count_head_type"] == "typed_state_delta"
    assert regime["target_routing"]["effective_count_sparsity_gate_strength"] == 0.0
    assert regime["count_source_routing"]["enabled"] is True
    assert regime["count_source_routing"]["effective_enabled"] is False
    assert regime["count_source_routing"]["effective_route_floor"] == 0.0
    assert regime["count_source_routing"]["effective_entropy_strength"] == 0.0
    assert regime["count_source_routing"]["effective_active_loss_strength"] == 0.0
    assert regime["count_source_specialists"]["enabled"] is True
    assert regime["count_source_specialists"]["effective_enabled"] is False
    assert regime["single_model_lite"]["delta_buckets"] == [-16.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0]