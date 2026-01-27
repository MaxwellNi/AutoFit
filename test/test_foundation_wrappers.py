import pytest
torch = pytest.importorskip("torch")

from narrative.models.foundation_wrappers import ChronosWrapper, LagLlamaWrapper, MoiraiWrapper


@pytest.mark.parametrize("wrapper_cls", [ChronosWrapper, LagLlamaWrapper, MoiraiWrapper])
def test_foundation_wrapper_predict(wrapper_cls):
    import torch
    x = torch.randn(2, 5, 4)
    wrapper = wrapper_cls(device="cpu", input_dim=4)
    wrapper.load(horizon=3)
    y = wrapper.predict(x, horizon=3)
    assert y.shape == (2, 3)
