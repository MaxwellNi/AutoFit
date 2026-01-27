import pytest
torch = pytest.importorskip("torch")

from narrative.models.irregular_patch import IrregularPatchEmbed


def test_irregular_patch_groups_by_time_delta():
    import torch
    x = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    time_delta = torch.tensor([[0.0, 1.0, 1.0, 5.0]])
    embed = IrregularPatchEmbed(input_dim=3, d_model=4, patch_size=3.0, max_patches=None)
    patches, mask = embed(x, time_delta=time_delta)
    assert patches.shape[1] == 2
    assert mask.tolist() == [[True, True]]


def test_irregular_patch_padding():
    import torch
    x = torch.randn(1, 4, 2)
    time_delta = torch.tensor([[0.0, 1.0, 1.0, 5.0]])
    embed = IrregularPatchEmbed(input_dim=2, d_model=4, patch_size=3.0, max_patches=3)
    patches, mask = embed(x, time_delta=time_delta)
    assert patches.shape == (1, 3, 4)
    assert mask.tolist() == [[False, True, True]]


def test_irregular_patch_no_time_delta():
    import torch
    x = torch.randn(2, 5, 3)
    embed = IrregularPatchEmbed(input_dim=3, d_model=4, patch_size=3.0)
    patches, mask = embed(x, time_delta=None)
    assert patches.shape == (2, 1, 4)
    assert mask.tolist() == [[True], [True]]
