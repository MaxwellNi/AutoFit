import pytest
torch = pytest.importorskip("torch")

import numpy as np
from pathlib import Path

from narrative.auto_fit.budget_search import successive_halving


def test_successive_halving_exports_best_config(tmp_path: Path):
    X = np.random.randn(20, 5, 4).astype(np.float32)
    y = np.random.randn(20).astype(np.float32)
    candidates = [
        {"backbone": "dlinear", "fusion_type": "none", "module_flags": {"nonstat": False, "multiscale": False, "ssm": False}},
        {"backbone": "patchtst", "fusion_type": "none", "module_flags": {"nonstat": False, "multiscale": False, "ssm": False}, "model_cfg": {"patch_len": 2}},
    ]
    results, best = successive_halving(
        candidates,
        X,
        y,
        seq_len=5,
        enc_in=4,
        device="cpu",
        budgets=(1,),
        output_dir=tmp_path,
        early_stopping_patience=1,
    )
    assert (tmp_path / "best_config.yaml").exists()
    assert len(results) >= 1
