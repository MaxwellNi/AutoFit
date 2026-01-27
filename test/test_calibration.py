import pytest
import numpy as np

from narrative.nbi_nci.calibration import (
    temperature_scale,
    reliability_metrics,
    export_reliability_metrics,
)


def test_reliability_metrics_and_export(tmp_path):
    logits = np.array([0.0, 1.0, -1.0, 2.0])
    probs = 1 / (1 + np.exp(-logits))
    labels = np.array([0, 1, 0, 1])
    metrics = reliability_metrics(probs, labels, n_bins=5)
    assert "ece" in metrics
    out_path = export_reliability_metrics(metrics, tmp_path / "calibration.json")
    assert out_path.exists()
