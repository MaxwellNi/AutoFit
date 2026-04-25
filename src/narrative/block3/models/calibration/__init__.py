"""Block-3 calibration submodule.

Hosts the NC-CoPo (Neural Compound-Poisson Conformal) calibrator that
delivers split-conformal finite-sample coverage for the random-sum
response Y = sum_{i=1}^{N(x)} M_i(x) with neural Poisson lambda_theta and
neural GEV severity g_theta. See
`docs/NCCOPO_THEOREM_DRAFT_20260423.md` for the Theorem 1 statement and
6-item verification plan.
"""

from .nccopo import NCCoPoCalibrator, NCCoPoFitResult, NCCoPoConfig

__all__ = ["NCCoPoCalibrator", "NCCoPoFitResult", "NCCoPoConfig"]
