"""Round-14 E13 diagnostic: NCCoPoCalibrator i.i.d. sanity test.

Goal: On i.i.d. synthetic data where exchangeability holds exactly, verify that
NCCoPoCalibrator produces test coverage == nominal (0.90 ± Monte-Carlo SE).

If this passes, the 0.685-0.870 coverage we see on real data is NOT an
implementation bug but a real exchangeability / heavy-tail violation --> paper
motivation. If this fails, the calibrator has a bug and Round-13 empirical
coverage numbers are all wrong.

Run:
  /mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/nccopo_iid_sanity.py
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "src")

from narrative.block3.models.calibration.nccopo import (
    NCCoPoCalibrator,
    NCCoPoConfig,
)


def run_single(rng: np.random.Generator, n_cal: int, n_test: int, alpha: float, noise: str):
    # y_true = signal + noise; y_pred = signal_hat = signal + sigma * epsilon_pred
    # For CP we just need residuals to be exchangeable; generate iid Gaussian / t
    if noise == "normal":
        r_cal = rng.standard_normal(n_cal)
        r_test = rng.standard_normal(n_test)
    elif noise == "t3":
        r_cal = rng.standard_t(df=3, size=n_cal)
        r_test = rng.standard_t(df=3, size=n_test)
    elif noise == "pareto":
        r_cal = rng.pareto(a=2.5, size=n_cal) * rng.choice([-1, 1], size=n_cal)
        r_test = rng.pareto(a=2.5, size=n_test) * rng.choice([-1, 1], size=n_test)
    else:
        raise ValueError(noise)

    # Construct pseudo y_pred = 0 so residuals = y_true
    y_pred_cal = np.zeros(n_cal)
    y_cal = r_cal
    y_pred_test = np.zeros(n_test)
    y_test = r_test

    cfg = NCCoPoConfig(alpha=alpha, use_studentized=False)
    cal = NCCoPoCalibrator(cfg)
    cal.fit(y_cal=y_cal, y_pred_cal=y_pred_cal, sigma_hat_cal=None)

    res = cal.predict_interval(y_pred=y_pred_test, sigma_hat=None, groups=None)
    # Handle result whether it returns tuple (lo, hi) or dict with coverage
    if isinstance(res, tuple) and len(res) == 2:
        lo, hi = res
    elif isinstance(res, dict):
        lo = np.asarray(res.get("lower"))
        hi = np.asarray(res.get("upper"))
    else:
        # Fallback: assume object with .lower / .upper
        lo = np.asarray(getattr(res, "lower"))
        hi = np.asarray(getattr(res, "upper"))

    covered = ((y_test >= lo) & (y_test <= hi)).mean()
    width = float(np.mean(hi - lo))
    return covered, width


def main():
    rng = np.random.default_rng(42)
    header = f"{'noise':>8s} {'n_cal':>6s} {'n_test':>7s} {'alpha':>6s} {'cov':>8s} {'width':>10s} {'verdict':>14s}"
    print(header)
    print("-" * len(header))
    cases = [
        ("normal", 2000, 2000, 0.10),
        ("normal", 5000, 5000, 0.10),
        ("normal", 2000, 2000, 0.20),
        ("t3",     2000, 2000, 0.10),
        ("t3",     5000, 5000, 0.10),
        ("pareto", 2000, 2000, 0.10),
        ("pareto", 5000, 5000, 0.10),
    ]
    rows = []
    for noise, nc, nt, alpha in cases:
        # Run 20 replicates and average
        covs, widths = [], []
        for _ in range(20):
            c, w = run_single(rng, nc, nt, alpha, noise)
            covs.append(c); widths.append(w)
        cov_mean = float(np.mean(covs))
        cov_sd = float(np.std(covs, ddof=1))
        w_mean = float(np.mean(widths))
        nominal = 1 - alpha
        # MC-SE bound: nominal should be within ~3 SD
        within = abs(cov_mean - nominal) <= 3 * cov_sd + 0.005
        verdict = "PASS" if within else "FAIL"
        rows.append((noise, nc, nt, alpha, cov_mean, cov_sd, w_mean, verdict))
        print(f"{noise:>8s} {nc:>6d} {nt:>7d} {alpha:>6.2f} {cov_mean:>7.4f}±{cov_sd:.3f} {w_mean:>10.4f} {verdict:>14s}")
    any_fail = any(r[-1] == "FAIL" for r in rows)
    print("\nOverall:", "FAIL" if any_fail else "PASS")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
