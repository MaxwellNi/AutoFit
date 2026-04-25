"""Verify Theorem 1 of NC-CoPo on a compound-Poisson + Gumbel-severity synthetic.

This is the Level-2 empirical backbone required for the Oral story of
the Round-12 novelty plan.  We draw

    Y = sum_{i=1}^{N(x)} M_i(x),
    N(x) ~ Poisson(lambda(x)),
    M_i(x) ~ Gumbel(mu(x), beta(x))  i.i.d. given x,

with x ~ Uniform[-2, 2]^d.  A purposely mis-specified plug-in predictor

    y_pred(x) = lambda_hat(x) * E[M|x]   with plug-in noise,

feeds the NC-CoPo split-conformal calibrator.  We then check that the
marginal coverage of the emitted prediction interval stays within
|empirical_cov - (1 - alpha)| <= O(1/sqrt(n_cal)) across many
independent replications — Theorem 1's finite-sample marginal guarantee.

Also reports:
  * Studentized vs un-studentized interval width (sharpness).
  * Mondrian per-slice coverage (stability under known covariate shift).
  * Heavy-tail GEV xi=0.4 stress run (approaches the xi<=1/2-delta
    constraint; Level-3 DRO lower-bounds the coverage loss).

Usage:
    python scripts/verify_nccopo_gumbel.py \
        --alpha 0.1 --reps 200 --n_cal 1000 --n_test 1000

Outputs:
  * stdout summary table,
  * `runs/verifications/nccopo_gumbel/<ts>/summary.json`,
  * `runs/verifications/nccopo_gumbel/<ts>/coverage_alpha.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Allow import without full narrative pkg side effects.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "nccopo",
    str(REPO / "src/narrative/block3/models/calibration/nccopo.py"),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["nccopo"] = _mod
_spec.loader.exec_module(_mod)
NCCoPoCalibrator = _mod.NCCoPoCalibrator
NCCoPoConfig = _mod.NCCoPoConfig


# ---------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------


def _lambda(x: np.ndarray) -> np.ndarray:
    # Heterogeneous positive intensity, range ~ [0.3, 8].
    return np.exp(0.7 * x[:, 0] + 0.3 * np.sin(2.0 * x[:, 1])) + 0.3


def _severity_params(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = 1.0 + 0.5 * x[:, 2]
    beta = 0.4 + 0.1 * np.abs(x[:, 3])
    return mu, beta


def _gumbel_mean(mu: np.ndarray, beta: np.ndarray) -> np.ndarray:
    euler = 0.5772156649015329
    return mu + beta * euler


def _gumbel_var(beta: np.ndarray) -> np.ndarray:
    return (np.pi ** 2) * (beta ** 2) / 6.0


def draw_compound_poisson(
    rng: np.random.Generator, x: np.ndarray, xi: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, y_pred, sigma_hat, groups).

    xi=0 gives Gumbel severity (Theorem 1 nominal).  xi>0 gives
    Fréchet-type GEV severity for heavy-tail stress (Theorem 1 boundary).
    xi must be strictly less than 0.5 for the second moment to exist.
    """

    lam = _lambda(x)
    n = rng.poisson(lam)
    mu, beta = _severity_params(x)
    y = np.zeros(x.shape[0])
    if xi <= 0.0:
        for i, ni in enumerate(n):
            if ni:
                y[i] = rng.gumbel(mu[i], beta[i], size=ni).sum()
    else:
        # GEV inverse-CDF sample: M = mu + beta * ((-log U)^(-xi) - 1) / xi
        for i, ni in enumerate(n):
            if ni:
                u = rng.uniform(1e-9, 1.0 - 1e-9, size=ni)
                sev = mu[i] + beta[i] * (((-np.log(u)) ** (-xi)) - 1.0) / xi
                y[i] = sev.sum()

    lam_hat = np.clip(lam + rng.normal(0, 0.15, size=x.shape[0]), 0.05, None)
    mu_hat = mu + rng.normal(0, 0.10, size=x.shape[0])
    beta_hat = np.clip(beta + rng.normal(0, 0.05, size=x.shape[0]), 0.05, None)
    # First-two-moment closed forms for GEV(mu, beta, xi) with xi<1/2.
    from math import gamma as _g

    if xi <= 0.0:
        E_M = _gumbel_mean(mu_hat, beta_hat)
        V_M = _gumbel_var(beta_hat)
    else:
        g1 = _g(1.0 - xi)
        g2 = _g(1.0 - 2.0 * xi)
        E_M = mu_hat + beta_hat * (g1 - 1.0) / xi
        V_M = (beta_hat ** 2) * (g2 - g1 ** 2) / (xi ** 2)
    y_pred = lam_hat * E_M
    sigma_hat = np.sqrt(np.clip(lam_hat * (V_M + E_M ** 2), 1e-8, None))

    deciles = np.quantile(lam_hat, np.linspace(0.1, 0.9, 9))
    groups = np.searchsorted(deciles, lam_hat)
    return y, y_pred, sigma_hat, groups.astype(int)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


@dataclass
class RepResult:
    alpha: float
    cov_studentized: float
    cov_absolute: float
    width_studentized: float
    width_absolute: float
    mondrian_min_cov: float
    mondrian_max_cov: float
    n_cal: int
    n_test: int


def _one_rep(
    rng: np.random.Generator,
    alpha: float,
    n_cal: int,
    n_test: int,
    d: int = 4,
    xi: float = 0.0,
) -> RepResult:
    x_cal = rng.uniform(-2.0, 2.0, size=(n_cal, d))
    x_test = rng.uniform(-2.0, 2.0, size=(n_test, d))
    y_cal, yp_cal, s_cal, g_cal = draw_compound_poisson(rng, x_cal, xi=xi)
    y_test, yp_test, s_test, g_test = draw_compound_poisson(rng, x_test, xi=xi)

    # studentized NC-CoPo
    cal_s = NCCoPoCalibrator(NCCoPoConfig(alpha=alpha, use_studentized=True))
    cal_s.fit(y_cal, yp_cal, sigma_hat_cal=s_cal)
    lo_s, hi_s = cal_s.predict_interval(yp_test, sigma_hat=s_test)
    cov_s = float(np.mean((y_test >= lo_s) & (y_test <= hi_s)))
    width_s = float(np.mean(hi_s - lo_s))

    # absolute-residual baseline
    cal_a = NCCoPoCalibrator(NCCoPoConfig(alpha=alpha, use_studentized=False))
    cal_a.fit(y_cal, yp_cal)
    lo_a, hi_a = cal_a.predict_interval(yp_test)
    cov_a = float(np.mean((y_test >= lo_a) & (y_test <= hi_a)))
    width_a = float(np.mean(hi_a - lo_a))

    # Mondrian studentized
    cal_m = NCCoPoCalibrator(
        NCCoPoConfig(alpha=alpha, use_studentized=True, mondrian_groups=g_cal)
    )
    cal_m.fit(y_cal, yp_cal, sigma_hat_cal=s_cal)
    lo_m, hi_m = cal_m.predict_interval(yp_test, sigma_hat=s_test, groups=g_test)
    per_group_cov = []
    for g in np.unique(g_test):
        mask = g_test == g
        if int(mask.sum()) >= 20:
            per_group_cov.append(
                float(np.mean((y_test[mask] >= lo_m[mask]) & (y_test[mask] <= hi_m[mask])))
            )
    mondrian_min = float(np.min(per_group_cov)) if per_group_cov else float("nan")
    mondrian_max = float(np.max(per_group_cov)) if per_group_cov else float("nan")

    return RepResult(
        alpha=alpha,
        cov_studentized=cov_s,
        cov_absolute=cov_a,
        width_studentized=width_s,
        width_absolute=width_a,
        mondrian_min_cov=mondrian_min,
        mondrian_max_cov=mondrian_max,
        n_cal=n_cal,
        n_test=n_test,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--n_cal", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260424)
    p.add_argument("--out_root", default=str(REPO / "runs" / "verifications" / "nccopo_gumbel"))
    p.add_argument("--xi", type=float, default=0.0, help="GEV shape for heavy-tail stress (xi<0.5)")
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(args.seed)
    per_alpha_rows = []
    summary = {}
    for alpha in args.alpha:
        reps = []
        for r in range(args.reps):
            rng = np.random.default_rng(rng_master.integers(0, 2 ** 31 - 1))
            reps.append(_one_rep(rng, alpha, args.n_cal, args.n_test, xi=args.xi))
        arr_cov_s = np.array([r.cov_studentized for r in reps])
        arr_cov_a = np.array([r.cov_absolute for r in reps])
        arr_w_s = np.array([r.width_studentized for r in reps])
        arr_w_a = np.array([r.width_absolute for r in reps])
        arr_m_min = np.array([r.mondrian_min_cov for r in reps])
        arr_m_max = np.array([r.mondrian_max_cov for r in reps])

        target = 1.0 - alpha
        sr = {
            "alpha": alpha,
            "target_cov": target,
            "studentized_mean_cov": float(arr_cov_s.mean()),
            "studentized_std_cov": float(arr_cov_s.std(ddof=1)),
            "absolute_mean_cov": float(arr_cov_a.mean()),
            "absolute_std_cov": float(arr_cov_a.std(ddof=1)),
            "studentized_mean_width": float(arr_w_s.mean()),
            "absolute_mean_width": float(arr_w_a.mean()),
            "width_ratio_student_over_abs": float(arr_w_s.mean() / max(arr_w_a.mean(), 1e-9)),
            "mondrian_mean_min_cov": float(np.nanmean(arr_m_min)),
            "mondrian_mean_max_cov": float(np.nanmean(arr_m_max)),
            "mondrian_mean_spread": float(np.nanmean(arr_m_max - arr_m_min)),
            "n_cal": args.n_cal,
            "n_test": args.n_test,
            "reps": args.reps,
        }
        per_alpha_rows.append(sr)
        summary[f"alpha_{alpha}"] = sr
        print(
            f"[alpha={alpha:.2f}] target={target:.2f} "
            f"stud_cov={sr['studentized_mean_cov']:.4f}±{sr['studentized_std_cov']:.4f} "
            f"abs_cov={sr['absolute_mean_cov']:.4f}±{sr['absolute_std_cov']:.4f} "
            f"w_ratio={sr['width_ratio_student_over_abs']:.3f} "
            f"mondrian_spread={sr['mondrian_mean_spread']:.4f}"
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    import csv

    with (out_dir / "coverage_alpha.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(per_alpha_rows[0].keys()))
        w.writeheader()
        for row in per_alpha_rows:
            w.writerow(row)

    # Theorem 1 pass criterion: |mean_cov - (1-alpha)| <= 3 / sqrt(n_test).
    tol = 3.0 / np.sqrt(args.n_test)
    failed = [
        a
        for a in per_alpha_rows
        if abs(a["studentized_mean_cov"] - a["target_cov"]) > tol
    ]
    if failed:
        print(f"[FAIL] Theorem 1 marginal coverage test failed at alpha={[f['alpha'] for f in failed]}")
        return 2
    print(f"[PASS] Theorem 1 marginal coverage within ±{tol:.4f} for all alpha; output={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
