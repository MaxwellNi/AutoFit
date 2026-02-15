#!/usr/bin/env bash
# ============================================================================
# Block 3 AutoFit V7.1 preflight gate (mandatory before large submissions)
# ============================================================================
#
# Enforces a verify-then-run workflow:
#   1) Freeze verification
#   2) Phase-7 code audit verification
#   3) AutoFitV71 factory sanity for all known grid variants
#   4) V7.1 leakage/fairness/reproducibility tests
#   5) Synthetic smoke fit/predict with strict coverage/routing audit
#
# Usage:
#   bash scripts/preflight_block3_v71_gate.sh
#   bash scripts/preflight_block3_v71_gate.sh --v71-variant g02
#   bash scripts/preflight_block3_v71_gate.sh --skip-pytest
# ============================================================================

set -euo pipefail

REPO="/home/users/npin/repo_root"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
V71_VARIANT="g02"
SKIP_PYTEST=false
SKIP_SMOKE=false

for arg in "$@"; do
    case "$arg" in
        --run-tag=*)
            RUN_TAG="${arg#*=}"
            ;;
        --v71-variant=*)
            V71_VARIANT="${arg#*=}"
            ;;
        --skip-pytest)
            SKIP_PYTEST=true
            ;;
        --skip-smoke)
            SKIP_SMOKE=true
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

declare -A V71_VARIANTS
V71_VARIANTS["g01"]='{"top_k":8,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":true,"enable_regime_retrieval":true}'
V71_VARIANTS["g02"]='{"top_k":10,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":true,"enable_regime_retrieval":true}'
V71_VARIANTS["g03"]='{"top_k":6,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":true,"enable_regime_retrieval":true}'
V71_VARIANTS["g04"]='{"top_k":8,"min_ensemble_size_heavy_tail":3,"dynamic_weighting":true,"enable_regime_retrieval":false}'
V71_VARIANTS["g05"]='{"top_k":8,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":false,"enable_regime_retrieval":true}'

if [[ -z "${V71_VARIANTS[$V71_VARIANT]:-}" ]]; then
    echo "Unknown --v71-variant=${V71_VARIANT}. Available: g01,g02,g03,g04,g05"
    exit 1
fi

SMOKE_OUT="runs/benchmarks/block3_preflight_v71_${RUN_TAG}"
SMOKE_KW="{\"AutoFitV71\":${V71_VARIANTS[$V71_VARIANT]}}"

echo "================================================================"
echo "Block3 V7.1 preflight gate"
echo "Run tag:    ${RUN_TAG}"
echo "Variant:    ${V71_VARIANT}"
echo "Smoke out:  ${SMOKE_OUT}"
echo "Skip pytest:${SKIP_PYTEST}"
echo "Skip smoke: ${SKIP_SMOKE}"
echo "================================================================"

export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

echo "[1/5] Freeze verification..."
python3 scripts/block3_verify_freeze.py

echo "[2/5] Phase-7 fix verification..."
python3 scripts/verify_phase7_fixes.py

echo "[3/5] AutoFitV71 factory sanity on all known variants..."
python3 - << 'PY'
import json
from narrative.block3.models.registry import get_model

variants = {
    "g01": {"top_k": 8, "min_ensemble_size_heavy_tail": 2, "dynamic_weighting": True, "enable_regime_retrieval": True},
    "g02": {"top_k": 10, "min_ensemble_size_heavy_tail": 2, "dynamic_weighting": True, "enable_regime_retrieval": True},
    "g03": {"top_k": 6, "min_ensemble_size_heavy_tail": 2, "dynamic_weighting": True, "enable_regime_retrieval": True},
    "g04": {"top_k": 8, "min_ensemble_size_heavy_tail": 3, "dynamic_weighting": True, "enable_regime_retrieval": False},
    "g05": {"top_k": 8, "min_ensemble_size_heavy_tail": 2, "dynamic_weighting": False, "enable_regime_retrieval": True},
}

for name, kwargs in variants.items():
    model = get_model("AutoFitV71", **kwargs)
    if model is None:
        raise RuntimeError(f"Variant {name} failed: model factory returned None")
print("AutoFitV71 variant factory sanity PASS")
PY

if ! $SKIP_PYTEST; then
    echo "[4/5] Targeted V7.1 tests..."
    python3 -m pytest -q \
        test/test_autofit_v71_no_leakage.py \
        test/test_autofit_v71_coverage_guard.py \
        test/test_autofit_v71_objective_switch.py \
        test/test_autofit_v71_reproducibility.py
else
    echo "[4/5] Targeted V7.1 tests... SKIPPED"
fi

if ! $SKIP_SMOKE; then
    echo "[5/5] Synthetic smoke execution with AutoFitV71..."
    CUDA_VISIBLE_DEVICES="" python3 - << PY
import json
import numpy as np
import pandas as pd

from narrative.block3.models.registry import get_model

smoke_kw = json.loads('''${SMOKE_KW}''')
v71_kw = smoke_kw.get("AutoFitV71", {})
rng = np.random.RandomState(42)

n = 1400
X = pd.DataFrame(
    {
        "f1": rng.normal(size=n),
        "f2": rng.lognormal(mean=0.3, sigma=0.7, size=n),
        "f3": rng.randint(0, 15, size=n).astype(float),
        "f4": rng.normal(loc=2.0, scale=3.0, size=n),
        "f5": rng.uniform(0.0, 1.0, size=n),
    }
)
X.loc[X.index % 9 == 0, "f2"] = np.nan
X.loc[X.index % 11 == 0, "f4"] = np.nan

targets = {
    "count": pd.Series(rng.poisson(lam=6.0 + 0.5 * np.abs(X["f1"].fillna(0)), size=n), name="investors_count"),
    "heavy_tail": pd.Series(
        rng.lognormal(mean=2.0 + 0.1 * X["f5"].fillna(0), sigma=1.1, size=n),
        name="funding_raised_usd",
    ),
}

for lane_name, y in targets.items():
    model = get_model("AutoFitV71", **v71_kw)
    model.fit(X, y, target=y.name, horizon=7)
    X_test = X.iloc[-280:].copy()
    pred = np.asarray(model.predict(X_test, target=y.name, horizon=7), dtype=float)

    if len(pred) != len(X_test):
        raise RuntimeError(f"{lane_name}: prediction length mismatch {len(pred)} != {len(X_test)}")

    coverage = float(np.isfinite(pred).mean())
    if coverage < 0.98:
        raise RuntimeError(f"{lane_name}: coverage {coverage:.4f} < 0.98")

    if lane_name == "count":
        if (pred < 0).any():
            raise RuntimeError("count lane produced negative predictions")
        if not np.allclose(pred, np.rint(pred), rtol=0.0, atol=1e-8):
            raise RuntimeError("count lane predictions are not integer-rounded")

    routing = model.get_routing_info()
    for key in [
        "lane_selected",
        "dynamic_thresholds",
        "meta_objective",
        "ensemble_diversity_stats",
        "regime_retrieval_enabled",
        "prediction_postprocess",
    ]:
        if key not in routing:
            raise RuntimeError(f"missing routing key: {key}")

    print(
        f"Synthetic smoke PASS [{lane_name}] "
        f"coverage={coverage:.4f} lane={routing.get('lane_selected')} "
        f"models={routing.get('final_selected')}"
    )
PY
else
    echo "[5/5] Smoke shard execution... SKIPPED"
fi

echo "================================================================"
echo "Preflight gate PASS"
echo "================================================================"
