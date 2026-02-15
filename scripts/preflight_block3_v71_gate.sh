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
#   5) One smoke shard with AutoFitV71 and strict output audit
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
    echo "[5/5] Smoke shard execution with AutoFitV71..."
    python3 scripts/run_block3_benchmark_shard.py \
        --task task1_outcome \
        --category autofit \
        --ablation core_only \
        --models AutoFitV71 \
        --preset smoke \
        --output-dir "${SMOKE_OUT}" \
        --seed 42 \
        --max-entities 500 \
        --max-rows 120000 \
        --model-kwargs-json "${SMOKE_KW}"

    echo "[5/5] Smoke output audit..."
    python3 - << PY
import json
import math
from pathlib import Path

metrics = Path("${SMOKE_OUT}") / "metrics.json"
if not metrics.exists():
    raise RuntimeError(f"Missing metrics.json: {metrics}")

rows = json.loads(metrics.read_text(encoding="utf-8"))
if not rows:
    raise RuntimeError("Smoke produced empty metrics.")

v71_rows = [r for r in rows if r.get("model_name") == "AutoFitV71"]
if not v71_rows:
    raise RuntimeError("Smoke produced no AutoFitV71 rows.")

for i, row in enumerate(v71_rows):
    mae = float(row.get("mae", "nan"))
    if not math.isfinite(mae):
        raise RuntimeError(f"AutoFitV71 row {i} has non-finite MAE.")
    fair = row.get("fairness_pass", True)
    if fair is False:
        raise RuntimeError(f"AutoFitV71 row {i} fairness_pass=false.")
    cov = row.get("prediction_coverage_ratio", 1.0)
    try:
        cov = float(cov)
    except Exception:
        cov = 0.0
    if cov < 0.98:
        raise RuntimeError(
            f"AutoFitV71 row {i} coverage {cov:.4f} < 0.98 in smoke preflight."
        )

print(f\"Smoke audit PASS: rows={len(rows)}, AutoFitV71_rows={len(v71_rows)}\")
PY
else
    echo "[5/5] Smoke shard execution... SKIPPED"
fi

echo "================================================================"
echo "Preflight gate PASS"
echo "================================================================"
