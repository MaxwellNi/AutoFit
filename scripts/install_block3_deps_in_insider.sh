#!/usr/bin/env bash
# =============================================================================
# Install/upgrade Block 3 runtime deps in the existing insider environment only
# =============================================================================
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "insider" ]]; then
    echo "FATAL: activate insider first (current: ${CONDA_DEFAULT_ENV:-<empty>})."
    exit 2
fi

PY_BIN="$(command -v python3 || true)"
if [[ -z "$PY_BIN" || "$PY_BIN" != *"insider"* ]]; then
    echo "FATAL: python3 is not from insider env: ${PY_BIN:-<missing>}"
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO="${REPO_ROOT:-${DEFAULT_REPO}}"

if [[ ! -f "${REPO}/pyproject.toml" ]]; then
    for cand in \
        "${DEFAULT_REPO}" \
        "/home/pni/project/repo_root" \
        "/home/pni/projects/repo_root" \
        "/home/users/npin/repo_root"
    do
        if [[ -f "${cand}/pyproject.toml" ]]; then
            REPO="${cand}"
            break
        fi
    done
fi

if [[ ! -f "${REPO}/pyproject.toml" ]]; then
    echo "FATAL: cannot locate repo root. Set REPO_ROOT and retry."
    exit 2
fi

cd "${REPO}"

PACKAGES=(
  "xgboost>=3.2.0"
  "lightgbm>=4.6.0"
  "catboost>=1.2.8"
  "tabpfn>=2.0.0"
)

echo "Installing into insider env: ${CONDA_DEFAULT_ENV}"
python3 -m pip install --upgrade "${PACKAGES[@]}"

echo "Verifying imports and versions..."
python3 - << 'PY'
import importlib
pkgs = ["xgboost", "lightgbm", "catboost", "tabpfn"]
for p in pkgs:
    m = importlib.import_module(p)
    print(f"{p}={getattr(m, '__version__', 'unknown')}")
PY

echo "Done."
