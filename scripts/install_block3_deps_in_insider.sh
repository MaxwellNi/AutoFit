#!/usr/bin/env bash
# =============================================================================
# Auto-repair runtime dependencies in insider environment (3090/4090/Iris)
#
# Behavior:
#   1) Auto-activate insider (reuse current env if already active)
#   2) Detect missing/outdated core runtime dependencies
#   3) Auto-install only required fixes (unless --force)
#   4) Verify critical imports and versions
#
# Usage:
#   bash scripts/install_block3_deps_in_insider.sh
#   bash scripts/install_block3_deps_in_insider.sh --dry-run
#   bash scripts/install_block3_deps_in_insider.sh --force
# =============================================================================
set -euo pipefail

FORCE=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --force)
            FORCE=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

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

activate_insider_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "insider" ]]; then
        return 0
    fi

    if command -v micromamba >/dev/null 2>&1; then
        local roots=()
        if [[ -n "${MAMBA_ROOT_PREFIX:-}" ]]; then
            roots+=("${MAMBA_ROOT_PREFIX}")
        fi
        roots+=(
            "/mnt/aiongpfs/projects/eint/envs/.micromamba"
            "${HOME}/.local/share/micromamba"
            "${HOME}/micromamba"
        )
        local r
        for r in "${roots[@]}"; do
            [[ -d "${r}" ]] || continue
            export MAMBA_ROOT_PREFIX="${r}"
            eval "$(micromamba shell hook -s bash)"
            if micromamba activate insider; then
                return 0
            fi
        done
    fi

    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1090
            source "${conda_base}/etc/profile.d/conda.sh"
            if conda activate insider; then
                return 0
            fi
        fi
    fi

    echo "FATAL: failed to activate insider environment."
    return 1
}

activate_insider_env

detect_insider_prefix() {
    if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_PREFIX}" == *"insider"* && -d "${CONDA_PREFIX}" ]]; then
        echo "${CONDA_PREFIX}"
        return 0
    fi
    local candidates=(
        "/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
        "${HOME}/miniforge3/envs/insider"
        "${HOME}/mambaforge/envs/insider"
        "${HOME}/anaconda3/envs/insider"
        "${HOME}/miniconda3/envs/insider"
    )
    local c
    for c in "${candidates[@]}"; do
        if [[ -d "${c}" ]]; then
            echo "${c}"
            return 0
        fi
    done
    return 1
}

repair_insider_bin_permissions() {
    local prefix="$1"
    local bindir="${prefix}/bin"
    [[ -d "${bindir}" ]] || return 0
    local repaired=0
    local exe
    for exe in python python3 python3.11 python3.12 pip pip3 pytest; do
        local path="${bindir}/${exe}"
        if [[ -f "${path}" && ! -x "${path}" ]]; then
            chmod u+x "${path}"
            repaired=$((repaired + 1))
        fi
    done
    if [[ "${repaired}" -gt 0 ]]; then
        echo "Repaired execute permissions for ${repaired} insider runtime executables."
    fi
}

INSIDER_PREFIX="$(detect_insider_prefix || true)"
if [[ -n "${INSIDER_PREFIX}" ]]; then
    repair_insider_bin_permissions "${INSIDER_PREFIX}"
    export PATH="${INSIDER_PREFIX}/bin:${PATH}"
fi

PY_BIN="$(command -v python3 || true)"
if [[ -z "${PY_BIN}" || "${PY_BIN}" != *"insider"* ]]; then
    echo "FATAL: python3 is not from insider env: ${PY_BIN:-<missing>}"
    exit 2
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(
        f"FATAL: python >=3.11 required, got {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
print(f"Runtime python OK: {sys.executable} ({sys.version.split()[0]})")
PY

cd "${REPO}"

# Ensure packaging is available for robust version checks
if ! python3 - << 'PY' >/dev/null 2>&1
from packaging.version import Version
print(Version("1.0"))
PY
then
    if $DRY_RUN; then
        echo "[DRY] Would install: packaging>=24.0"
    else
        python3 -m pip install --upgrade "packaging>=24.0"
    fi
fi

mapfile -t NEED_SPECS < <(python3 - << 'PY'
from __future__ import annotations
import importlib
import importlib.metadata as im
from packaging.version import Version

# dist_name, import_name, min_version, max_exclusive(optional), pip_spec, exact_version(optional)
REQS = [
    ("numpy", "numpy", "1.24.0", None, "numpy>=1.24.0", None),
    ("pandas", "pandas", "2.0.0", None, "pandas>=2.0.0", None),
    ("pyarrow", "pyarrow", "12.0.0", None, "pyarrow>=12.0.0", None),
    ("PyYAML", "yaml", "6.0.0", None, "PyYAML>=6.0.0", None),
    # Keep sklearn in [1.6, 1.8) to satisfy Chronos (>=1.6) and TabPFN (<1.8).
    ("scikit-learn", "sklearn", "1.6.0", "1.8.0", "scikit-learn>=1.6.0,<1.8", None),
    ("lightgbm", "lightgbm", "4.6.0", None, "lightgbm>=4.6.0", None),
    ("xgboost", "xgboost", "3.2.0", None, "xgboost>=3.2.0", None),
    ("catboost", "catboost", "1.2.8", None, "catboost>=1.2.8", None),
    ("tabpfn", "tabpfn", "2.0.0", None, "tabpfn>=2.0.0", None),
    ("pytest", "pytest", "8.0.0", None, "pytest>=8.0.0", None),
]

need = []
for dist_name, import_name, min_ver, max_exclusive, pip_spec, exact_ver in REQS:
    try:
        cur = im.version(dist_name)
    except Exception:
        need.append(pip_spec)
        continue

    try:
        if Version(cur) < Version(min_ver):
            need.append(pip_spec)
            continue
    except Exception:
        need.append(pip_spec)
        continue

    if max_exclusive is not None and Version(cur) >= Version(max_exclusive):
        need.append(pip_spec)
        continue

    if exact_ver is not None and cur != exact_ver:
        need.append(pip_spec)
        continue

    # Import smoke for ABI/runtime mismatch
    try:
        importlib.import_module(import_name)
    except Exception:
        need.append(pip_spec)

# stable dedupe
seen = set()
for spec in need:
    if spec not in seen:
        seen.add(spec)
        print(spec)
PY
)

if $FORCE; then
    NEED_SPECS=(
        "numpy>=1.24.0"
        "pandas>=2.0.0"
        "pyarrow>=12.0.0"
        "PyYAML>=6.0.0"
        "scikit-learn>=1.6.0,<2"
        "lightgbm>=4.6.0"
        "xgboost>=3.2.0"
        "catboost>=1.2.8"
        "tabpfn>=2.0.0"
        "pytest>=8.0.0"
    )
fi

if [[ "${#NEED_SPECS[@]}" -eq 0 ]]; then
    echo "Dependency check PASS: insider environment already compatible."
else
    echo "Dependencies to fix (${#NEED_SPECS[@]}):"
    printf '  - %s\n' "${NEED_SPECS[@]}"
    if $DRY_RUN; then
        echo "[DRY] Skipping installation."
        echo "Dry-run complete."
        exit 0
    else
        python3 -m pip install --upgrade "${NEED_SPECS[@]}"
    fi
fi

# Final verification
python3 - << 'PY'
from __future__ import annotations
import importlib
import importlib.metadata as im

checks = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("PyYAML", "yaml"),
    ("scikit-learn", "sklearn"),
    ("lightgbm", "lightgbm"),
    ("xgboost", "xgboost"),
    ("catboost", "catboost"),
    ("tabpfn", "tabpfn"),
    ("pytest", "pytest"),
]

print("Final dependency versions:")
for dist_name, import_name in checks:
    mod = importlib.import_module(import_name)
    ver = im.version(dist_name)
    print(f"  {dist_name}={ver}")

print("Dependency verification PASS")
PY

echo "Done."
