#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

export WIDE_STAMP="${WIDE_STAMP:?WIDE_STAMP is required}"
export SQLITE_DIR="${SQLITE_DIR:-${REPO_ROOT}/runs/_sqlite_wide_${WIDE_STAMP}}"
export MIN_FREE_GB="${MIN_FREE_GB:-200}"
export HOST_TAG="${HOST_TAG:-4090}"

NUM_THREADS="${NUM_THREADS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${NUM_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${NUM_THREADS}}"

# DuckDB limits (avoid OOM; override via env)
export DUCKDB_THREADS="${DUCKDB_THREADS:-${NUM_THREADS}}"
if [ -z "${DUCKDB_MEMORY_LIMIT_GB:-}" ] && [ -r /proc/meminfo ]; then
  MEM_KB="$(awk '/MemTotal/ {print $2; exit}' /proc/meminfo)"
  if [ -n "${MEM_KB}" ]; then
    MEM_GB=$(( MEM_KB / 1024 / 1024 ))
    if [ "${MEM_GB}" -gt 0 ]; then
      export DUCKDB_MEMORY_LIMIT_GB=$(( MEM_GB * 75 / 100 ))
    fi
  fi
fi

echo "WIDE_STAMP=${WIDE_STAMP}"
echo "SQLITE_DIR=${SQLITE_DIR}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "HOST_TAG=${HOST_TAG}"
echo "DUCKDB_MEMORY_LIMIT_GB=${DUCKDB_MEMORY_LIMIT_GB:-}"
echo "DUCKDB_THREADS=${DUCKDB_THREADS}"

# Preflight: ensure deltalake is available
if ! python - <<'PY' >/dev/null 2>&1
import deltalake  # noqa: F401
print("deltalake_ok", deltalake.__version__)
PY
then
  echo "deltalake missing; attempting install in active env..."
  if command -v micromamba >/dev/null 2>&1; then
    micromamba install -y -c conda-forge deltalake
  else
    python -m pip install deltalake
  fi
  python - <<'PY'
import deltalake
print("deltalake_ok", deltalake.__version__)
PY
fi

# Preflight: ensure duckdb is available
if ! python - <<'PY' >/dev/null 2>&1
import duckdb  # noqa: F401
print("duckdb_ok", duckdb.__version__)
PY
then
  echo "duckdb missing; attempting install in active env..."
  if command -v micromamba >/dev/null 2>&1; then
    micromamba install -y -c conda-forge duckdb
  else
    python -m pip install duckdb
  fi
  python - <<'PY'
import duckdb
print("duckdb_ok", duckdb.__version__)
PY
fi

bash scripts/run_wide_freeze_from_daily.sh
