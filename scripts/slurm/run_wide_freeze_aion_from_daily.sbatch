#!/bin/bash -l
#SBATCH --job-name=wide_freeze_aion_resume
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
mkdir -p runs/slurm

REPO_ROOT="${REPO_ROOT:-/home/users/npin/repo_root}"
cd "${REPO_ROOT}"

if [ -x "$HOME/.local/bin/micromamba" ]; then
  eval "$("$HOME/.local/bin/micromamba" shell hook -s bash)"
  micromamba activate insider
else
  source ~/.bashrc
  conda activate insider
fi

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

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

# Preflight: ensure duckdb is available (faster dedup backend)
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

export WIDE_STAMP="${WIDE_STAMP:?WIDE_STAMP is required}"
export SQLITE_DIR="${SQLITE_DIR:-${SLURM_TMPDIR:-runs/_sqlite_wide_${WIDE_STAMP}}}"
export MIN_FREE_GB="${MIN_FREE_GB:-200}"
export HOST_TAG="${HOST_TAG:-aion}"

# DuckDB limits (avoid OOM; can override via --export)
export DUCKDB_THREADS="${DUCKDB_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
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
echo "SLURM_CLUSTER_NAME=${SLURM_CLUSTER_NAME:-unknown}"
echo "DUCKDB_MEMORY_LIMIT_GB=${DUCKDB_MEMORY_LIMIT_GB:-}"
echo "DUCKDB_THREADS=${DUCKDB_THREADS}"

BASH_BIN="/usr/bin/bash"
if [ ! -x "$BASH_BIN" ]; then
  BASH_BIN="/bin/bash"
fi
if [ ! -x "$BASH_BIN" ]; then
  BASH_BIN="$(command -v bash || true)"
fi
if [ -z "${BASH_BIN}" ] || [ ! -x "${BASH_BIN}" ]; then
  echo "ERROR: bash not found on this node." >&2
  exit 2
fi

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "${BASH_BIN}" scripts/run_wide_freeze_from_daily.sh
