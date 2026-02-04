#!/bin/bash -l
#SBATCH --job-name=wide_freeze_aion
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
mkdir -p runs/slurm

# Repo root (override with REPO_ROOT env if needed)
REPO_ROOT="${REPO_ROOT:-/home/users/npin/repo_root}"
cd "${REPO_ROOT}"

# Activate environment (micromamba preferred)
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

# Defaults (override via sbatch --export)
export WIDE_STAMP="${WIDE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export SQLITE_DIR="${SQLITE_DIR:-${SLURM_TMPDIR:-runs/_sqlite_wide_${WIDE_STAMP}}}"
export MIN_FREE_GB="${MIN_FREE_GB:-200}"
export RAW_SAMPLE_ROWS_OFFERS="${RAW_SAMPLE_ROWS_OFFERS:-10000000}"
export RAW_SAMPLE_ROWS_EDGAR="${RAW_SAMPLE_ROWS_EDGAR:-10000000}"
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

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" "${BASH_BIN}" scripts/run_wide_freeze_full.sh
