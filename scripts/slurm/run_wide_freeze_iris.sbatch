#!/bin/bash -l
#SBATCH --job-name=wide_freeze_iris
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH -c 14
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail
mkdir -p runs/slurm

# Repo root (override with REPO_ROOT env if needed)
REPO_ROOT="${REPO_ROOT:-$HOME/repo_root}"
cd "${REPO_ROOT}"

# Activate environment (micromamba preferred on iris)
if [ -x "$HOME/.local/bin/micromamba" ]; then
  eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
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
export HOST_TAG="${HOST_TAG:-iris}"

echo "WIDE_STAMP=${WIDE_STAMP}"
echo "SQLITE_DIR=${SQLITE_DIR}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "SLURM_CLUSTER_NAME=${SLURM_CLUSTER_NAME:-unknown}"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" bash scripts/run_wide_freeze_full.sh
