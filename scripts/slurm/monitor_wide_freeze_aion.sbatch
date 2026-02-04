#!/bin/bash -l
#SBATCH --job-name=monitor_wide_freeze_aion
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem=4G
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
export WIDE_STAMP="${WIDE_STAMP:-20260203_225620}"
export INTERVAL_SEC="${INTERVAL_SEC:-900}"

echo "WIDE_STAMP=${WIDE_STAMP}"
echo "INTERVAL_SEC=${INTERVAL_SEC}"
echo "REPO_ROOT=${REPO_ROOT}"

bash scripts/monitor_wide_freeze_resume.sh
