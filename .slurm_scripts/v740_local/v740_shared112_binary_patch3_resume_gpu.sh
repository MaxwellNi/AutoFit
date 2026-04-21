#!/bin/bash
#SBATCH --job-name=v740_bin_p3r
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=150G
#SBATCH --time=0-08:00:00
#SBATCH --output=/work/projects/eint/logs/v740_local/%x_%j.out
#SBATCH --error=/work/projects/eint/logs/v740_local/%x_%j.err
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -euo pipefail
umask 002

mkdir -p /work/projects/eint/logs/v740_local

_requeue_handler() {
  echo "[v740-local] caught USR1, requeueing ${SLURM_JOB_ID}" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap _requeue_handler USR1

cd /home/users/npin/repo_root

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

RUN_ROOT=runs/benchmarks/v740_localclear_20260401

mkdir -p "${RUN_ROOT}"

/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 \
  scripts/run_v740_shared112_champion_loop.py \
  --target is_funded \
  --models v740_alpha,incumbent \
  --profile quick \
  --skip-existing \
  --output-dir "${RUN_ROOT}/v740_shared112_binary_patch3_16_20260401" \
  --summary-md docs/references/V740_SHARED112_BINARY_PATCH3_16_20260401.md \
  --surface-json "${RUN_ROOT}/v740_shared112_surface_20260401.json"