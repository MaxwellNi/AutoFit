#!/bin/bash
#SBATCH --job-name=v740_fnd_g4
#SBATCH --account=npin
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

PYTHON_BIN=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
RUN_ROOT=runs/benchmarks/v740_localclear_20260401
DUEL_ROOT="${RUN_ROOT}/v740_funding_bestbranch_duel_20260402"
SURFACE_JSON="${RUN_ROOT}/v740_shared112_surface_20260401.json"

mkdir -p "${DUEL_ROOT}"

run_branch() {
  local branch="$1"
  shift
  local -a extra_args=("$@")

  echo "[v740-bestbranch] branch=${branch}" >&2
  "${PYTHON_BIN}" \
    scripts/run_v740_shared112_champion_loop.py \
    --target funding_raised_usd \
    --models v740_alpha,incumbent \
    --profile quick \
    --skip-existing \
    --output-dir "${DUEL_ROOT}/${branch}" \
    --summary-md "${DUEL_ROOT}/${branch}.md" \
    --surface-json "${SURFACE_JSON}" \
    "${extra_args[@]}"
}

run_branch anchor_only_no_log_a085 \
  --disable-funding-log-domain \
  --disable-funding-source-scaling \
  --funding-anchor-strength 0.85

run_branch scale_anchor_no_log_a085 \
  --disable-funding-log-domain \
  --funding-anchor-strength 0.85