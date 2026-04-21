#!/bin/bash
#SBATCH --job-name=v740_repr_pa
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=150G
#SBATCH --time=0-06:00:00
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

PYTHON_BIN=${PYTHON_BIN:-/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3}
RUN_ROOT=${RUN_ROOT:-runs/benchmarks/v740_localclear_20260402}
SURFACE_JSON=${SURFACE_JSON:-${RUN_ROOT}/v740_shared112_surface_20260402.json}
BINARY_OUTPUT_DIR=${BINARY_OUTPUT_DIR:-${RUN_ROOT}/v740_shared112_binary_postaudit_20260402}
BINARY_SUMMARY_MD=${BINARY_SUMMARY_MD:-docs/references/V740_SHARED112_BINARY_POSTAUDIT_20260402.md}
INVESTORS_OUTPUT_DIR=${INVESTORS_OUTPUT_DIR:-${RUN_ROOT}/v740_shared112_investors_postaudit_20260402}
INVESTORS_SUMMARY_MD=${INVESTORS_SUMMARY_MD:-docs/references/V740_SHARED112_INVESTORS_POSTAUDIT_20260402.md}

mkdir -p "${RUN_ROOT}"

"${PYTHON_BIN}" \
  scripts/run_v740_shared112_champion_loop.py \
  --target is_funded \
  --models v740_alpha,incumbent \
  --profile quick \
  --output-dir "${BINARY_OUTPUT_DIR}" \
  --summary-md "${BINARY_SUMMARY_MD}" \
  --surface-json "${SURFACE_JSON}"

"${PYTHON_BIN}" \
  scripts/run_v740_shared112_champion_loop.py \
  --target investors_count \
  --models v740_alpha,incumbent \
  --profile quick \
  --output-dir "${INVESTORS_OUTPUT_DIR}" \
  --summary-md "${INVESTORS_SUMMARY_MD}" \
  --surface-json "${SURFACE_JSON}"