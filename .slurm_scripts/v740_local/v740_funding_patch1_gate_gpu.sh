#!/bin/bash
#SBATCH --job-name=v740_fnd_g1
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

PYTHON_BIN=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
RUN_ROOT=runs/benchmarks/v740_localclear_20260401
GATE_ROOT="${RUN_ROOT}/v740_funding_patch1_gate_20260401"
SURFACE_JSON="${RUN_ROOT}/v740_shared112_surface_20260401.json"

mkdir -p "${GATE_ROOT}"

run_cell() {
  local task="$1"
  local ablation="$2"
  local target="$3"
  local horizon="$4"
  local slug="$5"

  echo "[v740-gate] running ${slug}" >&2
  "${PYTHON_BIN}" \
    scripts/run_v740_shared112_champion_loop.py \
    --task "${task}" \
    --ablation "${ablation}" \
    --target "${target}" \
    --horizon "${horizon}" \
    --max-cells 1 \
    --models v740_alpha,incumbent \
    --profile quick \
    --skip-existing \
    --output-dir "${GATE_ROOT}/${slug}" \
    --summary-md "${GATE_ROOT}/${slug}.md" \
    --surface-json "${SURFACE_JSON}"
}

run_cell task2_forecast full funding_raised_usd 7 t2_full_funding_h7
run_cell task2_forecast full funding_raised_usd 30 t2_full_funding_h30
run_cell task2_forecast core_edgar funding_raised_usd 30 t2_core_edgar_funding_h30
run_cell task2_forecast core_only funding_raised_usd 30 t2_core_only_funding_h30
run_cell task1_outcome full is_funded 30 t1_full_is_funded_h30_guard