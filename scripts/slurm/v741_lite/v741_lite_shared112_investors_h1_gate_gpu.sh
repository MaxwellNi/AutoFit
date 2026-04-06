#!/bin/bash
#SBATCH --job-name=v741_lite_invh1
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=150G
#SBATCH --time=0-08:00:00
#SBATCH --output=/work/projects/eint/logs/v741_lite/%x_%j.out
#SBATCH --error=/work/projects/eint/logs/v741_lite/%x_%j.err
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -euo pipefail
umask 002

REPO_ROOT=/home/users/npin/repo_root
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
RUN_ROOT=runs/benchmarks/v741_lite_20260406
LABEL=v741_lite_shared112_investors_h1_gate_20260406

mkdir -p /work/projects/eint/logs/v741_lite

_requeue_handler() {
  echo "[v741-lite] caught USR1, requeueing ${SLURM_JOB_ID}" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap _requeue_handler USR1

cd "$REPO_ROOT"

export BLOCK3_CANONICAL_REPO_ROOT="$REPO_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_ROOT"

"$INSIDER_PY" scripts/run_v740_shared112_champion_loop.py \
  --models v741_lite,incumbent \
  --profile quick \
  --target investors_count \
  --horizon 1 \
  --max-cells 12 \
  --skip-existing \
  --output-dir "$RUN_ROOT/$LABEL" \
  --summary-md "docs/references/V741_LITE_SHARED112_INVESTORS_H1_GATE_20260406.md" \
  --surface-json "$RUN_ROOT/${LABEL}_surface.json"