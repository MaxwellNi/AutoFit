#!/bin/bash
#SBATCH --job-name=smm_np_p63nh
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=150G
#SBATCH --time=0-12:00:00
#SBATCH --output=/work/projects/eint/logs/%x_%j.out
#SBATCH --error=/work/projects/eint/logs/%x_%j.err
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -euo pipefail
umask 002

mkdir -p /work/projects/eint/logs

_requeue_handler() {
  echo "[single-model-mainline] caught USR1, requeueing ${SLURM_JOB_ID}" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap _requeue_handler USR1

REPO_ROOT=/work/projects/eint/repo_root
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
RUN_ROOT=runs/benchmarks/single_model_mainline_localclear_20260420
LABEL=mainline_p63_neural_hazard_np_20260422

export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export BLOCK3_CANONICAL_REPO_ROOT="$REPO_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

mkdir -p "$RUN_ROOT/$LABEL"

# P6.3 Neural Hazard Gate:
# - Abolished sklearn LogisticRegression binary head
# - End-to-end _NeuralHazardMLP: 2-layer (input→32→1), sigmoid, survival NLL loss
# - Class-balanced weighting, mini-batch SGD, gradient clipping, early stopping
# - Temperature scaling only (no isotonic/platt calibrators)
# - Target: beat DeepNPTS 16/16 封锁线 on is_funded

"$INSIDER_PY" scripts/run_v740_shared112_champion_loop.py \
  --models single_model_mainline,incumbent \
  --profile audit \
  --task task1_outcome \
  --target is_funded \
  --skip-existing \
  --output-dir "$RUN_ROOT/$LABEL" \
  --summary-md "$RUN_ROOT/$LABEL/summary.md" \
  --surface-json "$RUN_ROOT/$LABEL/surface.json"
