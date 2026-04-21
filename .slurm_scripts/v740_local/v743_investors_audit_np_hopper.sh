#!/bin/bash
#SBATCH --job-name=v743_np_inv
#SBATCH --account=npin
#SBATCH --partition=hopper
#SBATCH --qos=besteffort
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=189G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/projects/eint/logs/v743_local/%x_%j.out
#SBATCH --error=/work/projects/eint/logs/v743_local/%x_%j.err
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -euo pipefail
umask 002

mkdir -p /work/projects/eint/logs/v743_local

_requeue_handler() {
  echo "[v743-local] caught USR1, requeueing ${SLURM_JOB_ID}" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap _requeue_handler USR1

REPO_ROOT=/work/projects/eint/repo_root
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3

export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export BLOCK3_CANONICAL_REPO_ROOT="$REPO_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

RUN_ROOT=runs/benchmarks/v743_localclear_20260407
OUT_DIR="$RUN_ROOT/v743_shared112_investors_audit_np_20260407"
mkdir -p "$OUT_DIR"

"$INSIDER_PY" scripts/run_v740_shared112_champion_loop.py \
  --target investors_count \
  --models v743_factorized,incumbent \
  --profile audit \
  --skip-existing \
  --output-dir "$OUT_DIR" \
  --summary-md "$OUT_DIR/summary.md" \
  --surface-json "$OUT_DIR/surface.json"