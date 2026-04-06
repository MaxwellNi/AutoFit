#!/bin/bash
#SBATCH --job-name=v740_112_invh1_cf
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=besteffort
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

export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/users/npin/repo_root
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
RUN_ROOT="$REPO_ROOT/runs/benchmarks/v740_localclear_cfisch_20260404"

mkdir -p "$RUN_ROOT"
cd "$REPO_ROOT"

"$INSIDER_PY" scripts/run_v740_shared112_champion_loop.py \
  --target investors_count \
  --horizon 1 \
  --max-cells 12 \
  --models v740_alpha,incumbent \
  --profile quick \
  --target-route-experts 3 \
  --count-anchor-strength 0.70 \
  --count-jump-strength 0.30 \
  --skip-existing \
  --output-dir "$RUN_ROOT/v740_shared112_investors_routed_h1_probe_cfisch_20260404" \
  --summary-md "$RUN_ROOT/v740_shared112_investors_routed_h1_probe_cfisch_20260404.md" \
  --surface-json "$RUN_ROOT/v740_shared112_surface_cfisch_20260404.json"