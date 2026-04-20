#!/bin/bash
#SBATCH --job-name=smm_battle2_trunk
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=150G
#SBATCH --time=0-04:00:00
#SBATCH --output=/work/projects/eint/logs/%x_%j.out
#SBATCH --error=/work/projects/eint/logs/%x_%j.err
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -euo pipefail
umask 002
mkdir -p /work/projects/eint/logs

_requeue_handler() {
  echo "[battle2-trunk] caught USR1, requeueing ${SLURM_JOB_ID}" >&2
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
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

echo "=== Battle 2: Trunk Ablation Audit ==="
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-none}"

"$INSIDER_PY" scripts/run_battle2_trunk_ablation.py \
  --ablations core_only core_edgar \
  --horizons 1 7 14 30 \
  --output-dir runs/benchmarks/single_model_mainline_localclear_20260420/battle2_trunk_ablation

echo "=== Battle 2 DONE ==="
