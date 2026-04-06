#!/bin/bash
#SBATCH --job-name=af739_t1_s2_bin
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=4-00:00:00
#SBATCH --mem=280G
#SBATCH --cpus-per-task=11
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase15/af739_t1_s2_bin_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/af739_t1_s2_bin_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue
set -euo pipefail
_requeue_handler() { echo "$(date -Iseconds) USR1: timeout approaching, requeuing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; }
trap _requeue_handler USR1
umask 002
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
REPO_ROOT=/home/users/npin/repo_root
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
cd "$REPO_ROOT"
echo "AF739 ${SLURM_JOB_NAME} | Job $SLURM_JOB_ID on $(hostname)"
echo "$(date -Iseconds) | Git: $(git rev-parse --short HEAD)"
"$INSIDER_PY" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category autofit --ablation core_only_seed2 \
    --preset full --target-filter is_funded \
    --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/core_only_seed2 --seed 42