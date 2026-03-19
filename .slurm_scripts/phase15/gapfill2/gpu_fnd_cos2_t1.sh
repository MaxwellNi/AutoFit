#!/bin/bash
#SBATCH --job-name=gpu_fnd_cos2_t1
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase15/gpu_fnd_cos2_t1_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/gpu_fnd_cos2_t1_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue
set -euo pipefail
umask 002
cd /home/users/npin/repo_root
echo "Foundation seed2 ${SLURM_JOB_NAME} | Job $SLURM_JOB_ID on $(hostname)"
echo "$(date -Iseconds) | Git: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category foundation --ablation core_only_seed2 \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/foundation/core_only_seed2 --seed 42 \
    --models Chronos2,TTM
