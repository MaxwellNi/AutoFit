#!/bin/bash
#SBATCH --job-name=mtcn_t2_ce
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=0-12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase15/mtcn_t2_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/mtcn_t2_ce_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue
set -euo pipefail
umask 002
cd /home/users/npin/repo_root
echo "ModernTCN freq fix | Job $SLURM_JOB_ID on $(hostname)"
echo "$(date -Iseconds) | Git: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category tslib_sota --ablation core_edgar \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task2_forecast/tslib_sota/core_edgar --seed 42 \
    --models ModernTCN
