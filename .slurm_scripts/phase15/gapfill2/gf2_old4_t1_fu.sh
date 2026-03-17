#!/bin/bash
#SBATCH --job-name=gf2_old4_t1_fu
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase15/gf2_old4_t1_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/gf2_old4_t1_fu_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue
set -euo pipefail
umask 002
cd /home/users/npin/repo_root
echo "GF2 old4 ${SLURM_JOB_NAME} | Job $SLURM_JOB_ID on $(hostname)"
echo "$(date -Iseconds) | Git: $(git rev-parse --short HEAD)"
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category tslib_sota --ablation full \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/tslib_sota/full --seed 42 \
    --models CATS,FITS,KANAD,WPMixer
