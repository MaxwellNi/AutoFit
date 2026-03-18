#!/bin/bash
#SBATCH --job-name=ml_t1_fu_bm
#SBATCH --account=npin
#SBATCH --partition=bigmem
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --mem=640G
#SBATCH --cpus-per-task=14
#SBATCH --output=/work/projects/eint/logs/phase15/ml_t1_fu_bm_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/ml_t1_fu_bm_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue
set -euo pipefail
umask 002
cd /home/users/npin/repo_root
echo "ML t1_fu fix3 bigmem | Job $SLURM_JOB_ID on $(hostname)"
echo "$(date -Iseconds) | Git: $(git rev-parse --short HEAD)"
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category ml_tabular --ablation full \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/ml_tabular/full --seed 42
