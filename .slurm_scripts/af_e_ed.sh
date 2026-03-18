#!/usr/bin/env bash
#SBATCH --job-name=af_e_ed
#SBATCH --account=yves.letraon
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:volta:1
#SBATCH --output=runs/benchmarks/block3_20260203_225620_iris_full/task1_outcome/autofit/core_edgar/slurm_%j.log
#SBATCH --error=runs/benchmarks/block3_20260203_225620_iris_full/task1_outcome/autofit/core_edgar/slurm_%j.err
#SBATCH --export=ALL

set -e

# Activate environment
if [[ -f "/home/users/npin/miniforge3/etc/profile.d/micromamba.sh" ]]; then
    source /home/users/npin/miniforge3/etc/profile.d/micromamba.sh
    micromamba activate insider
elif [[ -f "/home/users/npin/miniforge3/etc/profile.d/conda.sh" ]]; then
    source /home/users/npin/miniforge3/etc/profile.d/conda.sh
    conda activate insider
fi

cd /home/users/npin/repo_root

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Task: task1_outcome | Category: autofit | Ablation: core_edgar"
echo "Python: $(which python3)"
echo "Memory limit: 256G"
echo "Start: $(date -Iseconds)"
echo "============================================================"

python3 scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category autofit \
    --ablation core_edgar \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_iris_full/task1_outcome/autofit/core_edgar \
    --seed 42 \
    --no-verify-first \
    

echo "Done: $(date -Iseconds)"
