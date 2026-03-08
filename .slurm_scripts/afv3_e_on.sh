#!/usr/bin/env bash
#SBATCH --job-name=afv3_e_on
#SBATCH --account=yves.letraon
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:volta:1
#SBATCH --output=runs/benchmarks/block3_20260203_225620_iris_full/task1_outcome/autofit/core_only/slurm_%j.log
#SBATCH --error=runs/benchmarks/block3_20260203_225620_iris_full/task1_outcome/autofit/core_only/slurm_%j.err
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
echo "Task: task1_outcome | Category: autofit | Ablation: core_only"
echo "Variants: AutoFitV1, V2, V2E, V3, V3E, V3Max"
echo "Python: $(which python3)"
echo "Memory limit: 256G"
echo "Start: $(date -Iseconds)"
echo "nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "============================================================"

python3 scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category autofit \
    --ablation core_only \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_iris_full/task1_outcome/autofit/core_only \
    --seed 42 \
    --no-verify-first

echo "============================================================"
echo "Done: $(date -Iseconds)"
echo "Results: $(wc -c < runs/benchmarks/block3_20260203_225620_iris_full/task1_outcome/autofit/core_only/metrics.json 2>/dev/null || echo 0) bytes"
echo "============================================================"
