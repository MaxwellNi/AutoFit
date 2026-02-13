#!/usr/bin/env bash
#SBATCH --job-name=p7_af2_t1_fu
#SBATCH --account=yves.letraon
#SBATCH --partition=batch
#SBATCH --qos=iris-batch-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=112G
#SBATCH --cpus-per-task=28

#SBATCH --output=/work/projects/eint/logs/phase7/p7_af2_t1_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7/p7_af2_t1_fu_%j.err
#SBATCH --export=ALL
#SBATCH --signal=B:USR1@120


set -e

# Activate micromamba environment
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(which python3)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

echo "Task: task1_outcome | Category: autofit | Ablation: full"
echo "Models: AutoFitV3Max,AutoFitV4,AutoFitV5,AutoFitV6,AutoFitV7"
echo "Preset: full | Seed: 42"
echo "Output: runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/full"
echo "============================================================"

python3 scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category autofit \
    --ablation full \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/full \
    --seed 42 \
    --no-verify-first \
    --models AutoFitV3Max,AutoFitV4,AutoFitV5,AutoFitV6,AutoFitV7

echo "Done: $(date -Iseconds)"
