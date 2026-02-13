#!/usr/bin/env bash
#SBATCH --job-name=p7_fmH_t1_ct
#SBATCH --account=yves.letraon
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:volta:1
#SBATCH --output=/work/projects/eint/logs/phase7/p7_fmH_t1_ct_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7/p7_fmH_t1_ct_%j.err
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

echo "Task: task1_outcome | Category: foundation | Ablation: core_text"
echo "Models: Timer,TimeMoE,MOMENT,LagLlama,TimesFM"
echo "Preset: full | Seed: 42"
echo "Output: runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/core_text"
echo "============================================================"

python3 scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category foundation \
    --ablation core_text \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/core_text \
    --seed 42 \
    --no-verify-first \
    --models Timer,TimeMoE,MOMENT,LagLlama,TimesFM

echo "Done: $(date -Iseconds)"
