#!/usr/bin/env bash
#SBATCH --job-name=p3_aut_t1o_edgar
#SBATCH --account=yves.letraon
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:volta:1
#SBATCH --output=runs/benchmarks/block3_20260203_225620_iris_phase3/task1_outcome/autofit/core_edgar/slurm_%j.log
#SBATCH --error=runs/benchmarks/block3_20260203_225620_iris_phase3/task1_outcome/autofit/core_edgar/slurm_%j.err
#SBATCH --export=ALL
#SBATCH --signal=B:TERM@180

set -e
mkdir -p runs/benchmarks/block3_20260203_225620_iris_phase3/task1_outcome/autofit/core_edgar

# ── Robust micromamba activation ──
# Method 1: Use MAMBA_EXE + shell hook (works on all ULHPC nodes)
export MAMBA_EXE="/mnt/aiongpfs/users/npin/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba"
if [[ -x "$MAMBA_EXE" ]]; then
    eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
    micromamba activate insider
fi

# Verify Python is from insider env
PYTHON=$(which python3 2>/dev/null || echo "/usr/bin/python3")
if [[ "$PYTHON" != *"insider"* ]]; then
    echo "FATAL: Python not from insider env: $PYTHON" >&2
    exit 1
fi

cd /home/users/npin/repo_root

echo "============================================================"
echo "Phase 3 — Job ${SLURM_JOB_ID} on $(hostname)"
echo "Task: task1_outcome | Category: autofit | Ablation: core_edgar"
echo "Python: $PYTHON"
echo "Memory: 256G | Walltime: 48:00:00"
echo "Start: $(date -Iseconds)"
echo "Git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "============================================================"

$PYTHON scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category autofit \
    --ablation core_edgar \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_iris_phase3/task1_outcome/autofit/core_edgar \
    --seed 42 \
    --no-verify-first

echo "Done: $(date -Iseconds)"
