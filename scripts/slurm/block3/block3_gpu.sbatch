#!/bin/bash
# Block 3 GPU SLURM Job Template
# For deep_classical, transformer_sota, foundation, and irregular models
#
# Usage:
#   export TASK=task1_outcome CATEGORY=transformer_sota ABLATION=core_only
#   export STAMP=20260203_225620 OUTROOT=/path/to/output
#   sbatch block3_gpu.sbatch

#SBATCH --job-name=b3_gpu
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/block3_gpu_%j.out
#SBATCH --error=logs/block3_gpu_%j.err

# ============================================================================
# Environment Setup
# ============================================================================

echo "=============================================="
echo "Block 3 Benchmark - GPU Job"
echo "=============================================="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "Task: ${TASK}"
echo "Category: ${CATEGORY}"
echo "Ablation: ${ABLATION}"
echo "Stamp: ${STAMP}"
echo "Output Root: ${OUTROOT}"
echo "Models: ${MODELS:-all}"
echo "=============================================="

# Show GPU info
nvidia-smi || echo "nvidia-smi not available"

# Validate required env vars
if [ -z "$TASK" ] || [ -z "$CATEGORY" ] || [ -z "$ABLATION" ] || [ -z "$STAMP" ] || [ -z "$OUTROOT" ]; then
    echo "ERROR: Missing required environment variables"
    echo "Required: TASK, CATEGORY, ABLATION, STAMP, OUTROOT"
    exit 1
fi

# Load modules (adjust for your cluster)
module purge
module load lang/Python/3.11.5-GCCcore-13.2.0
module load system/CUDA/12.1.1

# Activate conda environment
source ~/.bashrc
conda activate insider

# Change to repo directory
cd ~/projects/repo_root

# Create log directory
mkdir -p logs

# ============================================================================
# Run Benchmark Shard
# ============================================================================

OUTPUT_DIR="${OUTROOT}/${TASK}/${CATEGORY}/${ABLATION}/${SLURM_JOB_ID}"

echo "Output directory: ${OUTPUT_DIR}"
echo "Starting at: $(date)"

# Build models argument if specified
MODELS_ARG=""
if [ -n "$MODELS" ]; then
    MODELS_ARG="--models ${MODELS}"
fi

# Set CUDA options
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run the benchmark shard
python scripts.py \
    --preset full \
    --task "${TASK}" \
    --category "${CATEGORY}" \
    --ablation "${ABLATION}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed 42 \
    --num-workers 4 \
    ${MODELS_ARG}

EXIT_CODE=$?

echo "Finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
