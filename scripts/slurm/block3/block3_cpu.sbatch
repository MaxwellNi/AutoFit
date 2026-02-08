#!/bin/bash
# Block 3 CPU SLURM Job Template
# For statistical and ml_tabular models
#
# Usage:
#   export TASK=task1_outcome CATEGORY=ml_tabular ABLATION=core_only
#   export STAMP=20260203_225620 OUTROOT=/path/to/output
#   sbatch block3_cpu.sbatch

#SBATCH --job-name=b3_cpu
#SBATCH --partition=batch
#SBATCH --qos=iris-batch-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/block3_cpu_%j.out
#SBATCH --error=logs/block3_cpu_%j.err

# ============================================================================
# Environment Setup
# ============================================================================

echo "=============================================="
echo "Block 3 Benchmark - CPU Job"
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

# Validate required env vars
if [ -z "$TASK" ] || [ -z "$CATEGORY" ] || [ -z "$ABLATION" ] || [ -z "$STAMP" ] || [ -z "$OUTROOT" ]; then
    echo "ERROR: Missing required environment variables"
    echo "Required: TASK, CATEGORY, ABLATION, STAMP, OUTROOT"
    exit 1
fi

# Load modules (adjust for your cluster)
module purge
module load lang/Python/3.11.5-GCCcore-13.2.0

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

# Run the benchmark shard
python scripts/run_block3_benchmark_shard.py \
    --preset full \
    --task "${TASK}" \
    --category "${CATEGORY}" \
    --ablation "${ABLATION}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed 42 \
    --num-workers 8 \
    ${MODELS_ARG}

EXIT_CODE=$?

echo "Finished at: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
