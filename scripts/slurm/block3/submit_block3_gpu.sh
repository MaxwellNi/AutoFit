#!/bin/bash
# Submit GPU jobs for Block 3 benchmark (deep + transformer + foundation)
#
# Usage:
#   ./submit_block3_gpu.sh [STAMP] [OUTROOT]
#
# Example:
#   ./submit_block3_gpu.sh 20260203_225620 /scratch/user/block3_results
#
# Submits jobs for:
#   - Tasks: task1_outcome, task2_forecast, task3_risk_adjust
#   - Categories: deep_classical, transformer_sota, foundation, irregular
#   - Ablations: core_only, full

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${1:-20260203_225620}"
OUTROOT="${2:-runs/benchmarks/block3_${STAMP}}"

echo "=============================================="
echo "Block 3 GPU Job Submission"
echo "=============================================="
echo "Stamp: ${STAMP}"
echo "Output Root: ${OUTROOT}"
echo "=============================================="

# Create output directories
mkdir -p "${OUTROOT}"
mkdir -p logs

# GPU categories
GPU_CATEGORIES="deep_classical transformer_sota foundation irregular"

# Tasks
TASKS="task1_outcome task2_forecast task3_risk_adjust"

# Ablations
ABLATIONS="core_only full"

JOB_COUNT=0

for TASK in ${TASKS}; do
    for CATEGORY in ${GPU_CATEGORIES}; do
        for ABLATION in ${ABLATIONS}; do
            echo "Submitting: ${TASK} / ${CATEGORY} / ${ABLATION}"
            
            export TASK CATEGORY ABLATION STAMP OUTROOT
            
            JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/block3_gpu.sbatch")
            
            echo "  Job ID: ${JOB_ID}"
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo "=============================================="
echo "Submitted ${JOB_COUNT} GPU jobs"
echo "Monitor with: squeue -u \$USER"
echo "=============================================="
