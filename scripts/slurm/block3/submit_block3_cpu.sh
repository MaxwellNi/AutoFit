#!/bin/bash
# Submit CPU jobs for Block 3 benchmark (statistical + ml_tabular)
#
# Usage:
#   ./submit_block3_cpu.sh [STAMP] [OUTROOT]
#
# Example:
#   ./submit_block3_cpu.sh 20260203_225620 /scratch/user/block3_results
#
# Submits jobs for:
#   - Tasks: task1_outcome, task2_forecast, task3_risk_adjust
#   - Categories: statistical, ml_tabular
#   - Ablations: core_only, full

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${1:-20260203_225620}"
OUTROOT="${2:-runs/benchmarks/block3_${STAMP}}"

echo "=============================================="
echo "Block 3 CPU Job Submission"
echo "=============================================="
echo "Stamp: ${STAMP}"
echo "Output Root: ${OUTROOT}"
echo "=============================================="

# Create output directories
mkdir -p "${OUTROOT}"
mkdir -p logs

# CPU categories
CPU_CATEGORIES="statistical ml_tabular"

# Tasks
TASKS="task1_outcome task2_forecast task3_risk_adjust"

# Ablations
ABLATIONS="core_only full"

JOB_COUNT=0

for TASK in ${TASKS}; do
    for CATEGORY in ${CPU_CATEGORIES}; do
        for ABLATION in ${ABLATIONS}; do
            echo "Submitting: ${TASK} / ${CATEGORY} / ${ABLATION}"
            
            export TASK CATEGORY ABLATION STAMP OUTROOT
            
            JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/block3_cpu.sbatch")
            
            echo "  Job ID: ${JOB_ID}"
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo "=============================================="
echo "Submitted ${JOB_COUNT} CPU jobs"
echo "Monitor with: squeue -u \$USER"
echo "=============================================="
