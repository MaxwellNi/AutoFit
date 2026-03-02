#!/bin/bash
# Full Block 3 Benchmark Schedule for ULHPC Iris
#
# This script provides the complete execution plan for all Task1-3 benchmarks.
# It submits jobs in a controlled manner to avoid overwhelming the cluster.
#
# Usage:
#   ./full_benchmark_schedule.sh [preset] [dry-run]
#
# Example:
#   ./full_benchmark_schedule.sh standard        # Standard run (all data)
#   ./full_benchmark_schedule.sh quick           # Quick run (500 entities)
#   ./full_benchmark_schedule.sh standard dry    # Show jobs without submitting

set -e

PRESET="${1:-standard}"
DRY_RUN="${2:-}"

# Configuration
REPO_ROOT="${REPO_ROOT:-/scratch/users/${USER}/narrative/repo_root}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/users/${USER}/narrative/runs/benchmarks/block3_${PRESET}_$(date +%Y%m%d_%H%M%S)}"
SLURM_DIR="${REPO_ROOT}/scripts/slurm/block3"

# Tasks (NO task4 per requirements)
TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")

# Categories by compute type
CPU_CATEGORIES=("statistical" "ml_tabular")
GPU_CATEGORIES=("deep_classical" "transformer_sota" "foundation" "irregular")

# Ablations (core_only for fast iteration, full for final results)
ABLATIONS=("core_only" "full")

# Job tracking
declare -a JOB_IDS=()

echo "=============================================="
echo "Block 3 Full Benchmark Schedule"
echo "=============================================="
echo "Preset:      ${PRESET}"
echo "Output:      ${OUTPUT_BASE}"
echo "Repo:        ${REPO_ROOT}"
echo "=============================================="

# Function to submit a job
submit_job() {
    local SBATCH_FILE=$1
    local TASK=$2
    local CATEGORY=$3
    local ABLATION=$4
    local OUTPUT_DIR="${OUTPUT_BASE}/${TASK}/${CATEGORY}/${ABLATION}"
    
    if [ -n "$DRY_RUN" ]; then
        echo "[DRY-RUN] Would submit: ${TASK} / ${CATEGORY} / ${ABLATION}"
        return
    fi
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Submit job
    JOB_ID=$(sbatch \
        --export=TASK=${TASK},CATEGORY=${CATEGORY},ABLATION=${ABLATION},PRESET=${PRESET},OUTPUT_DIR=${OUTPUT_DIR},REPO_ROOT=${REPO_ROOT} \
        --output="${OUTPUT_DIR}/slurm-%j.out" \
        --error="${OUTPUT_DIR}/slurm-%j.err" \
        --job-name="b3_${TASK:0:2}_${CATEGORY:0:3}_${ABLATION:0:4}" \
        "${SBATCH_FILE}" | awk '{print $4}')
    
    JOB_IDS+=("${JOB_ID}")
    echo "[SUBMITTED] Job ${JOB_ID}: ${TASK} / ${CATEGORY} / ${ABLATION}"
}

# Phase 1: CPU jobs (statistical + ml_tabular)
echo ""
echo "Phase 1: CPU Jobs (Statistical + ML Tabular)"
echo "=============================================="
for TASK in "${TASKS[@]}"; do
    for CATEGORY in "${CPU_CATEGORIES[@]}"; do
        for ABLATION in "${ABLATIONS[@]}"; do
            submit_job "${SLURM_DIR}/block3_cpu.sbatch" "$TASK" "$CATEGORY" "$ABLATION"
        done
    done
done

CPU_COUNT=$((${#TASKS[@]} * ${#CPU_CATEGORIES[@]} * ${#ABLATIONS[@]}))
echo "Total CPU jobs: ${CPU_COUNT}"

# Phase 2: GPU jobs (deep + transformer + foundation + irregular)
echo ""
echo "Phase 2: GPU Jobs (Deep + Transformer + Foundation + Irregular)"
echo "=============================================="
for TASK in "${TASKS[@]}"; do
    for CATEGORY in "${GPU_CATEGORIES[@]}"; do
        for ABLATION in "${ABLATIONS[@]}"; do
            submit_job "${SLURM_DIR}/block3_gpu.sbatch" "$TASK" "$CATEGORY" "$ABLATION"
        done
    done
done

GPU_COUNT=$((${#TASKS[@]} * ${#GPU_CATEGORIES[@]} * ${#ABLATIONS[@]}))
echo "Total GPU jobs: ${GPU_COUNT}"

# Summary
echo ""
echo "=============================================="
echo "Submission Summary"
echo "=============================================="
echo "Total jobs submitted: $((CPU_COUNT + GPU_COUNT))"
echo "  CPU jobs: ${CPU_COUNT}"
echo "  GPU jobs: ${GPU_COUNT}"

if [ -z "$DRY_RUN" ]; then
    echo ""
    echo "Job IDs:"
    printf '%s\n' "${JOB_IDS[@]}" | column
    
    echo ""
    echo "Monitor with:"
    echo "  squeue -u ${USER}"
    echo "  sacct -j $(IFS=,; echo "${JOB_IDS[*]}")"
    
    # Save job manifest
    MANIFEST="${OUTPUT_BASE}/jobs_manifest.txt"
    mkdir -p "${OUTPUT_BASE}"
    {
        echo "# Block 3 Benchmark Job Manifest"
        echo "# Generated: $(date -Iseconds)"
        echo "# Preset: ${PRESET}"
        echo "# Total jobs: $((CPU_COUNT + GPU_COUNT))"
        echo ""
        for i in "${!JOB_IDS[@]}"; do
            echo "${JOB_IDS[$i]}"
        done
    } > "${MANIFEST}"
    echo ""
    echo "Job manifest saved to: ${MANIFEST}"
fi

echo ""
echo "When all jobs complete, consolidate results with:"
echo "  python ${REPO_ROOT}/scripts/consolidate_block3_results.py \\"
echo "      --input-dir ${OUTPUT_BASE} \\"
echo "      --output-dir ${OUTPUT_BASE}/paper_tables"
