#!/bin/bash
###############################################################################
# Block 3 Benchmark Launcher — Tasks 1/2/3 on Iris (NO TEXT ablations)
#
# Matrix: 3 tasks × 2 ablations × 6 categories = 36 shards
# Ablations: core_only, core_edgar  (skip core_text, full)
# Partition: batch (CPU) for ml_tabular/statistical
#            gpu   (V100) for deep_classical/transformer_sota/foundation/irregular
#
# Usage:
#   bash scripts/launch_iris_b3_tasks.sh [preset]
#   preset: smoke | quick | standard | full  (default: standard)
###############################################################################
set -euo pipefail

PRESET="${1:-standard}"
STAMP="20260203_225620"
REPO="/home/users/npin/repo_root"
OUTBASE="${REPO}/runs/benchmarks/block3_${STAMP}_iris"
SLURMLOG="${REPO}/runs/slurm"
MAMBA_EXE="/mnt/aiongpfs/users/npin/.local/bin/micromamba"

mkdir -p "${SLURMLOG}" "${OUTBASE}"

# Tasks and ablations (NO text)
TASKS="task1_outcome task2_forecast task3_risk_adjust"
ABLATIONS="core_only core_edgar"

# Category → partition mapping
CPU_CATS="statistical ml_tabular"
GPU_CATS="deep_classical transformer_sota foundation irregular"

echo "============================================================"
echo "Block 3 Benchmark Launch — Iris HPC"
echo "  Stamp:     ${STAMP}"
echo "  Preset:    ${PRESET}"
echo "  Output:    ${OUTBASE}"
echo "  Tasks:     ${TASKS}"
echo "  Ablations: ${ABLATIONS} (text skipped)"
echo "  CPU cats:  ${CPU_CATS}"
echo "  GPU cats:  ${GPU_CATS}"
echo "============================================================"

# Time limits by preset
case "${PRESET}" in
  smoke)    CPU_TIME="00:30:00"; GPU_TIME="01:00:00" ;;
  quick)    CPU_TIME="02:00:00"; GPU_TIME="04:00:00" ;;
  standard) CPU_TIME="06:00:00"; GPU_TIME="12:00:00" ;;
  full)     CPU_TIME="12:00:00"; GPU_TIME="24:00:00" ;;
  *)        echo "Unknown preset: ${PRESET}"; exit 1 ;;
esac

TOTAL_SHARDS=0
SUBMITTED_JOBS=""

submit_shard() {
    local TASK="$1"
    local CAT="$2"
    local ABL="$3"
    local PARTITION="$4"
    local TIMELIM="$5"
    local EXTRA_SBATCH="${6:-}"

    local OUTDIR="${OUTBASE}/${TASK}/${CAT}/${ABL}"
    local JOBNAME="b3_${TASK}_${CAT}_${ABL}"
    # Truncate job name to 30 chars for SLURM
    JOBNAME="${JOBNAME:0:30}"

    local SBATCH_SCRIPT=$(mktemp /tmp/b3_shard_XXXXXX.sbatch)
    cat > "${SBATCH_SCRIPT}" << HEREDOC
#!/bin/bash -l
#SBATCH --job-name=${JOBNAME}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time=${TIMELIM}
#SBATCH --output=${SLURMLOG}/${JOBNAME}_%j.out
#SBATCH --error=${SLURMLOG}/${JOBNAME}_%j.err
${EXTRA_SBATCH}
set -euo pipefail

echo "=== Block 3 Shard ==="
echo "Task:      ${TASK}"
echo "Category:  ${CAT}"
echo "Ablation:  ${ABL}"
echo "Partition: ${PARTITION}"
echo "Preset:    ${PRESET}"
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname -s)"
echo "Started:   \$(date -Iseconds)"
echo "====================="

cd ${REPO}

python scripts/run_block3_benchmark_shard.py \\
    --task ${TASK} \\
    --category ${CAT} \\
    --ablation ${ABL} \\
    --preset ${PRESET} \\
    --output-dir "${OUTDIR}" \\
    --no-verify-first \\
    --seed 42

echo "Shard completed: \$(date -Iseconds)"
HEREDOC

    local JOB_ID
    JOB_ID=$(sbatch "${SBATCH_SCRIPT}" 2>&1 | grep -oP '\d+$')
    echo "  Submitted: ${JOBNAME} → Job ${JOB_ID} (${PARTITION}, ${TIMELIM})"
    SUBMITTED_JOBS="${SUBMITTED_JOBS} ${JOB_ID}"
    TOTAL_SHARDS=$((TOTAL_SHARDS + 1))
    rm -f "${SBATCH_SCRIPT}"
}

# ============================================================
# Submit all shards
# ============================================================
for TASK in ${TASKS}; do
    echo ""
    echo "--- ${TASK} ---"

    for ABL in ${ABLATIONS}; do
        # CPU categories → batch partition
        for CAT in ${CPU_CATS}; do
            submit_shard "${TASK}" "${CAT}" "${ABL}" "batch" "${CPU_TIME}"
        done

        # GPU categories → gpu partition with 1 GPU
        for CAT in ${GPU_CATS}; do
            submit_shard "${TASK}" "${CAT}" "${ABL}" "gpu" "${GPU_TIME}" \
                "#SBATCH --gres=gpu:1"
        done
    done
done

echo ""
echo "============================================================"
echo "Total shards submitted: ${TOTAL_SHARDS}"
echo "Job IDs: ${SUBMITTED_JOBS}"
echo "Monitor: squeue -u \$USER"
echo "Output:  ${OUTBASE}"
echo "============================================================"
