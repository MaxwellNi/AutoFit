#!/bin/bash
###############################################################################
# Block 3 — RELAUNCH NON-TABULAR CATEGORIES (predict() fix applied)
#
# Context: Commit 5c83f2a fixed per-entity predict() for all non-tabular
# models. All prior non-tabular results had CONSTANT predictions (unique=1,
# std=0) due to np.full(h, scalar) in predict().
#
# This script:
# 1. Clears completed MANIFEST.json for non-tabular shards (so they re-run)
# 2. Resubmits 30 shards: 5 categories × 3 tasks × 2 ablations
#    - statistical (CPU/batch)
#    - deep_classical (GPU)
#    - transformer_sota (GPU)
#    - foundation (GPU)
#    - irregular (GPU)
# 3. Does NOT touch ml_tabular shards (still running, already correct)
#
# Usage:
#   bash scripts/relaunch_non_tabular_fix.sh
###############################################################################
set -euo pipefail

PRESET="full"
STAMP="20260203_225620"
REPO="/home/users/npin/repo_root"
OUTBASE="${REPO}/runs/benchmarks/block3_${STAMP}_iris_full"
SLURMLOG="${REPO}/runs/slurm"
MAMBA_EXE="/mnt/aiongpfs/users/npin/.local/bin/micromamba"

mkdir -p "${SLURMLOG}" "${OUTBASE}"

GIT_HASH=$(cd "${REPO}" && git rev-parse --short HEAD)
echo "Git hash: ${GIT_HASH}"

TASKS="task1_outcome task2_forecast task3_risk_adjust"
ABLATIONS="core_only core_edgar"

# Non-tabular categories that need re-running
CPU_CATS="statistical"
GPU_CATS="deep_classical transformer_sota foundation irregular"

CPU_TIME="12:00:00"
GPU_TIME="24:00:00"

# ============================================================
# Step 1: Invalidate old MANIFEST.json for non-tabular shards
# ============================================================
echo "=== Invalidating old non-tabular results ==="
for TASK in ${TASKS}; do
    for ABL in ${ABLATIONS}; do
        for CAT in ${CPU_CATS} ${GPU_CATS}; do
            MFILE="${OUTBASE}/${TASK}/${CAT}/${ABL}/MANIFEST.json"
            if [[ -f "${MFILE}" ]]; then
                # Rename to preserve for audit trail
                mv "${MFILE}" "${MFILE}.constant_pred_invalid"
                echo "  Invalidated: ${TASK}/${CAT}/${ABL}/MANIFEST.json"
            fi
        done
    done
done

# ============================================================
# Step 2: Submit shards
# ============================================================
TOTAL=0
JOBS=""

submit_shard() {
    local TASK="$1"
    local CAT="$2"
    local ABL="$3"
    local PARTITION="$4"
    local TIMELIM="$5"
    local EXTRA_SBATCH="${6:-}"

    local OUTDIR="${OUTBASE}/${TASK}/${CAT}/${ABL}"
    local JOBNAME="b3fx_${TASK}_${CAT}_${ABL}"
    JOBNAME="${JOBNAME:0:30}"

    local SBATCH_SCRIPT=$(mktemp /tmp/b3fx_XXXXXX.sbatch)
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

echo "=== Block 3 PREDICT FIX RERUN ==="
echo "Task:      ${TASK}"
echo "Category:  ${CAT}"
echo "Ablation:  ${ABL}"
echo "Partition: ${PARTITION}"
echo "Git hash:  ${GIT_HASH}"
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname -s)"
echo "Started:   \$(date -Iseconds)"

eval "\$(${MAMBA_EXE} shell hook --shell bash)"
micromamba activate insider

cd ${REPO}

# Verify per-entity predict fix is present
python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_block3_benchmark_shard import BenchmarkShard
assert hasattr(BenchmarkShard, '_TARGET_LEAK_GROUPS'), 'Leakage guard missing!'
# Verify predict kwargs are passed
import inspect
src = inspect.getsource(BenchmarkShard.run_model)
assert 'predict_kwargs' in src, 'predict_kwargs not in run_model — fix not applied!'
print('Predict fix: VERIFIED')
print('Leakage guard: VERIFIED')
"

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
    JOBS="${JOBS} ${JOB_ID}"
    TOTAL=$((TOTAL + 1))
    rm -f "${SBATCH_SCRIPT}"
}

echo ""
echo "=== Submitting 30 non-tabular shards ==="
for TASK in ${TASKS}; do
    echo ""
    echo "--- ${TASK} ---"
    for ABL in ${ABLATIONS}; do
        for CAT in ${CPU_CATS}; do
            submit_shard "${TASK}" "${CAT}" "${ABL}" "batch" "${CPU_TIME}"
        done
        for CAT in ${GPU_CATS}; do
            submit_shard "${TASK}" "${CAT}" "${ABL}" "gpu" "${GPU_TIME}" \
                "#SBATCH --gres=gpu:1"
        done
    done
done

echo ""
echo "============================================================"
echo "NON-TABULAR RELAUNCH — Summary"
echo "  Fix commit:   5c83f2a (per-entity predict)"
echo "  Shards:       ${TOTAL}"
echo "  Job IDs:     ${JOBS}"
echo "  ml_tabular:   NOT touched (still running, correct)"
echo "  Output:       ${OUTBASE}"
echo "  Monitor:      squeue -u \$USER"
echo "============================================================"
