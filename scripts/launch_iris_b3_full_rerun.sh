#!/bin/bash
###############################################################################
# Block 3 Benchmark — FULL PRESET RERUN (KDD'26 Standard)
#
# Rationale: ALL previous results are INVALID due to:
#   1. Target-synonym leakage (funding_raised as feature → funding_raised_usd)
#   2. y.fillna(0) bias on targets (NaN ≠ zero funding)
#   3. Foundation/irregular silent np.mean fallback
#
# Fixes applied in commits:
#   9ad83dd  CRITICAL: Fix target-synonym leakage + NaN-fill bias
#   33c701e  Fix foundation/irregular silent fallback
#
# Matrix: 3 tasks × 2 ablations × 6 categories = 36 shards
# Preset: FULL (horizons=[1,7,14,30], k=[7,14,30,60,90], bootstrap=1000)
# Partitions: batch (CPU) for statistical/ml_tabular
#             gpu   (V100) for deep_classical/transformer_sota/foundation/irregular
#
# Usage:
#   bash scripts/launch_iris_b3_full_rerun.sh
###############################################################################
set -euo pipefail

PRESET="full"
STAMP="20260203_225620"
REPO="/home/users/npin/repo_root"
OUTBASE="${REPO}/runs/benchmarks/block3_${STAMP}_iris_full"
SLURMLOG="${REPO}/runs/slurm"
MAMBA_EXE="/mnt/aiongpfs/users/npin/.local/bin/micromamba"

mkdir -p "${SLURMLOG}" "${OUTBASE}"

# Verify leakage fix is present
HARNESS="${REPO}/scripts/run_block3_benchmark_shard.py"
if ! grep -q '_TARGET_LEAK_GROUPS' "${HARNESS}"; then
    echo "FATAL: _TARGET_LEAK_GROUPS not found in harness — leakage fix missing!"
    exit 1
fi

# Verify commit hash
GIT_HASH=$(cd "${REPO}" && git rev-parse --short HEAD)
echo "Git hash: ${GIT_HASH}"

# Tasks and ablations (NO text — not available on Iris)
TASKS="task1_outcome task2_forecast task3_risk_adjust"
ABLATIONS="core_only core_edgar"

# Category → partition mapping
CPU_CATS="statistical ml_tabular"
GPU_CATS="deep_classical transformer_sota foundation irregular"

echo "============================================================"
echo "Block 3 FULL PRESET RERUN — KDD'26 Standard"
echo "  Stamp:     ${STAMP}"
echo "  Preset:    ${PRESET}"
echo "  Git hash:  ${GIT_HASH}"
echo "  Output:    ${OUTBASE}"
echo "  Tasks:     ${TASKS}"
echo "  Ablations: ${ABLATIONS} (text skipped)"
echo "  CPU cats:  ${CPU_CATS} (→ batch)"
echo "  GPU cats:  ${GPU_CATS} (→ gpu)"
echo "============================================================"

# Time limits — FULL preset needs more time
CPU_TIME="12:00:00"
GPU_TIME="24:00:00"

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
    local JOBNAME="b3f_${TASK}_${CAT}_${ABL}"
    # Truncate job name to 30 chars for SLURM
    JOBNAME="${JOBNAME:0:30}"

    # Skip if shard already completed
    if [[ -f "${OUTDIR}/MANIFEST.json" ]]; then
        local SHARD_STATUS
        SHARD_STATUS=$(python3 -c "
import json, sys
m = json.load(open('${OUTDIR}/MANIFEST.json'))
print(m.get('status', 'unknown'))
" 2>/dev/null || echo "unknown")
        if [[ "${SHARD_STATUS}" == "completed" ]]; then
            echo "  SKIP (completed): ${TASK}/${CAT}/${ABL}"
            return
        fi
    fi

    local SBATCH_SCRIPT=$(mktemp /tmp/b3_full_XXXXXX.sbatch)
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

echo "=== Block 3 FULL PRESET Shard ==="
echo "Task:      ${TASK}"
echo "Category:  ${CAT}"
echo "Ablation:  ${ABL}"
echo "Partition: ${PARTITION}"
echo "Preset:    ${PRESET}"
echo "Git hash:  ${GIT_HASH}"
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname -s)"
echo "Started:   \$(date -Iseconds)"
echo "================================="

# Activate conda environment
eval "\$(${MAMBA_EXE} shell hook --shell bash)"
micromamba activate insider

cd ${REPO}

# Verify leakage fix at runtime
python -c "
import sys
sys.path.insert(0, '.')
from scripts.run_block3_benchmark_shard import BenchmarkShard
assert hasattr(BenchmarkShard, '_TARGET_LEAK_GROUPS'), 'Leakage guard missing!'
assert 'funding_raised' in BenchmarkShard._TARGET_LEAK_GROUPS['funding_raised_usd'], 'funding_raised not in leak group!'
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
    SUBMITTED_JOBS="${SUBMITTED_JOBS} ${JOB_ID}"
    TOTAL_SHARDS=$((TOTAL_SHARDS + 1))
    rm -f "${SBATCH_SCRIPT}"
}

# ============================================================
# Submit all 36 shards
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
echo "FULL PRESET RERUN — Summary"
echo "  Total shards submitted: ${TOTAL_SHARDS}"
echo "  Job IDs: ${SUBMITTED_JOBS}"
echo "  Monitor: squeue -u \$USER"
echo "  Output:  ${OUTBASE}"
echo ""
echo "Verification:"
echo "  - Leakage fix:  VERIFIED (runtime check in each shard)"
echo "  - NaN handling: dropna() (not fillna(0))"
echo "  - Foundation:   hard-fail (no silent fallback)"
echo "  - Preset:       FULL (h=[1,7,14,30], k=[7,14,30,60,90])"
echo "============================================================"
