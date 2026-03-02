#!/usr/bin/env bash
# ── Phase 3 Full Benchmark Re-Run on ULHPC Iris ──────────────────────────
# Submits ALL shards for 7 categories × 3 tasks × 2 ablations = 42 shards
# Output: runs/benchmarks/block3_20260203_225620_iris_phase3/
#
# Phase 3 Fixes applied:
#   1. Deep/Transformer: 2000-entity coverage + Ridge fallback (not global_mean)
#   2. EDGAR: As-of join (merge_asof, backward, 90D tolerance)
#   3. AutoFit V3Max: K≤6 + 30min time budget
#   4. GBDT: Auto Tweedie/Poisson for count targets
#   5. Horizon: Single horizon for ml_tabular (cross-sectional)
#   6. Entity coverage: statistical 500, irregular 1000
#
# Usage:
#   bash scripts/launch_iris_phase3.sh          # submit all
#   bash scripts/launch_iris_phase3.sh --dry-run  # print without submitting

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN — no jobs will be submitted"
fi

STAMP="20260203_225620"
BASE_DIR="runs/benchmarks/block3_${STAMP}_iris_phase3"
SCRIPT="scripts/run_block3_benchmark_shard.py"
ACCOUNT="yves.letraon"
PARTITION="gpu"
QOS="normal"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_edgar")

# Category → walltime mapping (Phase 3 optimized)
declare -A WALLTIME
WALLTIME[ml_tabular]="12:00:00"
WALLTIME[statistical]="12:00:00"
WALLTIME[deep_classical]="24:00:00"
WALLTIME[transformer_sota]="36:00:00"
WALLTIME[foundation]="18:00:00"
WALLTIME[irregular]="18:00:00"
WALLTIME[autofit]="48:00:00"

# Category → memory mapping
declare -A MEMORY
MEMORY[ml_tabular]="256G"
MEMORY[statistical]="128G"
MEMORY[deep_classical]="256G"
MEMORY[transformer_sota]="256G"
MEMORY[foundation]="256G"
MEMORY[irregular]="256G"
MEMORY[autofit]="256G"

CATEGORIES=("ml_tabular" "statistical" "deep_classical" "transformer_sota" "foundation" "irregular" "autofit")

SLURM_DIR=".slurm_scripts/phase3"
mkdir -p "${SLURM_DIR}"

JOB_COUNT=0
JOB_IDS=()

for TASK in "${TASKS[@]}"; do
    for CAT in "${CATEGORIES[@]}"; do
        for ABL in "${ABLATIONS[@]}"; do
            WT="${WALLTIME[$CAT]}"
            MEM="${MEMORY[$CAT]}"
            OUT_DIR="${BASE_DIR}/${TASK}/${CAT}/${ABL}"

            # Short name for SLURM
            T_SHORT="${TASK/task/t}"
            T_SHORT="${T_SHORT/_outcome/o}"
            T_SHORT="${T_SHORT/_forecast/f}"
            T_SHORT="${T_SHORT/_risk_adjust/r}"
            C_SHORT="${CAT:0:3}"
            A_SHORT="${ABL/core_/}"
            JOB_NAME="p3_${C_SHORT}_${T_SHORT}_${A_SHORT}"

            SCRIPT_FILE="${SLURM_DIR}/${JOB_NAME}.sh"

            cat > "${SCRIPT_FILE}" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --time=${WT}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:volta:1
#SBATCH --output=${OUT_DIR}/slurm_%j.log
#SBATCH --error=${OUT_DIR}/slurm_%j.err
#SBATCH --export=ALL
#SBATCH --signal=B:TERM@180

set -e
mkdir -p ${OUT_DIR}

# ── Robust micromamba activation ──
# Method 1: Use MAMBA_EXE + shell hook (works on all ULHPC nodes)
export MAMBA_EXE="/mnt/aiongpfs/users/npin/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba"
if [[ -x "\$MAMBA_EXE" ]]; then
    eval "\$(\$MAMBA_EXE shell hook --shell bash --root-prefix \$MAMBA_ROOT_PREFIX 2>/dev/null)"
    micromamba activate insider
fi

# Verify Python is from insider env
PYTHON=\$(which python3 2>/dev/null || echo "/usr/bin/python3")
if [[ "\$PYTHON" != *"insider"* ]]; then
    echo "FATAL: Python not from insider env: \$PYTHON" >&2
    exit 1
fi

cd /home/users/npin/repo_root

echo "============================================================"
echo "Phase 3 — Job \${SLURM_JOB_ID} on \$(hostname)"
echo "Task: ${TASK} | Category: ${CAT} | Ablation: ${ABL}"
echo "Python: \$PYTHON"
echo "Memory: ${MEM} | Walltime: ${WT}"
echo "Start: \$(date -Iseconds)"
echo "Git: \$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "============================================================"

\$PYTHON ${SCRIPT} \\
    --task ${TASK} \\
    --category ${CAT} \\
    --ablation ${ABL} \\
    --preset full \\
    --output-dir ${OUT_DIR} \\
    --seed 42 \\
    --no-verify-first

echo "Done: \$(date -Iseconds)"
SLURM_EOF

            chmod +x "${SCRIPT_FILE}"

            if [[ "${DRY_RUN}" == "true" ]]; then
                echo "  [DRY] ${JOB_NAME}: ${TASK} ${CAT} ${ABL} (${WT}, ${MEM})"
            else
                # Create output directory first (for SLURM log files)
                mkdir -p "${OUT_DIR}"
                JID=$(sbatch --parsable "${SCRIPT_FILE}" 2>/dev/null || echo "FAILED")
                JOB_IDS+=("${JID}")
                echo "  Submitted ${JOB_NAME} → ${JID} (${WT}, ${MEM})"
            fi
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo ""
echo "============================================================"
echo "Phase 3 Benchmark Launch Summary"
echo "============================================================"
echo "Total shards: ${JOB_COUNT}"
echo "Tasks: ${TASKS[*]}"
echo "Categories: ${CATEGORIES[*]}"
echo "Ablations: ${ABLATIONS[*]}"
echo "Output: ${BASE_DIR}/"
echo ""
if [[ "${DRY_RUN}" == "false" ]]; then
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor: squeue -u \$USER --format='%.8i %.12j %.4T %.10M %.6D %R'"
fi
echo "============================================================"
