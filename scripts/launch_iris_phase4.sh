#!/usr/bin/env bash
# ── Phase 4 Full Benchmark on ULHPC Iris ──────────────────────────────────
# 59 models across 7 categories × 3 tasks × 2 ablations = 42 shards
# Output: runs/benchmarks/block3_20260203_225620_iris_phase4/
#
# Phase 4 additions (on top of Phase 3 fixes):
#   - transformer_sota: 15 → 20 (+DLinear, NLinear, TimeMixer, TimeXer, TSMixerx)
#   - foundation:        2 →  6 (+ChronosBolt, Chronos2, MoiraiLarge, Moirai2)
#   - autofit:           6 →  7 (+AutoFitV4: target-transform, full-OOF stacking, NCL diversity)
#   Total: 49 → 59 models (excl. unavailable: TimesFM, Sundial, Toto, Xihe, etc.)
#
# Usage:
#   bash scripts/launch_iris_phase4.sh              # submit all
#   bash scripts/launch_iris_phase4.sh --dry-run    # print without submitting
#   bash scripts/launch_iris_phase4.sh --only transformer_sota  # single category

set -euo pipefail

DRY_RUN=false
ONLY_CAT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --only) ONLY_CAT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY RUN — no jobs will be submitted"
fi

STAMP="20260203_225620"
BASE_DIR="runs/benchmarks/block3_${STAMP}_iris_phase4"
SCRIPT="scripts/run_block3_benchmark_shard.py"
ACCOUNT="yves.letraon"
PARTITION="gpu"
QOS="normal"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_edgar")

# Category → walltime (Phase 4 adjusted for new model counts)
declare -A WALLTIME
WALLTIME[ml_tabular]="12:00:00"       # 15 models, unchanged
WALLTIME[statistical]="12:00:00"      #  5 models, unchanged
WALLTIME[deep_classical]="24:00:00"   #  4 models, unchanged
WALLTIME[transformer_sota]="48:00:00" # 20 models (+5 new), was 36h
WALLTIME[foundation]="24:00:00"       #  6 models (+4 new), was 18h
WALLTIME[irregular]="18:00:00"        #  2 models, unchanged
WALLTIME[autofit]="72:00:00"          #  7 variants, V4 = 27cand×5fold, was 48h

# Category → memory
declare -A MEMORY
MEMORY[ml_tabular]="256G"
MEMORY[statistical]="128G"
MEMORY[deep_classical]="256G"
MEMORY[transformer_sota]="256G"
MEMORY[foundation]="256G"
MEMORY[irregular]="256G"
MEMORY[autofit]="256G"

ALL_CATEGORIES=("ml_tabular" "statistical" "deep_classical" "transformer_sota" "foundation" "irregular" "autofit")

if [[ -n "${ONLY_CAT}" ]]; then
    CATEGORIES=("${ONLY_CAT}")
    echo "Running only category: ${ONLY_CAT}"
else
    CATEGORIES=("${ALL_CATEGORIES[@]}")
fi

SLURM_DIR=".slurm_scripts/phase4"
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
            C_SHORT="${CAT:0:4}"
            A_SHORT="${ABL/core_/}"
            JOB_NAME="p4_${C_SHORT}_${T_SHORT}_${A_SHORT}"

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
echo "Phase 4 — Job \${SLURM_JOB_ID} on \$(hostname)"
echo "Task: ${TASK} | Category: ${CAT} | Ablation: ${ABL}"
echo "Python: \$PYTHON"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
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

RC=\$?
echo "Exit code: \$RC | Done: \$(date -Iseconds)"
exit \$RC
SLURM_EOF

            chmod +x "${SCRIPT_FILE}"

            if [[ "${DRY_RUN}" == "true" ]]; then
                echo "  [DRY] ${JOB_NAME}: ${TASK} ${CAT} ${ABL} (${WT}, ${MEM})"
            else
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
echo "Phase 4 Benchmark Launch Summary"
echo "============================================================"
echo "Total shards: ${JOB_COUNT}"
echo "Tasks:      ${TASKS[*]}"
echo "Categories: ${CATEGORIES[*]}"
echo "Ablations:  ${ABLATIONS[*]}"
echo "Output:     ${BASE_DIR}/"
echo ""
echo "Model counts: ml_tabular=15 statistical=5 deep_classical=4"
echo "              transformer_sota=20 foundation=6 irregular=2 autofit=7"
echo "              total=59"
echo ""
if [[ "${DRY_RUN}" == "false" && ${#JOB_IDS[@]} -gt 0 ]]; then
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Monitor: squeue -u \$USER --format='%.8i %.12j %.4T %.10M %.6D %R'"
    echo "Cancel:  scancel ${JOB_IDS[0]}-${JOB_IDS[-1]}"
fi
echo "============================================================"
