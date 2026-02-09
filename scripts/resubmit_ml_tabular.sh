#!/usr/bin/env bash
# ============================================================================
# Resubmit ML_TABULAR jobs — resume from existing partial metrics
# Fix: LogisticRegression now skipped for regression targets (investors_count)
# Fix: Resume support reads existing metrics.json and skips done combos
# ============================================================================
set -euo pipefail

STAMP="20260203_225620"
BASE="runs/benchmarks/block3_${STAMP}_iris_full"
SCRIPT="scripts/run_block3_benchmark_shard.py"
ACCOUNT="yves.letraon"
PARTITION="gpu"
QOS="normal"
TIME="24:00:00"
MEM="256G"
CPUS=7
GPUS=1
SEED=42
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
TMPDIR_SCRIPTS="${WORKDIR}/.slurm_scripts"
mkdir -p "${TMPDIR_SCRIPTS}"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_edgar")
SUBMITTED=0

echo "============================================================"
echo "ML_TABULAR Resume Launcher (with LogisticRegression fix)"
echo "Working dir: ${WORKDIR}"
echo "============================================================"

for TASK in "${TASKS[@]}"; do
    for ABL in "${ABLATIONS[@]}"; do
        JOB_NAME="mlt_${TASK: -1}_${ABL:5:2}"
        OUTDIR="${BASE}/${TASK}/ml_tabular/${ABL}"
        mkdir -p "${OUTDIR}"

        SBATCH_SCRIPT="${TMPDIR_SCRIPTS}/${JOB_NAME}_resume.sh"
        cat > "${SBATCH_SCRIPT}" << SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:volta:${GPUS}
#SBATCH --output=${OUTDIR}/slurm_%j.log
#SBATCH --error=${OUTDIR}/slurm_%j.err
#SBATCH --export=ALL

set -e

# Activate environment
if [[ -f "/home/users/npin/miniforge3/etc/profile.d/micromamba.sh" ]]; then
    source /home/users/npin/miniforge3/etc/profile.d/micromamba.sh
    micromamba activate insider
elif [[ -f "/home/users/npin/miniforge3/etc/profile.d/conda.sh" ]]; then
    source /home/users/npin/miniforge3/etc/profile.d/conda.sh
    conda activate insider
fi

cd ${WORKDIR}

echo "============================================================"
echo "Job \${SLURM_JOB_ID} on \$(hostname)"
echo "Task: ${TASK} | Category: ml_tabular | Ablation: ${ABL}"
echo "Python: \$(which python3)"
echo "Memory limit: ${MEM}"
echo "Start: \$(date -Iseconds)"
echo "nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "============================================================"

python3 ${SCRIPT} \\
    --task ${TASK} \\
    --category ml_tabular \\
    --ablation ${ABL} \\
    --preset full \\
    --output-dir ${OUTDIR} \\
    --seed ${SEED} \\
    --no-verify-first

echo "Done: \$(date -Iseconds)"
SBATCH_EOF

        chmod +x "${SBATCH_SCRIPT}"
        echo "  SUBMIT: ${TASK}/ml_tabular/${ABL} -> ${JOB_NAME}"
        sbatch "${SBATCH_SCRIPT}"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "============================================================"
echo "Submitted ${SUBMITTED} ml_tabular resume jobs"
echo "Monitor: squeue -u \$USER --sort=+i"
echo "============================================================"
