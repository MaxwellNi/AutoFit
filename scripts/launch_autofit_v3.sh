#!/usr/bin/env bash
# ============================================================================
# Block 3 — Resubmit AutoFit with Exhaustive Stacking (6 variants)
# ============================================================================
# New AutoFit variants:
#   - AutoFitV1:    Data-driven best single model + residual correction
#   - AutoFitV2:    Top-5 weighted ensemble (all candidates evaluated)
#   - AutoFitV2E:   Top-5 stacking + LightGBM meta-learner
#   - AutoFitV3:    Greedy forward selection + temporal CV + meta-learner
#   - AutoFitV3E:   Top-6 stacking with temporal CV
#   - AutoFitV3Max: Exhaustive 2^K subset search (K=8) + meta-learner
#
# Key improvements over old autofit:
#   - ZERO heuristic routing: all 17 base models evaluated empirically
#   - 2-fold temporal CV for OOF (V3 family)
#   - Anti-overfit guard: meta-learner discarded if it doesn't improve
#   - Greedy forward selection: stops when no improvement
#   - Strong L1/L2 regularization on meta-learner
#
# Memory: 256G (autofit internally fits multiple models)
# Time: 48h (V3Max does exhaustive search over all subsets)
# GPU: 1x volta (needed for TFT, Moirai, Chronos internals)
#
# Usage: bash scripts/launch_autofit_v3.sh
# ============================================================================
set -euo pipefail

STAMP="20260203_225620"
BASE="runs/benchmarks/block3_${STAMP}_iris_full"
SCRIPT="scripts/run_block3_benchmark_shard.py"
ACCOUNT="yves.letraon"
PARTITION="gpu"
QOS="normal"
TIME="48:00:00"
MEM="256G"
CPUS=7
GPUS=1
SEED=42
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
TMPDIR_SCRIPTS="${WORKDIR}/.slurm_scripts"
mkdir -p "${TMPDIR_SCRIPTS}"

echo "============================================================"
echo "AutoFit V3 Exhaustive Stacking — SLURM Launcher"
echo "Working dir: ${WORKDIR}"
echo "Output base: ${BASE}"
echo "Memory: ${MEM}, Time: ${TIME}"
echo "Variants: AutoFitV1, V2, V2E, V3, V3E, V3Max (6 total)"
echo "============================================================"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_edgar")
SUBMITTED=0

for TASK in "${TASKS[@]}"; do
    for ABL in "${ABLATIONS[@]}"; do
        JOB_NAME="afv3_${TASK: -1}_${ABL:5:2}"
        OUTDIR="${BASE}/${TASK}/autofit/${ABL}"
        mkdir -p "${OUTDIR}"

        # Clear old results
        rm -f "${OUTDIR}/MANIFEST.json" "${OUTDIR}/metrics.json" \
              "${OUTDIR}/predictions.parquet" 2>/dev/null || true

        # Create sbatch script
        SBATCH_SCRIPT="${TMPDIR_SCRIPTS}/${JOB_NAME}.sh"
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
echo "Task: ${TASK} | Category: autofit | Ablation: ${ABL}"
echo "Variants: AutoFitV1, V2, V2E, V3, V3E, V3Max"
echo "Python: \$(which python3)"
echo "Memory limit: ${MEM}"
echo "Start: \$(date -Iseconds)"
echo "nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "============================================================"

python3 ${SCRIPT} \\
    --task ${TASK} \\
    --category autofit \\
    --ablation ${ABL} \\
    --preset full \\
    --output-dir ${OUTDIR} \\
    --seed ${SEED} \\
    --no-verify-first

echo "============================================================"
echo "Done: \$(date -Iseconds)"
echo "Results: \$(wc -c < ${OUTDIR}/metrics.json 2>/dev/null || echo 0) bytes"
echo "============================================================"
SBATCH_EOF

        chmod +x "${SBATCH_SCRIPT}"
        echo "  SUBMIT: ${TASK}/autofit/${ABL} -> ${JOB_NAME}"
        sbatch "${SBATCH_SCRIPT}"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "============================================================"
echo "SUBMITTED: ${SUBMITTED} autofit jobs"
echo "Monitor:   squeue -u \$USER --sort=+i"
echo "Results:   python3 scripts/monitor_benchmark.py"
echo "============================================================"
