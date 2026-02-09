#!/usr/bin/env bash
# ============================================================================
# Block 3 — Submit ALL missing shards (ml_tabular + autofit)
# ============================================================================
# Fixes from previous attempts:
#   1. Uses sbatch script files (not --wrap) to avoid shell escaping issues
#   2. Memory 256G (previous ml_tabular OOM'd at 128G with MaxRSS=188G)
#   3. GC added to benchmark harness (gc.collect + torch.cuda.empty_cache)
#   4. ml_tabular split into:
#      - ml_tabular (all 15): single job per shard, 256G
#      - Fallback: if OOM, can split into sklearn subset
#   5. autofit: needs GPU for internal models (TFT, Moirai)
#
# Usage: bash scripts/launch_missing_jobs.sh
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

echo "============================================================"
echo "Block 3 Missing Jobs Launcher"
echo "Working dir: ${WORKDIR}"
echo "Output base: ${BASE}"
echo "Memory: ${MEM}"
echo "============================================================"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_edgar")
SUBMITTED=0
SKIPPED=0

# ============================================================================
# Helper: create and submit a proper sbatch script
# ============================================================================
submit_shard() {
    local TASK="$1"
    local HARNESS_CAT="$2"    # --category flag
    local DIR_CAT="$3"        # directory name
    local ABLATION="$4"
    local JOB_NAME="$5"
    local EXTRA_ARGS="${6:-}"  # e.g. --models XGBoost,LightGBM

    local OUTDIR="${BASE}/${TASK}/${DIR_CAT}/${ABLATION}"
    mkdir -p "${OUTDIR}"

    # Skip if already completed
    if [[ -f "${OUTDIR}/MANIFEST.json" ]]; then
        local STATUS
        STATUS=$(python3 -c "import json; print(json.load(open('${OUTDIR}/MANIFEST.json')).get('status',''))" 2>/dev/null || echo "")
        if [[ "${STATUS}" == "completed" ]]; then
            echo "  SKIP (completed): ${TASK}/${DIR_CAT}/${ABLATION}"
            SKIPPED=$((SKIPPED + 1))
            return 0
        fi
        echo "  RE-RUN (status=${STATUS}): ${TASK}/${DIR_CAT}/${ABLATION}"
    fi

    # Create sbatch script
    local SBATCH_SCRIPT="${TMPDIR_SCRIPTS}/${JOB_NAME}.sh"
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
echo "Task: ${TASK} | Category: ${DIR_CAT} | Ablation: ${ABLATION}"
echo "Python: \$(which python3)"
echo "Memory limit: ${MEM}"
echo "Start: \$(date -Iseconds)"
echo "============================================================"

python3 ${SCRIPT} \\
    --task ${TASK} \\
    --category ${HARNESS_CAT} \\
    --ablation ${ABLATION} \\
    --preset full \\
    --output-dir ${OUTDIR} \\
    --seed ${SEED} \\
    --no-verify-first \\
    ${EXTRA_ARGS}

echo "Done: \$(date -Iseconds)"
SBATCH_EOF

    chmod +x "${SBATCH_SCRIPT}"

    echo "  SUBMIT: ${TASK}/${DIR_CAT}/${ABLATION} -> ${JOB_NAME}"
    sbatch "${SBATCH_SCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
}

# ============================================================================
# 1. ML_TABULAR — all 15 models (3 tasks × 2 ablations = 6 jobs)
#    Previous OOM at 128G (MaxRSS=188G). Now 256G + GC between models.
# ============================================================================
echo ""
echo "=== [1/2] ML_TABULAR (15 models, 256G) ==="
for TASK in "${TASKS[@]}"; do
    for ABL in "${ABLATIONS[@]}"; do
        submit_shard "${TASK}" "ml_tabular" "ml_tabular" "${ABL}" \
            "mlt_${TASK: -1}_${ABL:5:2}"
    done
done

# ============================================================================
# 2. AUTOFIT — redesigned v1/v2/v2e (3 tasks × 2 ablations = 6 jobs)
#    Clear old broken results before resubmitting.
# ============================================================================
echo ""
echo "=== [2/2] AUTOFIT (3 models: V1, V2, V2E) ==="
for TASK in "${TASKS[@]}"; do
    for ABL in "${ABLATIONS[@]}"; do
        # Remove old broken results
        local_dir="${BASE}/${TASK}/autofit/${ABL}"
        if [[ -d "${local_dir}" ]]; then
            rm -f "${local_dir}/MANIFEST.json" "${local_dir}/metrics.json" \
                  "${local_dir}/predictions.parquet" 2>/dev/null || true
        fi
        submit_shard "${TASK}" "autofit" "autofit" "${ABL}" \
            "af_${TASK: -1}_${ABL:5:02}"
    done
done

# ============================================================================
# 3. Check GBM backup — the 4 GBM models only have 1 shard at root level
#    Need remaining 5 shards: task1/core_only, task2/*, task3/*
# ============================================================================
echo ""
echo "=== [3/3] GBM results gap-fill ==="
# Root MANIFEST has task1_outcome/core_edgar (48 records for 4 GBMs)
# Need to fill: task1/core_only + all task2 + all task3
# We'll rely on the full ml_tabular runs above to cover these

echo ""
echo "============================================================"
echo "SUMMARY: Submitted=${SUBMITTED}, Skipped=${SKIPPED}"
echo "Monitor:  squeue -u \$USER --sort=+i"
echo "Results:  ${BASE}/"
echo "Logs:     find ${BASE} -name 'slurm_*.log' -newer /tmp/launch_ts"
echo "============================================================"
touch /tmp/launch_ts
