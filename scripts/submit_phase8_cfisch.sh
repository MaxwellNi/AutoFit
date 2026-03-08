#!/usr/bin/env bash
# ============================================================================
# Phase 8 — cfisch submission: dcB (11 GPU) + fmTF (11 GPU) = 22 GPU jobs
# Moved from npin PENDING queue to cfisch for parallel scheduling
# ============================================================================
set -euo pipefail

REPO="/home/users/npin/repo_root"
STAMP="20260203_225620"
OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7"
LOG_DIR="/work/projects/eint/logs/phase8_cf"
SLURM_DIR="/work/projects/eint/slurm_scripts/phase8_cf"
ACCOUNT="christian.fisch"
PRESET="full"
SEED=42

mkdir -p "$LOG_DIR" "$SLURM_DIR"

# Activate insider env
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider

PY_BIN="$(command -v python3)"
if [[ "${PY_BIN}" != *"insider"* ]]; then
    echo "FATAL: python3 not from insider: ${PY_BIN}"
    exit 2
fi

TOTAL=0

read -r -d '' ENV_PREAMBLE << 'ENVBLOCK' || true
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"; exit 2
fi
echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(python3 -V) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"
ENVBLOCK

submit_gpu() {
    local JOB_NAME="$1" TIME="$2" MEM="$3" TASK="$4"
    local CATEGORY="$5" ABLATION="$6" MODELS="${7:-}"

    local OUTDIR="${OUTPUT_BASE}/${TASK}/${CATEGORY}/${ABLATION}"
    local SCRIPT="${SLURM_DIR}/${JOB_NAME}.sh"
    local MODELS_ARG=""
    [[ -n "$MODELS" ]] && MODELS_ARG="--models ${MODELS}"

    cat > "$SCRIPT" << SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:volta:1
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
${ENV_PREAMBLE}

echo "Task: ${TASK} | Cat: ${CATEGORY} | Abl: ${ABLATION} | Models: ${MODELS:-ALL}"
"\${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \\
    --task ${TASK} --category ${CATEGORY} --ablation ${ABLATION} \\
    --preset ${PRESET} --output-dir ${OUTDIR} --seed ${SEED} \\
    --no-verify-first ${MODELS_ARG}
echo "Done: \$(date -Iseconds)"
SLURM_EOF
    chmod +x "$SCRIPT"

    local JID
    JID=$(sbatch "$SCRIPT" 2>&1 | grep -oP '\d+')
    echo "  [SUB] ${JOB_NAME} → JobID ${JID}"
    TOTAL=$((TOTAL + 1))
}

abl_abbrev() {
    case "$1" in
        core_only) echo "co";; core_text) echo "ct";;
        core_edgar) echo "ce";; full) echo "fu";; *) echo "${1:0:2}";;
    esac
}
task_abbrev() {
    case "$1" in
        task1_outcome) echo "t1";; task2_forecast) echo "t2";;
        task3_risk_adjust) echo "t3";; *) echo "${1:0:2}";;
    esac
}

# Task-ablation matrix
TASK1_ABLS=("core_only" "core_text" "core_edgar" "full")
TASK2_ABLS=("core_only" "core_text" "core_edgar" "full")
TASK3_ABLS=("core_only" "core_edgar" "full")

ALL_TA=()
for a in "${TASK1_ABLS[@]}"; do ALL_TA+=("task1_outcome:${a}"); done
for a in "${TASK2_ABLS[@]}"; do ALL_TA+=("task2_forecast:${a}"); done
for a in "${TASK3_ABLS[@]}"; do ALL_TA+=("task3_risk_adjust:${a}"); done

echo "================================================================"
echo " cfisch Phase 8 Submission: dcB + fmTF"
echo " QOS: normal | Account: ${ACCOUNT}"
echo "================================================================"

# ============================================================================
# SHARD dcB: deep_classical backfill (5 models × 11 combos = 11 GPU)
# ============================================================================
echo ""
echo "--- deep_classical backfill (GRU,LSTM,TCN,MLP,DilatedRNN) ---"
DC_BACK="GRU,LSTM,TCN,MLP,DilatedRNN"
for ta in "${ALL_TA[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_gpu "cf_dcB_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "2-00:00:00" "256G" "$TASK" "deep_classical" "$ABL" "$DC_BACK"
done

# ============================================================================
# SHARD fmTF: TimesFM gap-fill (11 combos × 1 model = 11 GPU)
# ============================================================================
echo ""
echo "--- TimesFM gap-fill ---"
for ta in "${ALL_TA[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_gpu "cf_fmTF_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "1-12:00:00" "256G" "$TASK" "foundation" "$ABL" "TimesFM"
done

echo ""
echo "================================================================"
echo " DONE — cfisch submitted: ${TOTAL} GPU jobs"
echo "================================================================"
