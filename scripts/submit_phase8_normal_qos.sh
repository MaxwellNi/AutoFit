#!/usr/bin/env bash
# ============================================================================
# Phase 8 — SATURATED Submission via `normal` QOS
# ============================================================================
# KEY INSIGHT: `normal` QOS on gpu partition allows:
#   - MaxJobsPU = 100  (vs iris-gpu-long's 4!)
#   - GrpNodes  = unlimited (vs iris-gpu-long's 6!)
#   - Priority  = 100  (same as iris-gpu-long)
#   - MaxWall   = 2 days (partition limit, vs iris-gpu-long's 14d)
#
# Account: christian.fisch (NOT yves.letraon — that one lacks normal QOS)
# All GPU jobs capped at 2-day wall time.
# ============================================================================
set -euo pipefail

REPO="/home/users/npin/repo_root"
STAMP="20260203_225620"
OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7"
LOG_DIR="/work/projects/eint/logs/phase8_sat"
SLURM_DIR="${REPO}/.slurm_scripts/phase8_sat"
ACCOUNT="christian.fisch"

PRESET="full"
SEED=42
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done

# ============================================================================
# Activate insider env
# ============================================================================
if [[ "${CONDA_DEFAULT_ENV:-}" != "insider" ]]; then
    export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
    eval "$(micromamba shell hook -s bash)"
    micromamba activate insider
fi

PY_BIN="$(command -v python3)"
if [[ "${PY_BIN}" != *"insider"* ]]; then
    echo "FATAL: python3 not from insider: ${PY_BIN}"
    exit 2
fi

mkdir -p "$LOG_DIR" "$SLURM_DIR"

# ============================================================================
# Shared preamble for all SLURM scripts
# ============================================================================
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

TOTAL=0

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

    if $DRY_RUN; then
        echo "  [DRY] ${JOB_NAME} → gpu/normal ${TIME} ${MEM}"
    else
        local JID
        JID=$(sbatch "$SCRIPT" 2>&1 | grep -oP '\d+')
        echo "  [SUB] ${JOB_NAME} → JobID ${JID}"
    fi
    TOTAL=$((TOTAL + 1))
}

submit_batch() {
    local JOB_NAME="$1" TIME="$2" MEM="$3" CPUS="$4" TASK="$5"
    local CATEGORY="$6" ABLATION="$7" MODELS="${8:-}"

    local OUTDIR="${OUTPUT_BASE}/${TASK}/${CATEGORY}/${ABLATION}"
    local SCRIPT="${SLURM_DIR}/${JOB_NAME}.sh"
    local MODELS_ARG=""
    [[ -n "$MODELS" ]] && MODELS_ARG="--models ${MODELS}"

    cat > "$SCRIPT" << SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
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

    if $DRY_RUN; then
        echo "  [DRY] ${JOB_NAME} → batch/normal ${TIME} ${MEM} ${CPUS}c"
    else
        local JID
        JID=$(sbatch "$SCRIPT" 2>&1 | grep -oP '\d+')
        echo "  [SUB] ${JOB_NAME} → JobID ${JID}"
    fi
    TOTAL=$((TOTAL + 1))
}

# ============================================================================
# Task-ablation matrix (11 combos)
# ============================================================================
TASK1_ABLS=("core_only" "core_text" "core_edgar" "full")
TASK2_ABLS=("core_only" "core_text" "core_edgar" "full")
TASK3_ABLS=("core_only" "core_edgar" "full")

ALL_TA=()
for a in "${TASK1_ABLS[@]}"; do ALL_TA+=("task1_outcome:${a}"); done
for a in "${TASK2_ABLS[@]}"; do ALL_TA+=("task2_forecast:${a}"); done
for a in "${TASK3_ABLS[@]}"; do ALL_TA+=("task3_risk_adjust:${a}"); done

echo "================================================================"
echo " Phase 8 SATURATED Submission"
echo " QOS:     normal (MaxJobsPU=100, no GrpNodes limit)"
echo " Account: ${ACCOUNT}"
echo " Combos:  ${#ALL_TA[@]} task-ablation pairs"
echo " Output:  ${OUTPUT_BASE}"
echo "================================================================"

# ============================================================================
# SHARD 8a: TSLib SOTA (14 models, GPU, 2d wall)
# ============================================================================
echo ""
echo "--- Shard 8a: tslib_sota (14 models → 2 sub-shards × 11 combos = 22 GPU) ---"
TSLIB_T1="TimeFilter,WPMixer,MultiPatchFormer,MSGNet,PAttn,MambaSimple,Crossformer"
TSLIB_T2="Koopa,FreTS,MICN,SegRNN,NonstationaryTransformer,FiLM,SCINet"

for ta in "${ALL_TA[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_gpu "sat_tsA_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "2-00:00:00" "320G" "$TASK" "tslib_sota" "$ABL" "$TSLIB_T1"
    submit_gpu "sat_tsB_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "2-00:00:00" "320G" "$TASK" "tslib_sota" "$ABL" "$TSLIB_T2"
done

# ============================================================================
# SHARD 8b: New Foundation Models (4 models, GPU, 2d wall)
# ============================================================================
echo ""
echo "--- Shard 8b: foundation_new (4 models × 11 combos = 11 GPU) ---"
FM_NEW="Sundial,TTM,TimerXL,TimesFM2"
for ta in "${ALL_TA[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_gpu "sat_fmN_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "2-00:00:00" "256G" "$TASK" "foundation" "$ABL" "$FM_NEW"
done

# ============================================================================
# SHARD 8c: Deep Classical Backfill (5 models, GPU, 2d wall)
# ============================================================================
echo ""
echo "--- Shard 8c: deep_classical backfill (5 models × 11 combos = 11 GPU) ---"
DC_BACK="GRU,LSTM,TCN,MLP,DilatedRNN"
for ta in "${ALL_TA[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_gpu "sat_dcB_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "2-00:00:00" "256G" "$TASK" "deep_classical" "$ABL" "$DC_BACK"
done

# ============================================================================
# SHARD 8d: NF Transformer Gap-Fill (10 models, ce+fu gaps, GPU, 2d wall)
# ============================================================================
echo ""
echo "--- Shard 8d: NF transformer gap-fill (t3 all 3 abl + t1/t2 ce+fu = 7 GPU) ---"
NF_GAP="PatchTST,iTransformer,TimesNet,TSMixer,Informer,Autoformer,FEDformer,VanillaTransformer,TiDE,NBEATSx"
for abl in "core_only" "core_edgar" "full"; do
    submit_gpu "sat_nfG_t3_$(abl_abbrev $abl)" \
        "2-00:00:00" "320G" "task3_risk_adjust" "transformer_sota" "$abl" "$NF_GAP"
done
for task in "task1_outcome" "task2_forecast"; do
    for abl in "core_edgar" "full"; do
        submit_gpu "sat_nfG_$(task_abbrev $task)_$(abl_abbrev $abl)" \
            "2-00:00:00" "320G" "$task" "transformer_sota" "$abl" "$NF_GAP"
    done
done

# ============================================================================
# SHARD 8d-ext: TimesFM gap-fill (all 11 combos, GPU, 1.5d wall)
# ============================================================================
echo ""
echo "--- Shard 8d-ext: TimesFM gap-fill (11 GPU) ---"
for ta in "${ALL_TA[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_gpu "sat_fmTF_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "1-12:00:00" "256G" "$TASK" "foundation" "$ABL" "TimesFM"
done

# ============================================================================
# SHARD 8e: AutoFit Gap-Fill (batch, 2d wall, normal QOS = 100 max)
# ============================================================================
echo ""
echo "--- Shard 8e: autofit V1-V7 gap-fill (10 models × 11 combos = 11 batch) ---"
AF_FILL="AutoFitV1,AutoFitV2,AutoFitV2E,AutoFitV3,AutoFitV3E,AutoFitV3Max,AutoFitV4,AutoFitV5,AutoFitV6,AutoFitV7"
for ta in "${ALL_TA[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_batch "sat_afG_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "2-00:00:00" "112G" "28" "$TASK" "autofit" "$ABL" "$AF_FILL"
done

echo ""
echo "--- Shard 8e: autofit V72-V736 (5 models × 11 combos = 11 batch) ---"
AF_V7X="AutoFitV72,AutoFitV73,AutoFitV731,AutoFitV733,AutoFitV736"
for ta in "${ALL_TA[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_batch "sat_af7_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "2-00:00:00" "112G" "28" "$TASK" "autofit" "$ABL" "$AF_V7X"
done

echo ""
echo "================================================================"
echo " SUBMISSION COMPLETE"
echo " Total jobs: ${TOTAL}"
echo " GPU jobs:  gpu/normal  (MaxJobsPU=100, up to 24 nodes)"
echo " Batch jobs: batch/normal (MaxJobsPU=100)"
echo " All Priority 100 — same as iris-*-long"
echo "================================================================"
