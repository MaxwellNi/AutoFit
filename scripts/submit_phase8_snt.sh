#!/usr/bin/env bash
# ============================================================================
# Phase 8 — SNT Saturated Resubmission
# Uses iris-snt-long QOS (MaxJobsPU=100, no GrpNodes limit)
# Splits GPU jobs 50/50 between npin and cfisch accounts
# ============================================================================
set -euo pipefail

REPO="/home/users/npin/repo_root"
STAMP="20260203_225620"
OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7"
LOG_DIR="/work/projects/eint/logs/phase8_snt"
SLURM_DIR="${REPO}/.slurm_scripts/phase8_snt"
ACCOUNT="yves.letraon"
PRESET="full"
SEED=42
DRY_RUN=false
TARGET_USER="${1:-npin}"  # npin or cfisch

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        npin|cfisch) TARGET_USER="$arg" ;;
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
# Shared preamble
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

submit_job() {
    local JOB_NAME="$1" PARTITION="$2" QOS="$3" TIME="$4"
    local MEM="$5" CPUS="$6" GRES="$7" TASK="$8"
    local CATEGORY="$9" ABLATION="${10}" MODELS="${11}"

    local OUTDIR="${OUTPUT_BASE}/${TASK}/${CATEGORY}/${ABLATION}"
    local SCRIPT="${SLURM_DIR}/${JOB_NAME}.sh"
    local GRES_LINE=""
    [[ -n "$GRES" ]] && GRES_LINE="#SBATCH --gres=${GRES}"
    local MODELS_ARG=""
    [[ -n "$MODELS" ]] && MODELS_ARG="--models ${MODELS}"

    cat > "$SCRIPT" << SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
${GRES_LINE}
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
        echo "  [DRY] ${JOB_NAME} → ${PARTITION}/${QOS} ${TIME} ${MEM} ${CPUS}c ${GRES:-no-gpu}"
    else
        local JID
        JID=$(sbatch "$SCRIPT" 2>&1 | grep -oP '\d+')
        echo "  [SUB] ${JOB_NAME} → JobID ${JID}"
        TOTAL=$((TOTAL + 1))
    fi
}

# ============================================================================
# Task-ablation matrix
# ============================================================================
TASK1_ABLS=("core_only" "core_text" "core_edgar" "full")
TASK2_ABLS=("core_only" "core_text" "core_edgar" "full")
TASK3_ABLS=("core_only" "core_edgar" "full")

ALL_TA=()
for a in "${TASK1_ABLS[@]}"; do ALL_TA+=("task1_outcome:${a}"); done
for a in "${TASK2_ABLS[@]}"; do ALL_TA+=("task2_forecast:${a}"); done
for a in "${TASK3_ABLS[@]}"; do ALL_TA+=("task3_risk_adjust:${a}"); done

echo "================================================================"
echo "Phase 8 SNT Saturated Submission — user=${TARGET_USER}"
echo "QOS: iris-snt-long (MaxJobs=100, no GrpNodes limit)"
echo "Combos: ${#ALL_TA[@]} | Output: ${OUTPUT_BASE}"
echo "================================================================"

# ============================================================================
# GPU JOBS (iris-snt-long QOS — 100 concurrent, no node limit)
# ============================================================================
GPU_QOS="iris-snt-long"
BATCH_QOS="iris-snt-long"

if [[ "$TARGET_USER" == "npin" ]]; then
    # NPIN gets: tslib_sota T1 (8 models), foundation_new, deep_backfill, NF gap-fill
    echo ""
    echo "--- [npin] 8a: tslib_sota shard T1 (8 models, gpu SNT 3d) ---"
    TSLIB_T1="TimeFilter,WPMixer,MultiPatchFormer,MSGNet,PAttn,MambaSimple,Crossformer,Koopa"
    for ta in "${ALL_TA[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "s8n_tsT1_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "$GPU_QOS" "3-00:00:00" "320G" "14" "gpu:volta:1" \
            "$TASK" "tslib_sota" "$ABL" "$TSLIB_T1"
    done

    echo ""
    echo "--- [npin] 8b: foundation_new (4 models, gpu SNT 2d) ---"
    FM_NEW="Sundial,TTM,TimerXL,TimesFM2"
    for ta in "${ALL_TA[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "s8n_fmN_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "$GPU_QOS" "2-00:00:00" "256G" "14" "gpu:volta:1" \
            "$TASK" "foundation" "$ABL" "$FM_NEW"
    done

    echo ""
    echo "--- [npin] 8c: deep_classical backfill (5 models, gpu SNT 2d) ---"
    DC_BACK="GRU,LSTM,TCN,MLP,DilatedRNN"
    for ta in "${ALL_TA[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "s8n_dcB_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "$GPU_QOS" "2-00:00:00" "256G" "14" "gpu:volta:1" \
            "$TASK" "deep_classical" "$ABL" "$DC_BACK"
    done

    echo ""
    echo "--- [npin] 8d: NF transformer gap-fill (10 models, gpu SNT 2d) ---"
    NF_GAP="PatchTST,iTransformer,TimesNet,TSMixer,Informer,Autoformer,FEDformer,VanillaTransformer,TiDE,NBEATSx"
    for abl in "core_only" "core_edgar" "full"; do
        submit_job "s8n_nfG_t3_$(abl_abbrev $abl)" \
            "gpu" "$GPU_QOS" "2-00:00:00" "320G" "14" "gpu:volta:1" \
            "task3_risk_adjust" "transformer_sota" "$abl" "$NF_GAP"
    done
    for task in "task1_outcome" "task2_forecast"; do
        for abl in "core_edgar" "full"; do
            submit_job "s8n_nfG_$(task_abbrev $task)_$(abl_abbrev $abl)" \
                "gpu" "$GPU_QOS" "2-00:00:00" "320G" "14" "gpu:volta:1" \
                "$task" "transformer_sota" "$abl" "$NF_GAP"
        done
    done

    echo ""
    echo "--- [npin] 8d: TimesFM gap-fill (gpu SNT 1d) ---"
    for ta in "${ALL_TA[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "s8n_fmTF_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "$GPU_QOS" "1-00:00:00" "256G" "14" "gpu:volta:1" \
            "$TASK" "foundation" "$ABL" "TimesFM"
    done

elif [[ "$TARGET_USER" == "cfisch" ]]; then
    # CFISCH gets: tslib_sota T2 (6 models), autofit V736 + gap-fill
    echo ""
    echo "--- [cfisch] 8a: tslib_sota shard T2 (6 models, gpu SNT 3d) ---"
    TSLIB_T2="FreTS,MICN,SegRNN,NonstationaryTransformer,FiLM,SCINet"
    for ta in "${ALL_TA[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "s8c_tsT2_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "$GPU_QOS" "3-00:00:00" "320G" "14" "gpu:volta:1" \
            "$TASK" "tslib_sota" "$ABL" "$TSLIB_T2"
    done

    echo ""
    echo "--- [cfisch] 8e: autofit V1-V7 gap-fill (batch SNT 2d) ---"
    AF_FILL="AutoFitV1,AutoFitV2,AutoFitV2E,AutoFitV3,AutoFitV3E,AutoFitV3Max,AutoFitV4,AutoFitV5,AutoFitV6,AutoFitV7"
    for ta in "${ALL_TA[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "s8c_afG_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "batch" "$BATCH_QOS" "2-00:00:00" "112G" "28" "" \
            "$TASK" "autofit" "$ABL" "$AF_FILL"
    done

    echo ""
    echo "--- [cfisch] 8e: autofit V72-V736 gap-fill (batch SNT 2d) ---"
    AF_V7X="AutoFitV72,AutoFitV73,AutoFitV731,AutoFitV733,AutoFitV736"
    for ta in "${ALL_TA[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "s8c_af7_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "batch" "$BATCH_QOS" "2-00:00:00" "112G" "28" "" \
            "$TASK" "autofit" "$ABL" "$AF_V7X"
    done
fi

echo ""
echo "================================================================"
if $DRY_RUN; then
    echo "DRY RUN — no jobs submitted"
else
    echo "Jobs submitted on ${TARGET_USER}: ${TOTAL}"
fi
echo "QOS: iris-snt-long (100 max concurrent, priority=100)"
echo "================================================================"
