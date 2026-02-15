#!/usr/bin/env bash
# ============================================================================
# Block 3 AutoFit V7.1 Extreme Submission Script
# Stage-A: pilot grid search (fairness-first, per-model kwargs overrides)
# Stage-B: full comparison with selected V7.1 variant
# ============================================================================
#
# Usage:
#   bash scripts/submit_phase7_v71_extreme.sh --pilot
#   bash scripts/submit_phase7_v71_extreme.sh --full --v71-variant g03
#   bash scripts/submit_phase7_v71_extreme.sh --pilot --dry-run
#   ALLOW_UNSAFE_SKIP_PREFLIGHT=1 bash scripts/submit_phase7_v71_extreme.sh --full --v71-variant g02 --skip-preflight
# ============================================================================

set -euo pipefail

REPO="/home/users/npin/repo_root"
STAMP="20260203_225620"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
MODE="pilot"
DRY_RUN=false
BEST_V71_VARIANT="g01"
SKIP_PREFLIGHT=false

for arg in "$@"; do
    case "$arg" in
        --pilot)
            MODE="pilot"
            ;;
        --full)
            MODE="full"
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --v71-variant=*)
            BEST_V71_VARIANT="${arg#*=}"
            ;;
        --run-tag=*)
            RUN_TAG="${arg#*=}"
            ;;
        --skip-preflight)
            SKIP_PREFLIGHT=true
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

if $SKIP_PREFLIGHT && [[ "${ALLOW_UNSAFE_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Refusing --skip-preflight without explicit override."
    echo "If you must bypass the gate, rerun with:"
    echo "  ALLOW_UNSAFE_SKIP_PREFLIGHT=1 bash scripts/submit_phase7_v71_extreme.sh --${MODE} --skip-preflight"
    exit 2
fi

OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7_v71extreme_${RUN_TAG}"
LOG_DIR="/work/projects/eint/logs/phase7_v71extreme_${RUN_TAG}"
SLURM_DIR="${REPO}/.slurm_scripts/phase7_v71extreme_${RUN_TAG}"
ACCOUNT="yves.letraon"
PRESET="full"
SEED=42
TOTAL_SUBMITTED=0

mkdir -p "$LOG_DIR" "$SLURM_DIR"

if $DRY_RUN; then
    echo "=== DRY RUN MODE ==="
fi
echo "=== Mode: ${MODE} | Run tag: ${RUN_TAG} | Output: ${OUTPUT_BASE} ==="

if ! $DRY_RUN && ! $SKIP_PREFLIGHT; then
    PREFLIGHT_VARIANT="${BEST_V71_VARIANT}"
    if [[ "$MODE" == "pilot" ]]; then
        # Pilot submits all variants; keep smoke/audit anchored on g02.
        PREFLIGHT_VARIANT="g02"
    fi
    echo "--- Preflight gate: variant=${PREFLIGHT_VARIANT} ---"
    bash scripts/preflight_block3_v71_gate.sh --v71-variant="${PREFLIGHT_VARIANT}"
fi

read -r -d '' ENV_PREAMBLE << 'ENVBLOCK' || true
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(which python3)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"
ENVBLOCK

abl_abbrev() {
    case "$1" in
        core_only)  echo "co";;
        core_text)  echo "ct";;
        core_edgar) echo "ce";;
        full)       echo "fu";;
        *)          echo "${1:0:2}";;
    esac
}

task_abbrev() {
    case "$1" in
        task1_outcome)     echo "t1";;
        task2_forecast)    echo "t2";;
        task3_risk_adjust) echo "t3";;
        *)                 echo "${1:0:2}";;
    esac
}

submit_job() {
    local JOB_NAME="$1"
    local PARTITION="$2"
    local QOS="$3"
    local TIME="$4"
    local MEM="$5"
    local CPUS="$6"
    local GRES="$7"
    local CONSTRAINT="$8"
    local TASK="$9"
    local CATEGORY="${10}"
    local ABLATION="${11}"
    local MODELS="${12}"
    local SUBDIR="${13}"
    local MODEL_KWARGS_JSON="${14:-}"

    local OUTDIR="${OUTPUT_BASE}/${SUBDIR}/${TASK}/${CATEGORY}/${ABLATION}"
    local SCRIPT="${SLURM_DIR}/${JOB_NAME}.sh"
    local GRES_LINE=""
    local CONSTRAINT_LINE=""
    local KW_FILE=""

    if [[ -n "$GRES" ]]; then
        GRES_LINE="#SBATCH --gres=${GRES}"
    fi
    if [[ -n "$CONSTRAINT" ]]; then
        CONSTRAINT_LINE="#SBATCH --constraint=${CONSTRAINT}"
    fi
    if [[ -n "$MODEL_KWARGS_JSON" ]]; then
        KW_FILE="${SLURM_DIR}/${JOB_NAME}_model_kwargs.json"
        printf '%s\n' "$MODEL_KWARGS_JSON" > "$KW_FILE"
    fi

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
${CONSTRAINT_LINE}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=B:USR1@120

set -e
${ENV_PREAMBLE}

echo "Task: ${TASK} | Category: ${CATEGORY} | Ablation: ${ABLATION}"
echo "Models: ${MODELS:-ALL}"
echo "Preset: ${PRESET} | Seed: ${SEED}"
echo "Output: ${OUTDIR}"
echo "============================================================"

CMD=(
    python3 scripts/run_block3_benchmark_shard.py
    --task ${TASK}
    --category ${CATEGORY}
    --ablation ${ABLATION}
    --preset ${PRESET}
    --output-dir ${OUTDIR}
    --seed ${SEED}
    --no-verify-first
)
if [[ -n "${MODELS}" ]]; then
    CMD+=(--models "${MODELS}")
fi
if [[ -n "${KW_FILE}" ]]; then
    CMD+=(--model-kwargs-file "${KW_FILE}")
fi
"\${CMD[@]}"

echo "Done: \$(date -Iseconds)"
SLURM_EOF

    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "  [DRY] ${JOB_NAME} -> ${PARTITION}/${QOS} ${TIME} ${MEM} ${CPUS}c ${GRES:-cpu} ${CONSTRAINT:-}"
    else
        local JID
        JID=$(sbatch "$SCRIPT" 2>&1 | grep -oP '\d+')
        echo "  [SUB] ${JOB_NAME} -> JobID ${JID}"
        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
    fi
}

TASK1_ABLATIONS=("core_only" "core_text" "core_edgar" "full")
TASK2_ABLATIONS=("core_only" "core_text" "core_edgar" "full")
TASK3_ABLATIONS=("core_only" "core_edgar" "full")
ALL_TASKS_ABLATIONS=()
for abl in "${TASK1_ABLATIONS[@]}"; do
    ALL_TASKS_ABLATIONS+=("task1_outcome:${abl}")
done
for abl in "${TASK2_ABLATIONS[@]}"; do
    ALL_TASKS_ABLATIONS+=("task2_forecast:${abl}")
done
for abl in "${TASK3_ABLATIONS[@]}"; do
    ALL_TASKS_ABLATIONS+=("task3_risk_adjust:${abl}")
done

ML_REF_MODELS="RandomForest,ExtraTrees,LightGBM,XGBoost"
ML_REF_KW='{"RandomForest":{"n_estimators":900,"min_samples_leaf":2},"ExtraTrees":{"n_estimators":1200,"min_samples_leaf":2},"LightGBM":{"n_estimators":3000,"learning_rate":0.03,"num_leaves":255,"min_child_samples":30,"subsample":0.9,"colsample_bytree":0.85},"XGBoost":{"n_estimators":3000,"learning_rate":0.03,"max_depth":10,"min_child_weight":2.0,"subsample":0.9,"colsample_bytree":0.85,"reg_lambda":1.5}}'
DEEP_REF_MODELS="NBEATS,NHITS,TFT,DeepAR"
TS_REF_MODELS="PatchTST,iTransformer,TimesNet,TSMixer,Informer"
FD_REF_MODELS="Chronos2,Moirai2,TimesFM"
AF_BASELINE_MODELS="AutoFitV6,AutoFitV7"

declare -A V71_VARIANTS
V71_VARIANTS["g01"]='{"AutoFitV71":{"top_k":8,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":true,"enable_regime_retrieval":true}}'
V71_VARIANTS["g02"]='{"AutoFitV71":{"top_k":10,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":true,"enable_regime_retrieval":true}}'
V71_VARIANTS["g03"]='{"AutoFitV71":{"top_k":6,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":true,"enable_regime_retrieval":true}}'
V71_VARIANTS["g04"]='{"AutoFitV71":{"top_k":8,"min_ensemble_size_heavy_tail":3,"dynamic_weighting":true,"enable_regime_retrieval":false}}'
V71_VARIANTS["g05"]='{"AutoFitV71":{"top_k":8,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":false,"enable_regime_retrieval":true}}'

if [[ "$MODE" == "pilot" ]]; then
    echo "--- Stage-A Pilot: refs + autofit baseline + V7.1 grid ---"

    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "p7x_mlr_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" "" \
            "$TASK" "ml_tabular" "$ABL" "$ML_REF_MODELS" "pilot/ml_refs" "$ML_REF_KW"
    done

    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "p7x_dpr_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "iris-gpu-long" "2-00:00:00" "320G" "14" "gpu:volta:1" "volta32" \
            "$TASK" "deep_classical" "$ABL" "$DEEP_REF_MODELS" "pilot/deep_refs" ""
    done

    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "p7x_tsr_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "iris-gpu-long" "2-00:00:00" "320G" "14" "gpu:volta:1" "volta32" \
            "$TASK" "transformer_sota" "$ABL" "$TS_REF_MODELS" "pilot/ts_refs" ""
    done

    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "p7x_fdr_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "iris-gpu-long" "2-00:00:00" "320G" "14" "gpu:volta:1" "volta32" \
            "$TASK" "foundation" "$ABL" "$FD_REF_MODELS" "pilot/foundation_refs" ""
    done

    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "p7x_afb_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" "" \
            "$TASK" "autofit" "$ABL" "$AF_BASELINE_MODELS" "pilot/autofit_baseline" ""
    done

    for variant in g01 g02 g03 g04 g05; do
        KW="${V71_VARIANTS[$variant]}"
        for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
            IFS=':' read -r TASK ABL <<< "$ta"
            submit_job "p7x_${variant}_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
                "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" "" \
                "$TASK" "autofit" "$ABL" "AutoFitV71" "pilot/v71_${variant}" "$KW"
        done
    done
fi

if [[ "$MODE" == "full" ]]; then
    if [[ -z "${V71_VARIANTS[$BEST_V71_VARIANT]:-}" ]]; then
        echo "Unknown --v71-variant=${BEST_V71_VARIANT}. Available: g01,g02,g03,g04,g05"
        exit 1
    fi
    echo "--- Stage-B Full: selected V7.1 variant=${BEST_V71_VARIANT} + SOTA refs ---"

    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "p7xF_mlr_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" "" \
            "$TASK" "ml_tabular" "$ABL" "$ML_REF_MODELS" "full/ml_refs" "$ML_REF_KW"
    done

    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "p7xF_dpr_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "iris-gpu-long" "2-00:00:00" "320G" "14" "gpu:volta:1" "volta32" \
            "$TASK" "deep_classical" "$ABL" "$DEEP_REF_MODELS" "full/deep_refs" ""
    done

    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "p7xF_tsr_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "iris-gpu-long" "2-00:00:00" "320G" "14" "gpu:volta:1" "volta32" \
            "$TASK" "transformer_sota" "$ABL" "$TS_REF_MODELS" "full/ts_refs" ""
    done

    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "p7xF_fdr_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "iris-gpu-long" "2-00:00:00" "320G" "14" "gpu:volta:1" "volta32" \
            "$TASK" "foundation" "$ABL" "$FD_REF_MODELS" "full/foundation_refs" ""
    done

    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "p7xF_afb_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" "" \
            "$TASK" "autofit" "$ABL" "$AF_BASELINE_MODELS" "full/autofit_baseline" ""
    done

    KW="${V71_VARIANTS[$BEST_V71_VARIANT]}"
    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job "p7xF_${BEST_V71_VARIANT}_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" "" \
            "$TASK" "autofit" "$ABL" "AutoFitV71" "full/v71_${BEST_V71_VARIANT}" "$KW"
    done
fi

echo ""
echo "================================================================"
if $DRY_RUN; then
    echo "DRY RUN COMPLETE"
else
    echo "Total jobs submitted: ${TOTAL_SUBMITTED}"
fi
echo "Mode: ${MODE}"
echo "Run tag: ${RUN_TAG}"
echo "Output: ${OUTPUT_BASE}"
echo "Logs: ${LOG_DIR}"
echo "Monitor: squeue -u ${USER}"
echo "================================================================"
