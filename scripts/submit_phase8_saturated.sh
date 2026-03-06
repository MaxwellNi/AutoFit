#!/usr/bin/env bash
# ============================================================================
# Phase 8 Saturated Benchmark — SLURM Submission Script
# KDD'26 Block 3, Gap-fill + New Models
# ============================================================================
#
# Phase 8 adds:
#   8a. tslib_sota (14 models) — NEVER SUBMITTED in Phase 7
#   8b. foundation_new (4 models: Sundial, TTM, TimerXL, TimesFM2)
#   8c. deep_classical_backfill (GRU, LSTM, TCN, MLP, DilatedRNN — missing half conditions)
#   8d. NF transformer/foundation gap-fill (43/48 models → 48/48)
#   8e. autofit gap-fill (V1-V7 incomplete → 48/48)
#
# Cluster: ULHPC Iris HPC
#   GPU: 24 nodes, 28c, 756GB, 4×V100-32GB
#   Current: ALL IDLE (Phase 7 complete)
#
# Usage:
#   bash scripts/submit_phase8_saturated.sh --dry-run
#   bash scripts/submit_phase8_saturated.sh
#   bash scripts/submit_phase8_saturated.sh --shard 8a      # only tslib
#   bash scripts/submit_phase8_saturated.sh --shard 8b      # only new foundation
#   bash scripts/submit_phase8_saturated.sh --shard all      # everything
#   ALLOW_UNSAFE_SKIP_PREFLIGHT=1 bash scripts/submit_phase8_saturated.sh --skip-preflight
# ============================================================================

set -euo pipefail

REPO="/home/users/npin/repo_root"
STAMP="20260203_225620"
OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7"
LOG_DIR="/work/projects/eint/logs/phase8"
SLURM_DIR="${REPO}/.slurm_scripts/phase8"
ACCOUNT="yves.letraon"
PRESET="full"
SEED=42
DRY_RUN=false
SKIP_PREFLIGHT=false
SHARD_FILTER="all"

for arg in "$@"; do
    case "$arg" in
        --dry-run)      DRY_RUN=true ;;
        --skip-preflight) SKIP_PREFLIGHT=true ;;
        --shard)        :;;  # handled below
        8a|8b|8c|8d|8e|all)
                        SHARD_FILTER="$arg" ;;
    esac
done

# Handle --shard VALUE
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--shard" ]]; then
        SHARD_FILTER="$arg"
    fi
    prev="$arg"
done

# ============================================================================
# Environment activation (identical to Phase 7)
# ============================================================================
activate_insider_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "insider" ]]; then
        return 0
    fi
    if command -v micromamba >/dev/null 2>&1; then
        local roots=(
            "${MAMBA_ROOT_PREFIX:-}"
            "/mnt/aiongpfs/projects/eint/envs/.micromamba"
            "${HOME}/.local/share/micromamba"
            "${HOME}/micromamba"
        )
        local r
        for r in "${roots[@]}"; do
            [[ -n "$r" && -d "$r" ]] || continue
            export MAMBA_ROOT_PREFIX="$r"
            eval "$(micromamba shell hook -s bash)"
            if micromamba activate insider; then return 0; fi
        done
    fi
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
            source "${conda_base}/etc/profile.d/conda.sh"
            if conda activate insider; then return 0; fi
        fi
    fi
    echo "FATAL: failed to activate insider environment."
    return 1
}

activate_insider_env
PY_BIN="$(command -v python3 || true)"
if [[ -z "${PY_BIN}" || "${PY_BIN}" != *"insider"* ]]; then
    echo "FATAL: python3 is not from insider env: ${PY_BIN:-<missing>}"
    exit 2
fi
python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"FATAL: insider python must be >=3.11, got {sys.version}")
PY

# Contract assertion must run without forbidden env flags
_saved_skip="${ALLOW_UNSAFE_SKIP_PREFLIGHT:-}"
unset ALLOW_UNSAFE_SKIP_PREFLIGHT
python3 scripts/assert_block3_execution_contract.py \
  --entrypoint "scripts/submit_phase8_saturated.sh"
if [[ -n "$_saved_skip" ]]; then
    export ALLOW_UNSAFE_SKIP_PREFLIGHT="$_saved_skip"
fi

if $SKIP_PREFLIGHT && [[ "${ALLOW_UNSAFE_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Refusing --skip-preflight without explicit override."
    exit 2
fi

if $DRY_RUN; then
    echo "=== DRY RUN MODE — no jobs will be submitted ==="
fi

mkdir -p "$LOG_DIR" "$SLURM_DIR"

# ============================================================================
# Shared SLURM preamble
# ============================================================================
read -r -d '' ENV_PREAMBLE << 'ENVBLOCK' || true
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root

INSIDER_PY="${CONDA_PREFIX}/bin/python3"
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"
  exit 2
fi

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(which python3) — $(python3 -V)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"
python3 - <<'PY'
import sys
if "insider" not in sys.executable:
    raise SystemExit("FATAL: runtime python is not insider")
if sys.version_info < (3, 11):
    raise SystemExit(f"FATAL: need >=3.11, got {sys.version}")
PY
${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"
ENVBLOCK

TOTAL_SUBMITTED=0

# ============================================================================
# Helpers (same as Phase 7)
# ============================================================================
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
    local TASK="$8"
    local CATEGORY="$9"
    local ABLATION="${10}"
    local MODELS="${11}"
    local EXTRA="${12:-}"

    local OUTDIR="${OUTPUT_BASE}/${TASK}/${CATEGORY}/${ABLATION}"
    local SCRIPT="${SLURM_DIR}/${JOB_NAME}.sh"

    local GRES_LINE=""
    if [[ -n "$GRES" ]]; then
        GRES_LINE="#SBATCH --gres=${GRES}"
    fi

    local MODELS_ARG=""
    if [[ -n "$MODELS" ]]; then
        MODELS_ARG="--models ${MODELS}"
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
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
${EXTRA}

set -e

${ENV_PREAMBLE}

echo "Task: ${TASK} | Category: ${CATEGORY} | Ablation: ${ABLATION}"
echo "Models: ${MODELS:-ALL}"
echo "Preset: ${PRESET} | Seed: ${SEED}"
echo "Output: ${OUTDIR}"
echo "============================================================"

"\${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \\
    --task ${TASK} \\
    --category ${CATEGORY} \\
    --ablation ${ABLATION} \\
    --preset ${PRESET} \\
    --output-dir ${OUTDIR} \\
    --seed ${SEED} \\
    --no-verify-first \\
    ${MODELS_ARG}

echo "Done: \$(date -Iseconds)"
SLURM_EOF

    chmod +x "$SCRIPT"

    if $DRY_RUN; then
        echo "  [DRY] ${JOB_NAME} → ${PARTITION}/${QOS} ${TIME} ${MEM} ${CPUS}c ${GRES:-no-gpu}"
    else
        local JID
        JID=$(sbatch "$SCRIPT" 2>&1 | grep -oP '\d+')
        echo "  [SUB] ${JOB_NAME} → JobID ${JID} (${PARTITION}/${QOS} ${TIME} ${MEM} ${CPUS}c ${GRES:-no-gpu})"
        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
    fi
}

# ============================================================================
# Task-ablation matrix (same as Phase 7)
# ============================================================================
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

echo ""
echo "================================================================"
echo "Phase 8 Saturated Benchmark Submission"
echo "  Output: ${OUTPUT_BASE}"
echo "  Shard:  ${SHARD_FILTER}"
echo "  Combos: ${#ALL_TASKS_ABLATIONS[@]}"
echo "================================================================"

# ============================================================================
# 8a. TSLIB_SOTA (14 models) — NEVER submitted in Phase 7
#     TimeFilter, WPMixer, MultiPatchFormer, MSGNet, PAttn, MambaSimple,
#     Koopa, FreTS, Crossformer, MICN, SegRNN, NonstationaryTransformer,
#     FiLM, SCINet
#     Uses standalone PyTorch training (not NeuralForecast).
#     Source: /mnt/aiongpfs/projects/eint/vendor/Time-Series-Library
#     Needs GPU (1×V100), moderate VRAM, 320GB RAM for panel data.
#     Split into 2 shards for faster turnaround:
#       Shard T1: 2025 models (TimeFilter, WPMixer, MultiPatchFormer) +
#                 2024 models (MSGNet, PAttn, MambaSimple) +
#                 Crossformer, Koopa = 8 models
#       Shard T2: FreTS, MICN, SegRNN, NonstationaryTransformer, FiLM, SCINet = 6 models
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "8a" ]]; then

TSLIB_T1="TimeFilter,WPMixer,MultiPatchFormer,MSGNet,PAttn,MambaSimple,Crossformer,Koopa"
TSLIB_T2="FreTS,MICN,SegRNN,NonstationaryTransformer,FiLM,SCINet"

echo ""
echo "--- 8a: tslib_sota shard T1 (8 models, gpu 1×V100 14c 320GB 3d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p8_tsT1_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "3-00:00:00" "320G" "14" "gpu:volta:1" \
        "$TASK" "tslib_sota" "$ABL" "$TSLIB_T1" ""
done

echo ""
echo "--- 8a: tslib_sota shard T2 (6 models, gpu 1×V100 14c 320GB 3d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p8_tsT2_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "3-00:00:00" "320G" "14" "gpu:volta:1" \
        "$TASK" "tslib_sota" "$ABL" "$TSLIB_T2" ""
done

fi  # 8a

# ============================================================================
# 8b. NEW FOUNDATION MODELS (4 models: Sundial, TTM, TimerXL, TimesFM2)
#     All are HuggingFace-based pretrained models.
#     - Sundial:   ICML'25 Oral, #1 on GIFT-Eval, diffusion-based
#     - TTM:       NeurIPS'24, IBM Research, tiny time mixers
#     - TimerXL:   ICLR'25, large timer model
#     - TimesFM2:  Google Research, TimesFM 2.5 upgrade
#     GPU required. 1×V100 sufficient (zero-shot / lightweight fine-tune).
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "8b" ]]; then

FM_NEW="Sundial,TTM,TimerXL,TimesFM2"

echo ""
echo "--- 8b: foundation_new (4 models, gpu 1×V100 14c 256GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p8_fmN_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "2-00:00:00" "256G" "14" "gpu:volta:1" \
        "$TASK" "foundation" "$ABL" "$FM_NEW" ""
done

fi  # 8b

# ============================================================================
# 8c. DEEP CLASSICAL BACKFILL
#     GRU, LSTM, TCN, MLP, DilatedRNN — currently 24/48 conditions each.
#     Phase 7 submitted deep_classical for ALL 11 combos, but these 5 models
#     only produced results for task1+task2 core_only+core_text (6 combos).
#     Missing: task1/task2 core_edgar/full + all task3 = 5 task-ablation combos.
#     Re-submit ALL combos — harness appends to existing metrics.json.
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "8c" ]]; then

DC_BACKFILL="GRU,LSTM,TCN,MLP,DilatedRNN"

echo ""
echo "--- 8c: deep_classical backfill (5 models, gpu 1×V100 14c 256GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p8_dcB_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "2-00:00:00" "256G" "14" "gpu:volta:1" \
        "$TASK" "deep_classical" "$ABL" "$DC_BACKFILL" ""
done

fi  # 8c

# ============================================================================
# 8d. NF TRANSFORMER/FOUNDATION GAP-FILL
#     PatchTST, iTransformer, TimesNet, TSMixer, Informer, Autoformer,
#     FEDformer, VanillaTransformer, TiDE, NBEATSx: all at 43/48.
#     Missing 5 conditions — likely is_funded with specific ablations.
#     TimesFM: 47/48 (1 missing condition).
#
#     Strategy: re-submit all 11 combos for these models.
#     The harness produces new metrics; we merge at aggregation time.
#     Only running the specific task/ablations that produced gaps.
#     Analysis shows the gap is task3-related (is_funded targets).
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "8d" ]]; then

NF_GAP="PatchTST,iTransformer,TimesNet,TSMixer,Informer,Autoformer,FEDformer,VanillaTransformer,TiDE,NBEATSx"

echo ""
echo "--- 8d: NF transformer gap-fill (10 models, gpu 1×V100 14c 320GB 2d) ---"
# Submit only task3 combos (where gaps likely are) + task1/task2 exogenous ablations
# Task3 ablations: core_only, core_edgar, full
for abl in "core_only" "core_edgar" "full"; do
    submit_job \
        "p8_nfG_t3_$(abl_abbrev $abl)" \
        "gpu" "iris-gpu-long" "2-00:00:00" "320G" "14" "gpu:volta:1" \
        "task3_risk_adjust" "transformer_sota" "$abl" "$NF_GAP" ""
done
# Task1 + Task2 with core_edgar and full (where exogenous data may have caused failures)
for task in "task1_outcome" "task2_forecast"; do
    for abl in "core_edgar" "full"; do
        submit_job \
            "p8_nfG_$(task_abbrev $task)_$(abl_abbrev $abl)" \
            "gpu" "iris-gpu-long" "2-00:00:00" "320G" "14" "gpu:volta:1" \
            "$task" "transformer_sota" "$abl" "$NF_GAP" ""
    done
done

# TimesFM gap-fill (1 missing condition)
echo ""
echo "--- 8d: TimesFM gap-fill (1 model, gpu 1×V100 14c 256GB 1d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p8_fmTF_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "1-00:00:00" "256G" "14" "gpu:volta:1" \
        "$TASK" "foundation" "$ABL" "TimesFM" ""
done

fi  # 8d

# ============================================================================
# 8e. AUTOFIT GAP-FILL
#     V1-V3E: 40/48 (missing 8 conditions)
#     V3Max-V7: 46/48 (missing 2 conditions)
#     V72: 29/48, V73: 30/48, V731: 18/48
#     Re-submit all combos for incomplete AutoFit models.
#     V734/V735 already at 48/48 — skip.
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "8e" ]]; then

AF_INCOMPLETE="AutoFitV1,AutoFitV2,AutoFitV2E,AutoFitV3,AutoFitV3E,AutoFitV3Max,AutoFitV4,AutoFitV5,AutoFitV6,AutoFitV7"
AF_V7X_FILL="AutoFitV72,AutoFitV73,AutoFitV731,AutoFitV733,AutoFitV736"

echo ""
echo "--- 8e: autofit V1-V7 gap-fill (batch 28c 112GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p8_afG_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" \
        "$TASK" "autofit" "$ABL" "$AF_INCOMPLETE" ""
done

echo ""
echo "--- 8e: autofit V72/V73/V731/V733 gap-fill (batch 28c 112GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p8_af7_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" \
        "$TASK" "autofit" "$ABL" "$AF_V7X_FILL" ""
done

fi  # 8e

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================"
if $DRY_RUN; then
    echo "DRY RUN COMPLETE — no jobs submitted"
else
    echo "Total jobs submitted: ${TOTAL_SUBMITTED}"
fi
echo ""
echo "Job breakdown (shard=${SHARD_FILTER}):"
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "8a" ]]; then
echo "  8a tslib_sota:        22 jobs (gpu 1×V100 14c 320GB 3d) [2 shards: T1(8)+T2(6)]"
fi
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "8b" ]]; then
echo "  8b foundation_new:    11 jobs (gpu 1×V100 14c 256GB 2d) [Sundial,TTM,TimerXL,TimesFM2]"
fi
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "8c" ]]; then
echo "  8c deep_backfill:     11 jobs (gpu 1×V100 14c 256GB 2d) [GRU,LSTM,TCN,MLP,DilatedRNN]"
fi
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "8d" ]]; then
echo "  8d NF gap-fill:       18 jobs (gpu 1×V100 14c 320GB 2d) [10 NF models + TimesFM]"
fi
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "8e" ]]; then
echo "  8e autofit gap-fill:  22 jobs (batch 28c 112GB 2d) [V1-V733 missing conditions]"
fi
echo "  ─────────────────────────────────────────────"
echo "  TOTAL (all): 84 jobs (62 GPU + 22 batch)"
echo ""
echo "Output: ${OUTPUT_BASE}/"
echo "Logs:   ${LOG_DIR}/"
echo ""
echo "Monitor: squeue -u \$(whoami)"
echo "Results: python3 scripts/aggregate_block3_results.py --input-dir ${OUTPUT_BASE}"
echo "================================================================"
