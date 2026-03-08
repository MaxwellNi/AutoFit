#!/usr/bin/env bash
# ============================================================================
# Phase 9 — Ultimate Fair Benchmark
# KDD'26 Block 3, Full Re-Run with Bug Fixes + New Models
# ============================================================================
#
# Phase 9 rationale (invalidates ALL prior Phase 7/8 results):
#   1. CRITICAL: tslib_sota single-window bug → per-entity prediction
#   2. CRITICAL: Foundation prediction_length=7 hardcode → actual horizon
#   3. CRITICAL: Moirai/Moirai2 50-entity cap removed → ALL entities
#   4. CRITICAL: LagLlama 50-entity cap removed + horizon fix
#   5. FIX: ml_tabular single-horizon restriction removed → ALL horizons
#   6. FIX: NF configs use original committed values (no equalization applied)
#   7. FIX: BiTCN hidden_size 16→128, TimesNet scaler robust
#   8. FIX: tslib epochs 50→100, patience 7→15, training normalization
#   9. NEW: +6 tslib_sota (ETSformer, LightTS, Pyraformer, Reformer, TiRex, Mamba)
#  10. NEW: +10 statistical (Croston×3, DynOptTheta, AutoCES, Holt, HoltWinters, Naive, HistAvg, WinAvg)
#  11. NEW: +2 irregular (BRITS, CSDI)
#
# Model census:
#   statistical     : 15 models (was 5)
#   ml_tabular      : 15 models (same, now ALL horizons)
#   deep_classical  :  9 models
#   transformer_sota: 23 models
#   foundation      : 11 models (Chronos, ChronosBolt, Chronos2, Moirai,
#                      MoiraiLarge, Moirai2, Timer, TimeMoE, MOMENT, LagLlama, TimesFM)
#   irregular       :  4 models (was 2)
#   tslib_sota      : 20 models (was 14)
#   autofit         :  3 models (V734, V735, V736 only — fair subset)
#   TOTAL           : 100 SOTA models × 104 conditions
#
# Conditions (104 total):
#   task1_outcome:     3 targets × 4 horizons × 4 ablations = 48
#   task2_forecast:    2 targets × 4 horizons × 4 ablations = 32
#   task3_risk_adjust: 2 targets × 4 horizons × 3 ablations = 24
#
# Cluster: ULHPC Iris HPC
#   GPU: 24 nodes, 28c, 756GB, 4×V100-32GB
#   Bigmem: 4 nodes, 112c, 3TB
#   QOS: normal (2-day wall), long (3-day wall)
#
# Usage:
#   bash scripts/submit_phase9_ultimate.sh --dry-run
#   bash scripts/submit_phase9_ultimate.sh
#   bash scripts/submit_phase9_ultimate.sh --shard 9a
#   bash scripts/submit_phase9_ultimate.sh --shard all
# ============================================================================

set -euo pipefail

REPO="/home/users/npin/repo_root"
STAMP="20260203_225620"
OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase9"
LOG_DIR="/work/projects/eint/logs/phase9"
SLURM_DIR="${REPO}/.slurm_scripts/phase9"
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
        --shard)        :;;
        9a|9b|9c|9d|9e|9f|9g|9h|all)
                        SHARD_FILTER="$arg" ;;
    esac
done

prev=""
for arg in "$@"; do
    if [[ "$prev" == "--shard" ]]; then
        SHARD_FILTER="$arg"
    fi
    prev="$arg"
done

# ============================================================================
# Environment activation
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

_saved_skip="${ALLOW_UNSAFE_SKIP_PREFLIGHT:-}"
unset ALLOW_UNSAFE_SKIP_PREFLIGHT
python3 scripts/assert_block3_execution_contract.py \
  --entrypoint "scripts/submit_phase9_ultimate.sh"
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
# Helpers
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
echo "Phase: 9 (Ultimate Fair Benchmark)"
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
# Task-ablation matrix: 11 combos = 104 conditions
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
echo "Phase 9 ULTIMATE FAIR BENCHMARK"
echo "  Output: ${OUTPUT_BASE}"
echo "  Shard:  ${SHARD_FILTER}"
echo "  Combos: ${#ALL_TASKS_ABLATIONS[@]} task×ablation"
echo "  Changes: tslib per-entity fix, foundation horizon fix,"
echo "           Moirai/LagLlama entity cap removed, NF budget equalized,"
echo "           +18 new models (6 tslib + 10 stat + 2 irregular)"
echo "================================================================"

# ============================================================================
# 9a. STATISTICAL (15 models) — CPU-only, fast
#     AutoARIMA, AutoETS, AutoTheta, MSTL, SF_SeasonalNaive,
#     CrostonClassic, CrostonOptimized, CrostonSBA,
#     DynamicOptimizedTheta, AutoCES, Holt, HoltWinters,
#     Naive, HistoricAverage, WindowAverage
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9a" ]]; then

echo ""
echo "--- 9a: statistical (15 models, batch 28c 112GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p9_stat_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "batch" "normal" "2-00:00:00" "112G" "28" "" \
        "$TASK" "statistical" "$ABL" "" ""
done

fi  # 9a

# ============================================================================
# 9b. ML_TABULAR (15 models) — CPU-only, NOW all 4 horizons
#     Was restricted to single horizon in Phase 7/8. Now runs ALL horizons.
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9b" ]]; then

echo ""
echo "--- 9b: ml_tabular (15 models × 4 horizons, batch 28c 112GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p9_mlT_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "batch" "normal" "2-00:00:00" "112G" "28" "" \
        "$TASK" "ml_tabular" "$ABL" "" ""
done

fi  # 9b

# ============================================================================
# 9c. DEEP_CLASSICAL (9 models) — GPU
#     NBEATS, NHITS, TFT, DeepAR, GRU, LSTM, TCN, MLP, DilatedRNN
#     All use original committed NF configs.
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9c" ]]; then

echo ""
echo "--- 9c: deep_classical (9 models, gpu 1×V100 14c 256GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p9_deep_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "normal" "2-00:00:00" "256G" "14" "gpu:volta:1" \
        "$TASK" "deep_classical" "$ABL" "" ""
done

fi  # 9c

# ============================================================================
# 9d. TRANSFORMER_SOTA (23 models) — GPU, 2 shards
#     Shard A (12): PatchTST, iTransformer, TimesNet, TSMixer, Informer,
#                   Autoformer, FEDformer, VanillaTransformer, TiDE, NBEATSx,
#                   BiTCN, KAN
#     Shard B (11): RMoK, SOFTS, StemGNN, DLinear, NLinear, TimeMixer,
#                   TimeXer, TSMixerx, xLSTM, TimeLLM, DeepNPTS
#     All use original committed NF configs.
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9d" ]]; then

TF_A="PatchTST,iTransformer,TimesNet,TSMixer,Informer,Autoformer,FEDformer,VanillaTransformer,TiDE,NBEATSx,BiTCN,KAN"
TF_B="RMoK,SOFTS,StemGNN,DLinear,NLinear,TimeMixer,TimeXer,TSMixerx,xLSTM,TimeLLM,DeepNPTS"

echo ""
echo "--- 9d: transformer_sota shard A (12 models, gpu 1×V100 14c 320GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p9_tfA_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "normal" "2-00:00:00" "320G" "14" "gpu:volta:1" \
        "$TASK" "transformer_sota" "$ABL" "$TF_A" ""
done

echo ""
echo "--- 9d: transformer_sota shard B (11 models, gpu 1×V100 14c 320GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p9_tfB_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "normal" "2-00:00:00" "320G" "14" "gpu:volta:1" \
        "$TASK" "transformer_sota" "$ABL" "$TF_B" ""
done

fi  # 9d

# ============================================================================
# 9e. FOUNDATION (11 models) — GPU
#     Chronos, ChronosBolt, Chronos2: horizon fix (was hardcoded 7)
#     Moirai, MoiraiLarge, Moirai2: entity cap removed
#     Timer, TimeMoE, MOMENT, LagLlama: horizon + entity fixes
#     TimesFM: already correct
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9e" ]]; then

echo ""
echo "--- 9e: foundation (11 models, gpu 1×V100 14c 512GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p9_fnd_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "normal" "2-00:00:00" "512G" "14" "gpu:volta:1" \
        "$TASK" "foundation" "$ABL" "" ""
done

fi  # 9e

# ============================================================================
# 9f. IRREGULAR (4 models) — GPU
#     GRU-D, SAITS (existing) + BRITS, CSDI (new)
#     All use PyPOTS imputation → tail-mean prediction.
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9f" ]]; then

echo ""
echo "--- 9f: irregular (4 models, gpu 1×V100 14c 256GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p9_irr_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "normal" "2-00:00:00" "256G" "14" "gpu:volta:1" \
        "$TASK" "irregular" "$ABL" "" ""
done

fi  # 9f

# ============================================================================
# 9g. TSLIB_SOTA (20 models) — GPU, 3 shards
#     CRITICAL FIX: per-entity prediction (was single-window broadcast)
#     All: 100 epochs, patience=15, training normalization saved
#     Shard A (7): TimeFilter, WPMixer, MultiPatchFormer, TiRex,
#                  MSGNet, PAttn, MambaSimple
#     Shard B (7): Mamba, Koopa, FreTS, Crossformer, MICN, SegRNN, ETSformer
#     Shard C (6): NonstationaryTransformer, FiLM, SCINet, LightTS,
#                  Pyraformer, Reformer
#     Increased memory to 640G (Phase 8 OOM at 320G for full ablation).
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9g" ]]; then

TSLIB_A="TimeFilter,WPMixer,MultiPatchFormer,TiRex,MSGNet,PAttn,MambaSimple"
TSLIB_B="Mamba,Koopa,FreTS,Crossformer,MICN,SegRNN,ETSformer"
TSLIB_C="NonstationaryTransformer,FiLM,SCINet,LightTS,Pyraformer,Reformer"

echo ""
echo "--- 9g: tslib_sota shard A (7 models, gpu 1×V100 14c 640GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p9_tsA_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "normal" "2-00:00:00" "640G" "14" "gpu:volta:1" \
        "$TASK" "tslib_sota" "$ABL" "$TSLIB_A" ""
done

echo ""
echo "--- 9g: tslib_sota shard B (7 models, gpu 1×V100 14c 640GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p9_tsB_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "normal" "2-00:00:00" "640G" "14" "gpu:volta:1" \
        "$TASK" "tslib_sota" "$ABL" "$TSLIB_B" ""
done

echo ""
echo "--- 9g: tslib_sota shard C (6 models, gpu 1×V100 14c 640GB 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p9_tsC_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "normal" "2-00:00:00" "640G" "14" "gpu:volta:1" \
        "$TASK" "tslib_sota" "$ABL" "$TSLIB_C" ""
done

fi  # 9g

# ============================================================================
# 9h. AUTOFIT (V734, V735, V736 only) — bigmem
#     Only the latest/best AutoFit versions. V1-V733 excluded from
#     fair comparison (per user request: old versions not meaningful).
#     Bigmem needed: AutoFit ensemble requires loading all base models.
# ============================================================================
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9h" ]]; then

AF_SELECT="AutoFitV734,AutoFitV735,AutoFitV736"

echo ""
echo "--- 9h: autofit V734/V735/V736 (3 models, bigmem 28c 1T 2d) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p9_aft_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "bigmem" "normal" "2-00:00:00" "1T" "28" "" \
        "$TASK" "autofit" "$ABL" "$AF_SELECT" ""
done

fi  # 9h

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
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9a" ]]; then
echo "  9a statistical:       11 jobs (batch 28c 112GB 2d) [15 models]"
fi
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9b" ]]; then
echo "  9b ml_tabular:        11 jobs (batch 28c 112GB 2d) [15 models × ALL horizons]"
fi
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9c" ]]; then
echo "  9c deep_classical:    11 jobs (gpu 1×V100 14c 256GB 2d) [9 models]"
fi
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9d" ]]; then
echo "  9d transformer_sota:  22 jobs (gpu 1×V100 14c 320GB 2d) [2 shards: A(12)+B(11)]"
fi
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9e" ]]; then
echo "  9e foundation:        11 jobs (gpu 1×V100 14c 512GB 2d) [11 models, horizon fix]"
fi
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9f" ]]; then
echo "  9f irregular:         11 jobs (gpu 1×V100 14c 256GB 2d) [4 models: +BRITS,CSDI]"
fi
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9g" ]]; then
echo "  9g tslib_sota:        33 jobs (gpu 1×V100 14c 640GB 2d) [3 shards: A(7)+B(7)+C(6)]"
fi
if [[ "$SHARD_FILTER" == "all" || "$SHARD_FILTER" == "9h" ]]; then
echo "  9h autofit:           11 jobs (bigmem 28c 1T 2d) [V734,V735,V736 only]"
fi
echo "  ─────────────────────────────────────────────"
echo "  TOTAL (all): 121 jobs (88 GPU + 22 batch + 11 bigmem)"
echo ""
echo "  Recommended split (MaxJobsPU=100):"
echo "    npin:   9a,9b,9c,9d,9e,9f = 77 jobs (66 GPU + 11 batch)"
echo "    cfisch: 9g,9h             = 44 jobs (33 GPU + 11 bigmem)"
echo ""
echo "Output: ${OUTPUT_BASE}/"
echo "Logs:   ${LOG_DIR}/"
echo ""
echo "Monitor: squeue -u \$(whoami) -o '%.10i %.12j %.8T %.10M %.14R'"
echo "Results: python3 scripts/aggregate_block3_results.py --input-dir ${OUTPUT_BASE}"
echo "================================================================"
