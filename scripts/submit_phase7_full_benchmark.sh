#!/usr/bin/env bash
# ============================================================================
# Phase 7 Full Benchmark — SLURM Submission Script
# KDD'26 Block 3, All 67 models × 3 tasks × 4 ablations
# ============================================================================
#
# Cluster: ULHPC Iris HPC
#   GPU:    24 nodes, 28c, 756GB RAM, 4×V100-32GB each
#   Hopper: 1 node, 112c, 2TB RAM,   4×H100-80GB
#   Batch:  168 nodes, 28c, 112GB RAM
#   BigMem: 4 nodes, 112c, 3TB RAM
#
# Model-to-compute mapping (rationale in comments):
#   ml_tabular (15):     batch — CPU-only sklearn/LightGBM, needs RAM for 5.7M rows
#   statistical (5):     batch — CPU-only StatsForecast, modest compute
#   deep_classical (4):  gpu — NeuralForecast, moderate VRAM
#   transformer_sota (20): gpu — NeuralForecast, split into 2-3 shards for parallelism
#   foundation (11):     gpu — Chronos/Moirai/HF models, split by family
#   irregular (2):       gpu — PyPOTS GRU-D/SAITS, light GPU
#   autofit (11):        batch — mostly tree-based CV, heavy CPU
#
# Tasks × ablations:
#   task1_outcome:       [core_only, core_text, core_edgar, full]  = 4
#   task2_forecast:      [core_only, core_text, core_edgar, full]  = 4
#   task3_risk_adjust:   [core_only, core_edgar, full]             = 3
#   Total = 11 task-ablation combos × 7 categories
#
# Usage:
#   bash scripts/submit_phase7_full_benchmark.sh
#   bash scripts/submit_phase7_full_benchmark.sh --dry-run   # preview only
#   bash scripts/submit_phase7_full_benchmark.sh --pilot     # stage-A pilot
#   ALLOW_UNSAFE_SKIP_PREFLIGHT=1 bash scripts/submit_phase7_full_benchmark.sh --skip-preflight
# ============================================================================

set -euo pipefail

REPO="/home/users/npin/repo_root"
STAMP="20260203_225620"
OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7"
LOG_DIR="/work/projects/eint/logs/phase7"
SLURM_DIR="${REPO}/.slurm_scripts/phase7"
ACCOUNT="yves.letraon"
PRESET="full"
SEED=42
DRY_RUN=false
PILOT=false
SKIP_PREFLIGHT=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        --pilot)
            PILOT=true
            ;;
        --skip-preflight)
            SKIP_PREFLIGHT=true
            ;;
    esac
done

if $SKIP_PREFLIGHT && [[ "${ALLOW_UNSAFE_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Refusing --skip-preflight without explicit override."
    echo "If you must bypass the gate, rerun with:"
    echo "  ALLOW_UNSAFE_SKIP_PREFLIGHT=1 bash scripts/submit_phase7_full_benchmark.sh --skip-preflight"
    exit 2
fi

if $DRY_RUN; then
    echo "=== DRY RUN MODE — no jobs will be submitted ==="
fi
if $PILOT; then
    echo "=== PILOT MODE — submitting Stage-A subset only ==="
fi

if ! $DRY_RUN && ! $SKIP_PREFLIGHT; then
    # Full benchmark includes AutoFitV71 in autofit shard 2; gate before submit.
    echo "=== Running mandatory V7.1 preflight gate (g01 default) ==="
    bash scripts/preflight_block3_v71_gate.sh --v71-variant=g01
fi

mkdir -p "$LOG_DIR" "$SLURM_DIR"

# Micromamba activation preamble (shared by all jobs)
read -r -d '' ENV_PREAMBLE << 'ENVBLOCK' || true
# Activate micromamba environment
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(which python3)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"
ENVBLOCK

TOTAL_SUBMITTED=0

# ============================================================================
# Abbreviation helpers
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

# ============================================================================
# Helper: generate and submit a SLURM job
# ============================================================================
submit_job() {
    local JOB_NAME="$1"
    local PARTITION="$2"
    local QOS="$3"
    local TIME="$4"
    local MEM="$5"
    local CPUS="$6"
    local GRES="$7"       # empty for batch
    local TASK="$8"
    local CATEGORY="$9"
    local ABLATION="${10}"
    local MODELS="${11}"   # comma-separated or empty=all
    local EXTRA="${12:-}"  # extra sbatch options

    local OUTDIR="${OUTPUT_BASE}/${TASK}/${CATEGORY}/${ABLATION}"
    local SCRIPT="${SLURM_DIR}/${JOB_NAME}.sh"

    # Build GRES line
    local GRES_LINE=""
    if [[ -n "$GRES" ]]; then
        GRES_LINE="#SBATCH --gres=${GRES}"
    fi

    # Build models argument
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

python3 scripts/run_block3_benchmark_shard.py \\
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
# Task-ablation matrix
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
echo "Phase 7 Full Benchmark Submission"
echo "  Output base: ${OUTPUT_BASE}"
echo "  Preset: ${PRESET} | Seed: ${SEED}"
echo "  Task-ablation combos: ${#ALL_TASKS_ABLATIONS[@]}"
echo "================================================================"

if $PILOT; then
    ML_REF="RandomForest,ExtraTrees,LightGBM,XGBoost"
    DC_REF="NBEATS,NHITS"
    TS_REF="PatchTST,iTransformer,TimesNet"
    AF_PILOT="AutoFitV6,AutoFitV7,AutoFitV71"

    echo ""
    echo "--- PILOT ml_tabular refs (batch 28c 112GB) ---"
    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job \
            "p7p_mlt_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" \
            "$TASK" "ml_tabular" "$ABL" "$ML_REF" ""
    done

    echo ""
    echo "--- PILOT deep_classical refs (gpu 1×V100 14c 256GB) ---"
    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job \
            "p7p_dc_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "iris-gpu-long" "2-00:00:00" "256G" "14" "gpu:volta:1" \
            "$TASK" "deep_classical" "$ABL" "$DC_REF" ""
    done

    echo ""
    echo "--- PILOT transformer refs (gpu 1×V100 14c 320GB) ---"
    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job \
            "p7p_ts_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "iris-gpu-long" "2-00:00:00" "320G" "14" "gpu:volta:1" \
            "$TASK" "transformer_sota" "$ABL" "$TS_REF" ""
    done

    echo ""
    echo "--- PILOT autofit (V6/V7/V71, batch 28c 112GB) ---"
    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job \
            "p7p_af_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" \
            "$TASK" "autofit" "$ABL" "$AF_PILOT" ""
    done

    echo ""
    echo "================================================================"
    if $DRY_RUN; then
        echo "PILOT DRY RUN COMPLETE — no jobs submitted"
    else
        echo "PILOT jobs submitted: ${TOTAL_SUBMITTED}"
    fi
    echo "Pilot categories: ml_tabular refs + deep refs + transformer refs + autofit V6/V7/V71"
    echo "Output: ${OUTPUT_BASE}/"
    echo "Logs:   ${LOG_DIR}/"
    echo "================================================================"
    exit 0
fi

# ============================================================================
# 1. ML TABULAR (15 models) — batch nodes, CPU-only
#    5.7M rows × 139 cols → needs RAM. 28 cores for parallel tree building.
#    CatBoost/XGBoost/LightGBM use all cores. ~6-12h per shard.
# ============================================================================
echo ""
echo "--- ml_tabular (15 models, batch 28c 112GB) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7_mlt_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" \
        "$TASK" "ml_tabular" "$ABL" "" ""
done

# ============================================================================
# 2. STATISTICAL (5 models) — batch nodes, CPU-only
#    StatsForecast runs per-entity. Light compute but many entities.
#    All entities now used (no cap). ~2-4h per shard.
# ============================================================================
echo ""
echo "--- statistical (5 models, batch 14c 64GB) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7_sta_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "batch" "iris-batch-long" "2-00:00:00" "64G" "14" "" \
        "$TASK" "statistical" "$ABL" "" ""
done

# ============================================================================
# 3. DEEP CLASSICAL (4 models: NBEATS, NHITS, TFT, DeepAR)
#    NeuralForecast GPU training. 1×V100 sufficient.
#    Now training on ALL entities (no cap for channel-independent).
#    max_steps=1000-2000, ~4-8h per shard. Need 256GB for panel data.
# ============================================================================
echo ""
echo "--- deep_classical (4 models, gpu 1×V100 14c 256GB) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7_dc_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "2-00:00:00" "256G" "14" "gpu:volta:1" \
        "$TASK" "deep_classical" "$ABL" "" ""
done

# ============================================================================
# 4. TRANSFORMER SOTA (20 models) — GPU nodes
#    Split into 2 shards of 10 for parallelism:
#      Shard A: PatchTST,iTransformer,TimesNet,TSMixer,Informer,
#               Autoformer,FEDformer,VanillaTransformer,TiDE,NBEATSx
#      Shard B: BiTCN,KAN,RMoK,SOFTS,StemGNN,DLinear,NLinear,
#               TimeMixer,TimeXer,TSMixerx
#    Cross-series models (iTransformer,TSMixer,RMoK,SOFTS,StemGNN,
#    TimeMixer,TimeXer,TSMixerx) need n_series → 5000 entities max.
#    1×V100, 14c, 256GB each. ~12-24h per shard.
# ============================================================================
TSOTA_A="PatchTST,iTransformer,TimesNet,TSMixer,Informer,Autoformer,FEDformer,VanillaTransformer,TiDE,NBEATSx"
TSOTA_B="BiTCN,KAN,RMoK,SOFTS,StemGNN,DLinear,NLinear,TimeMixer,TimeXer,TSMixerx"

echo ""
echo "--- transformer_sota shard A (10 models, gpu 1×V100 14c 320GB) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7_tsA_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "2-00:00:00" "320G" "14" "gpu:volta:1" \
        "$TASK" "transformer_sota" "$ABL" "$TSOTA_A" ""
done

echo ""
echo "--- transformer_sota shard B (10 models, gpu 1×V100 14c 320GB) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7_tsB_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "2-00:00:00" "320G" "14" "gpu:volta:1" \
        "$TASK" "transformer_sota" "$ABL" "$TSOTA_B" ""
done

# ============================================================================
# 5. FOUNDATION (11 models) — GPU nodes
#    Split into 3 shards by family:
#      Shard C: Chronos,ChronosBolt,Chronos2          (3 models, ~4h)
#      Shard M: Moirai,MoiraiLarge,Moirai2            (3 models, ~6h)
#      Shard H: Timer,TimeMoE,MOMENT,LagLlama,TimesFM (5 models, ~8h)
#    ALL entities used (no cap). Need VRAM for model weights.
#    1×V100 each, 14c, 256GB. TimesFM uses Docker container.
# ============================================================================
FM_CHRONOS="Chronos,ChronosBolt,Chronos2"
FM_MOIRAI="Moirai,MoiraiLarge,Moirai2"
FM_HF="Timer,TimeMoE,MOMENT,LagLlama,TimesFM"

echo ""
echo "--- foundation shard Chronos (3 models, gpu 1×V100 14c 256GB) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7_fmC_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "2-00:00:00" "256G" "14" "gpu:volta:1" \
        "$TASK" "foundation" "$ABL" "$FM_CHRONOS" ""
done

echo ""
echo "--- foundation shard Moirai (3 models, gpu 1×V100 14c 256GB) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7_fmM_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "2-00:00:00" "256G" "14" "gpu:volta:1" \
        "$TASK" "foundation" "$ABL" "$FM_MOIRAI" ""
done

echo ""
echo "--- foundation shard HF (5 models, gpu 1×V100 14c 256GB) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7_fmH_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "2-00:00:00" "256G" "14" "gpu:volta:1" \
        "$TASK" "foundation" "$ABL" "$FM_HF" ""
done

# ============================================================================
# 6. IRREGULAR (2 models: GRU-D, SAITS) — GPU nodes
#    PyPOTS with CUDA. Small models, 1×V100 is generous.
#    max_entities=5000. ~2-4h per shard.
# ============================================================================
echo ""
echo "--- irregular (2 models, gpu 1×V100 7c 128GB) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7_irr_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "1-00:00:00" "128G" "7" "gpu:volta:1" \
        "$TASK" "irregular" "$ABL" "" ""
done

# ============================================================================
# 7. AUTOFIT (11 models: V1-V7 + V2E, V3E, V3Max, V71)
#    AutoFit wraps tree models (CPU) but may invoke GPU candidates if CUDA
#    available. Split into 2 shards for parallelism:
#      Shard 1: AutoFitV1,AutoFitV2,AutoFitV2E,AutoFitV3,AutoFitV3E (simpler)
#      Shard 2: AutoFitV3Max,AutoFitV4,AutoFitV5,AutoFitV6,AutoFitV7,AutoFitV71
#    Use batch nodes (28c 112GB). GPU not needed: _fit_single_candidate now
#    skips GPU models when torch.cuda.is_available() is False.
#    ~8-16h per shard due to temporal CV × multiple candidates.
# ============================================================================
AF_SIMPLE="AutoFitV1,AutoFitV2,AutoFitV2E,AutoFitV3,AutoFitV3E"
AF_HEAVY="AutoFitV3Max,AutoFitV4,AutoFitV5,AutoFitV6,AutoFitV7,AutoFitV71"

echo ""
echo "--- autofit shard 1 (V1-V3E, batch 28c 112GB) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7_af1_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" \
        "$TASK" "autofit" "$ABL" "$AF_SIMPLE" ""
done

echo ""
echo "--- autofit shard 2 (V4-V7, batch 28c 112GB) ---"
for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7_af2_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "batch" "iris-batch-long" "2-00:00:00" "112G" "28" "" \
        "$TASK" "autofit" "$ABL" "$AF_HEAVY" ""
done

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
echo "Job breakdown:"
echo "  ml_tabular:       11 jobs (batch 28c 112GB, 2d)"
echo "  statistical:      11 jobs (batch 14c 64GB, 2d)"
echo "  deep_classical:   11 jobs (gpu 1×V100 14c 256GB, 2d)"
echo "  transformer_sota: 22 jobs (gpu 1×V100 14c 320GB, 2d) [2 shards]"
echo "  foundation:       33 jobs (gpu 1×V100 14c 256GB, 2d) [3 shards]"
echo "  irregular:        11 jobs (gpu 1×V100 7c 128GB, 1d)"
echo "  autofit:          22 jobs (batch 28c 112GB, 2d) [2 shards]"
echo "  ─────────────────────────────────────────────"
echo "  TOTAL:           121 jobs"
echo ""
echo "Output: ${OUTPUT_BASE}/"
echo "Logs:   ${LOG_DIR}/"
echo ""
echo "Monitor: squeue -u npin"
echo "Results: python3 scripts/aggregate_block3_results.py --input-dir ${OUTPUT_BASE}"
echo "================================================================"
