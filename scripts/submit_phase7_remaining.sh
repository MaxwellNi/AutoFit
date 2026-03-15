#!/usr/bin/env bash
# DEPRECATED LEGACY SURFACE
# This script belongs to the retired Phase 7 / V72 / early V73 execution line.
# It is preserved for historical traceability only.
# Do not use it for current Phase 9 / AutoFitV739 submissions or status decisions.
# Current truth docs: docs/CURRENT_SOURCE_OF_TRUTH.md and docs/PHASE9_V739_FACT_ALIGNMENT.md

# ============================================================================
# Phase 7 Remaining Experiments — SLURM Submission Script
# KDD'26 Block 3: V73 GPU + V72 GPU + Transformer SOTA shard C
# ============================================================================
#
# Rationale:
#   - 77/77 main phase7 shards complete (all baseline categories done)
#   - 104/104 condition keys materialized with 3681 strict records
#   - V72 ran on CPU-only batch nodes → 0/104 rank-1 wins (GPU gate issue)
#   - 3 new SOTA models (xLSTM, TimeLLM, DeepNPTS) added but never benchmarked
#   - V73 newly implemented with RL + multi-agent coordination, needs GPU run
#
# What this script submits:
#   A. Transformer SOTA shard C (xLSTM, TimeLLM, DeepNPTS):   11 GPU jobs
#   B. AutoFit V72 on GPU (fix root cause — GPU gate):        11 GPU jobs
#   C. AutoFit V73 on GPU (new RL + multi-agent):             11 GPU jobs
#                                                        ──────────────────
#   Total:                                                     33 GPU jobs
#
# Cluster: ULHPC Iris HPC
# ============================================================================
#
# Resource allocation strategy:
#
#   ┌─────────────────────────────────────────────────────────────────────────┐
#   │ Partition: gpu (24 nodes × 28c × 756GB × 4×V100-32GB)                 │
#   │ QOS strategy: DUAL QOS for max parallelism + safety                   │
#   │                                                                       │
#   │ 1) `normal` QOS on GPU partition                                      │
#   │    - MaxJobsPU: 100 concurrent jobs (vs 4 for iris-gpu-long)          │
#   │    - MaxWall:   2 days (partition limit)                              │
#   │    - No GrpNodes limit (vs 2 for iris-gpu-long)                       │
#   │    → Use for: tsota_C (est 1-3h), V72 (est 12-24h)                   │
#   │                                                                       │
#   │ 2) `iris-gpu-long` QOS                                                │
#   │    - MaxJobsPU: 4, MaxNodes: 2 concurrent                             │
#   │    - MaxWall:   14 days                                               │
#   │    → Use for: V73 task1 (heaviest, est 30-42h, tight for 2d limit)   │
#   │                                                                       │
#   │ 3) `normal` QOS for remaining V73                                     │
#   │    → Use for: V73 task2+task3 (est 18-28h, comfortably within 2d)    │
#   └─────────────────────────────────────────────────────────────────────────┘
#
# Memory budget:
#   GPU node: 756GB, MaxMemPerCPU=27000MB
#   14 CPUs × 27000MB = 378GB per job → safe for panel data + model weights
#   Fits 2 jobs per node (378GB × 2 = 756GB ≤ 756GB node capacity)
#
# Historical runtime reference (completed GPU autofit jobs):
#   afv3_e_on (V3 extreme, task1):  1d 16h 48m (40.8h)  — max observed
#   afv3_t_on (V3, task2/3):        23h 50m              — typical
#   p7_tsA_t1_fu (10 transform):    11h 07m              — shard A max
#   p7_tsB_t1_fu (10 transform):     4h 48m              — shard B max
#   p7_dc_t1_fu  (4 deep):           1h 15m              — deep max
#
# V73 estimate:
#   Timeout per candidate: 900s, 39 candidates × 4 CV folds
#   Theoretical max sequential: 39h, practical: ~25-30h (many fail fast)
#   task1 (3 targets × 4 horizons): heaviest → up to 42h
#   task2/task3: ~18-28h
#
# Account: christian.fisch (sacctmgr-verified association for npin on iris)
#
# Usage:
#   bash scripts/submit_phase7_remaining.sh
#   bash scripts/submit_phase7_remaining.sh --dry-run
#   bash scripts/submit_phase7_remaining.sh --v73-only
#   bash scripts/submit_phase7_remaining.sh --cancel-stale
# ============================================================================

set -euo pipefail

REPO="/home/users/npin/repo_root"
STAMP="20260203_225620"
OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7"
LOG_DIR="/work/projects/eint/logs/phase7_remaining"
SLURM_DIR="${REPO}/.slurm_scripts/phase7_remaining"
ACCOUNT="christian.fisch"
PRESET="full"
SEED=42
DRY_RUN=false
V73_ONLY=false
CANCEL_STALE=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)        DRY_RUN=true ;;
        --v73-only)       V73_ONLY=true ;;
        --cancel-stale)   CANCEL_STALE=true ;;
    esac
done

# ============================================================================
# Environment activation
# ============================================================================
activate_insider_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "insider" ]]; then
        return 0
    fi
    if command -v micromamba >/dev/null 2>&1; then
        local roots=()
        if [[ -n "${MAMBA_ROOT_PREFIX:-}" ]]; then
            roots+=("${MAMBA_ROOT_PREFIX}")
        fi
        roots+=(
            "/mnt/aiongpfs/projects/eint/envs/.micromamba"
            "${HOME}/.local/share/micromamba"
            "${HOME}/micromamba"
        )
        local r
        for r in "${roots[@]}"; do
            [[ -d "${r}" ]] || continue
            export MAMBA_ROOT_PREFIX="${r}"
            eval "$(micromamba shell hook -s bash)"
            if micromamba activate insider; then
                return 0
            fi
        done
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
    raise SystemExit(
        f"FATAL: insider python must be >=3.11, got {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
PY
python3 scripts/assert_block3_execution_contract.py \
    --entrypoint "scripts/submit_phase7_remaining.sh"

mkdir -p "$LOG_DIR" "$SLURM_DIR"

# ============================================================================
# Cancel stale pending jobs if requested
# ============================================================================
if $CANCEL_STALE; then
    echo "=== Scanning for stale pending p7xF_fdr_* jobs ==="
    STALE_JOBS=$(squeue -u "$USER" -h -o "%i %j" | grep "p7xF_fdr" | awk '{print $1}' || true)
    if [[ -n "$STALE_JOBS" ]]; then
        echo "  Cancelling stale foundation rerun jobs: ${STALE_JOBS}"
        if ! $DRY_RUN; then
            for jid in $STALE_JOBS; do
                scancel "$jid" && echo "    cancelled $jid" || echo "    failed to cancel $jid"
            done
        else
            echo "  [DRY] Would cancel: ${STALE_JOBS}"
        fi
    else
        echo "  No stale p7xF_fdr_ jobs found."
    fi
    echo ""
fi

# ============================================================================
# Micromamba activation preamble (shared by all SLURM jobs)
# ============================================================================
read -r -d '' ENV_PREAMBLE << 'ENVBLOCK' || true
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root

INSIDER_PY="${CONDA_PREFIX}/bin/python3"
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing or non-executable: ${INSIDER_PY}"
  exit 2
fi

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(which python3)"
echo "PythonV: $(python3 -V)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA:   $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"
echo "============================================================"
python3 - <<'PY'
import sys, torch
print("sys.executable:", sys.executable)
if "insider" not in sys.executable:
    raise SystemExit("FATAL: runtime python is not insider")
if sys.version_info < (3, 11):
    raise SystemExit(f"FATAL: python >={3}.{11} required, got {sys.version_info}")
if not torch.cuda.is_available():
    raise SystemExit("FATAL: GPU required but torch.cuda.is_available()=False")
PY
${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"
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
    local GRES="$7"
    local TASK="$8"
    local CATEGORY="$9"
    local ABLATION="${10}"
    local MODELS="${11}"

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
        echo "  [SUB] ${JOB_NAME} → JobID ${JID} (${PARTITION}/${QOS} ${TIME} ${MEM} ${CPUS}c ${GRES})"
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

# Separate task1 (heavy) from task2+task3 (lighter) for V73
TASK1_ABLATION_LIST=()
for abl in "${TASK1_ABLATIONS[@]}"; do
    TASK1_ABLATION_LIST+=("task1_outcome:${abl}")
done
TASK23_ABLATION_LIST=()
for abl in "${TASK2_ABLATIONS[@]}"; do
    TASK23_ABLATION_LIST+=("task2_forecast:${abl}")
done
for abl in "${TASK3_ABLATIONS[@]}"; do
    TASK23_ABLATION_LIST+=("task3_risk_adjust:${abl}")
done

echo ""
echo "================================================================"
echo "Phase 7 Remaining Experiments Submission"
echo "  Output base: ${OUTPUT_BASE}"
echo "  Account: ${ACCOUNT}"
echo "  Preset: ${PRESET} | Seed: ${SEED}"
echo "  Task-ablation combos: ${#ALL_TASKS_ABLATIONS[@]}"
echo "================================================================"

# ============================================================================
# A. TRANSFORMER SOTA SHARD C: xLSTM, TimeLLM, DeepNPTS
#    3 new models added in Phase 7.5 (ICML'24, ICLR'24, NeurIPS'23)
#    QOS: normal (est runtime: 1-4h per shard, well within 2d)
#    Resources: 1×V100, 14c, 378G — generous for only 3 models
# ============================================================================
if ! $V73_ONLY; then
    TSOTA_C="xLSTM,TimeLLM,DeepNPTS"

    echo ""
    echo "--- [A] transformer_sota shard C: ${TSOTA_C} (gpu normal, 2d) ---"
    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job \
            "p7r_tsC_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "normal" "2-00:00:00" "378G" "14" "gpu:volta:1" \
            "$TASK" "transformer_sota" "$ABL" "$TSOTA_C"
    done
fi

# ============================================================================
# B. AUTOFIT V72 on GPU (re-run with GPU access)
#    Root cause fix: V72 previously ran on batch (CPU-only) → GPU gate dropped
#    all deep/transformer/foundation candidates. Now runs on GPU partition.
#    QOS: normal (est runtime: 12-24h per shard)
#    Resources: 1×V100, 14c, 378G
# ============================================================================
if ! $V73_ONLY; then
    echo ""
    echo "--- [B] autofit V72 GPU re-run (gpu normal, 2d) ---"
    for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
        IFS=':' read -r TASK ABL <<< "$ta"
        submit_job \
            "p7r_v72_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
            "gpu" "normal" "2-00:00:00" "378G" "14" "gpu:volta:1" \
            "$TASK" "autofit" "$ABL" "AutoFitV72"
    done
fi

# ============================================================================
# C. AUTOFIT V73 on GPU (new: RL policy + multi-agent coordination)
#
#    V73 = factored contextual bandit (Thompson Sampling) + 4-agent blackboard
#    protocol (Recon/Scout/Composer/Critic). 39 candidate models including all
#    GPU models (NBEATS, PatchTST, NHITS → 86/104 individual champions).
#
#    QOS strategy (dual QOS for max throughput + safety):
#      task1 (4 jobs, heaviest): iris-gpu-long (14d wall, 2 concurrent nodes)
#        → Guarantees no timeout even for worst-case 42h runtime
#      task2+task3 (7 jobs, lighter): normal (2d wall, 100 concurrent)
#        → Max parallelism for jobs completing in 18-28h
#
#    Resources: 1×V100, 14c, 378G
# ============================================================================
echo ""
echo "--- [C] autofit V73 GPU (task1: iris-gpu-long 7d; task2+3: normal 2d) ---"

# C.1: V73 task1 on iris-gpu-long (guaranteed no timeout)
for ta in "${TASK1_ABLATION_LIST[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7r_v73_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "7-00:00:00" "378G" "14" "gpu:volta:1" \
        "$TASK" "autofit" "$ABL" "AutoFitV73"
done

# C.2: V73 task2+task3 on normal QOS (max parallelism)
for ta in "${TASK23_ABLATION_LIST[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7r_v73_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "normal" "2-00:00:00" "378G" "14" "gpu:volta:1" \
        "$TASK" "autofit" "$ABL" "AutoFitV73"
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
if ! $V73_ONLY; then
    echo "  [A] transformer_sota shard C:  11 jobs (gpu/normal   1×V100 14c 378G 2d)"
    echo "      Models: xLSTM, TimeLLM, DeepNPTS"
    echo "  [B] autofit V72 GPU:           11 jobs (gpu/normal   1×V100 14c 378G 2d)"
    echo "      Root cause fix: GPU-enabled candidate pool"
fi
echo "  [C] autofit V73 GPU:"
echo "      task1 (heavy):              4 jobs (gpu/long     1×V100 14c 378G 7d)"
echo "      task2+task3 (lighter):      7 jobs (gpu/normal   1×V100 14c 378G 2d)"
echo "      RL policy + multi-agent coordination, 39 candidates"
echo ""
echo "Concurrent job limits:"
echo "  normal QOS:        100 jobs/user (no node limit)"
echo "  iris-gpu-long QOS:   4 jobs/user (2 nodes max simultaneous)"
echo "  → All normal jobs start quickly; long jobs drain 2-at-a-time"
echo ""
echo "Expected timeline:"
echo "  tsota_C:  complete in ~4h"
echo "  V72:      complete in ~24h"
echo "  V73 t2/3: complete in ~28h"
echo "  V73 t1:   complete in ~42h (2 concurrent → 2 rounds × ~42h = ~3.5d)"
echo ""
echo "  Estimated total wall time: ~3.5 days for all 33 experiments"
echo ""
echo "Output: ${OUTPUT_BASE}/"
echo "Logs:   ${LOG_DIR}/"
echo ""
echo "Monitor:"
echo "  squeue -u npin -o '%.10i %.9P %.15j %.8T %.10M %.10L %.4C %.7m %R'"
echo "  watch -n 60 'squeue -u npin | wc -l'"
echo ""
echo "Results:"
echo "  python3 scripts/consolidate_block3_results.py"
echo "  python3 scripts/aggregate_block3_results.py --input-dir ${OUTPUT_BASE}"
echo "================================================================"
