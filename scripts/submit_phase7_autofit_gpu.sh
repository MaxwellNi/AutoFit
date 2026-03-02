#!/usr/bin/env bash
# ============================================================================
# Phase 7 — AutoFit V72 + V73 GPU Resubmission
# ============================================================================
#
# ROOT CAUSE FIX:
#   V72 ran on batch (CPU-only) nodes, so _fit_single_candidate's GPU gate
#   silently dropped ALL deep_classical, transformer_sota, and foundation
#   models. V72 trained only on ~9 tabular models, while all 104 champions
#   are GPU models (NBEATS 39/104, PatchTST 24/104, NHITS 23/104).
#
#   This script runs AutoFit V72 + V73 on GPU nodes so that ALL 39
#   candidate models are available in the ensemble selection pool.
#
# Requirements: 1×V100, 14 cores, 320GB RAM (matching transformer_sota)
# Expected wall time: 6-18h per condition (36 candidates × 3×4 temporal CV)
#
# Usage:
#   bash scripts/submit_phase7_autofit_gpu.sh
#   bash scripts/submit_phase7_autofit_gpu.sh --dry-run
#   bash scripts/submit_phase7_autofit_gpu.sh --v73-only
# ============================================================================

set -euo pipefail

REPO="/home/users/npin/repo_root"
STAMP="20260203_225620"
OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7"
LOG_DIR="/work/projects/eint/logs/phase7_autofit_gpu"
SLURM_DIR="${REPO}/.slurm_scripts/phase7_autofit_gpu"
ACCOUNT="yves.letraon"
PRESET="full"
SEED=42
DRY_RUN=false
V73_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=true ;;
        --v73-only)   V73_ONLY=true ;;
    esac
done

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
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
            source "${conda_base}/etc/profile.d/conda.sh"
            if conda activate insider; then
                return 0
            fi
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
python3 scripts/assert_block3_execution_contract.py \
    --entrypoint "scripts/submit_phase7_autofit_gpu.sh"

mkdir -p "$LOG_DIR" "$SLURM_DIR"

# Micromamba activation preamble
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
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
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
    raise SystemExit("FATAL: GPU required for AutoFit V72/V73 but torch.cuda.is_available()=False")
PY
${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"
ENVBLOCK

TOTAL_SUBMITTED=0

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

echo ""
echo "================================================================"
echo "Phase 7 AutoFit GPU Resubmission"
echo "  Output base: ${OUTPUT_BASE}"
echo "  Preset: ${PRESET} | Seed: ${SEED}"
echo "  Task-ablation combos: ${#ALL_TASKS_ABLATIONS[@]}"
echo "================================================================"

# ============================================================================
# AutoFit V72 + V73 on GPU
#
# CRITICAL: GPU nodes enable all 39 candidate models including:
#   deep_classical:    NBEATS, NHITS, TFT, DeepAR
#   transformer_sota:  PatchTST, DLinear, NLinear, TimeMixer, TimeXer, KAN,
#                      xLSTM, TimeLLM, DeepNPTS*
#   foundation:        Moirai, Chronos, Timer, MOMENT, etc.
#
# * KAN is registered under transformer_sota in the model registry.
#
# Resources: 1×V100 32GB, 14 cores, 320GB RAM
# Wall time: 3 days (36 candidates × 3×4 temporal CV = heavy)
# ============================================================================

if $V73_ONLY; then
    AF_MODELS="AutoFitV73"
    echo ""
    echo "--- autofit V73 only (gpu 1×V100 14c 320GB, 3d) ---"
else
    AF_MODELS="AutoFitV72,AutoFitV73"
    echo ""
    echo "--- autofit V72+V73 (gpu 1×V100 14c 320GB, 3d) ---"
fi

for ta in "${ALL_TASKS_ABLATIONS[@]}"; do
    IFS=':' read -r TASK ABL <<< "$ta"
    submit_job \
        "p7g_af_$(task_abbrev $TASK)_$(abl_abbrev $ABL)" \
        "gpu" "iris-gpu-long" "3-00:00:00" "320G" "14" "gpu:volta:1" \
        "$TASK" "autofit" "$ABL" "$AF_MODELS"
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
echo "  autofit V72+V73: ${#ALL_TASKS_ABLATIONS[@]} jobs (gpu 1×V100 14c 320GB, 3d)"
echo ""
echo "  KEY CHANGE: GPU partition enables ALL 36 candidate models"
echo "  Previously: batch (CPU-only) → only ~9 tabular models survived"
echo "  Now: gpu (1×V100) → all deep/transformer/foundation models available"
echo ""
echo "  Expected: V73 ensemble should include NBEATS, PatchTST, NHITS"
echo "  as primary candidates (these win 86/104 conditions individually)"
echo ""
echo "Output: ${OUTPUT_BASE}/"
echo "Logs:   ${LOG_DIR}/"
echo ""
echo "Monitor: squeue -u npin"
echo "Results: python3 scripts/aggregate_block3_results.py --input-dir ${OUTPUT_BASE}"
echo "================================================================"
