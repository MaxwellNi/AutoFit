#!/usr/bin/env bash
# ============================================================================
# Block 3 COMPLETE Benchmark — ALL models, ALL tasks, fair conditions
# ============================================================================
# Launches ONLY the missing/incomplete shards:
#   - ml_tabular:  15 models × 3 tasks × 2 ablations  (MISSING)
#   - autofit:      3 models × 3 tasks × 2 ablations  (RE-RUN: redesigned)
#   - Other categories: re-run if status != "completed"
#
# Already completed (32 models, 48 shards):
#   - statistical (5), deep_classical (4), foundation (2), irregular (2)
#   - transformer_sota_A (5), _B (5), _C (5)
#
# Fair conditions:
#   - Same seed=42, same temporal split, same node class (V100)
#   - Same preset=full (horizons [1,7,14,30], bootstrap=1000)
#   - Same 128GB memory, 7 CPUs, 1 GPU per job
#
# Usage: bash scripts/launch_full_benchmark.sh
# ============================================================================
set -euo pipefail

STAMP="20260203_225620"
BASE="runs/benchmarks/block3_${STAMP}_iris_full"
SCRIPT="scripts/run_block3_benchmark_shard.py"
ACCOUNT="yves.letraon"
PARTITION="gpu"
QOS="normal"
TIME="24:00:00"
MEM="128G"
CPUS=7
GPUS=1
SEED=42
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================================"
echo "Block 3 Full Benchmark Launch"
echo "Working dir: ${WORKDIR}"
echo "Output base: ${BASE}"
echo "============================================================"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_edgar")
SUBMITTED=0
SKIPPED=0

# ============================================================================
# Helper: submit one shard
# ============================================================================
submit_shard() {
    local TASK="$1"
    local HARNESS_CAT="$2"    # category flag for --category
    local DIR_CAT="$3"        # directory name (may differ, e.g. transformer_sota_A)
    local ABLATION="$4"
    local JOB_NAME="$5"
    local MODELS="${6:-}"     # optional --models override

    local OUTDIR="${BASE}/${TASK}/${DIR_CAT}/${ABLATION}"
    mkdir -p "${OUTDIR}"

    # Skip if already completed
    if [[ -f "${OUTDIR}/MANIFEST.json" ]]; then
        local STATUS
        STATUS=$(python3 -c "import json; print(json.load(open('${OUTDIR}/MANIFEST.json')).get('status',''))" 2>/dev/null || echo "")
        if [[ "${STATUS}" == "completed" ]]; then
            SKIPPED=$((SKIPPED + 1))
            return 0
        fi
        echo "  RE-RUN (status=${STATUS}): ${TASK}/${DIR_CAT}/${ABLATION}"
    fi

    local MODELS_FLAG=""
    if [[ -n "${MODELS}" ]]; then
        MODELS_FLAG="--models ${MODELS}"
    fi

    echo "  SUBMIT: ${TASK}/${DIR_CAT}/${ABLATION} -> ${JOB_NAME}"

    sbatch \
        --job-name="${JOB_NAME}" \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --qos="${QOS}" \
        --time="${TIME}" \
        --mem="${MEM}" \
        --cpus-per-task="${CPUS}" \
        --gres="gpu:volta:${GPUS}" \
        --output="${OUTDIR}/slurm_%j.log" \
        --error="${OUTDIR}/slurm_%j.log" \
        --export=ALL \
        --wrap="
set -e
source \$(conda info --base 2>/dev/null || echo \${MAMBA_ROOT_PREFIX:-/home/users/npin/miniforge3})/etc/profile.d/micromamba.sh 2>/dev/null || true
micromamba activate insider 2>/dev/null || conda activate insider 2>/dev/null || true
cd ${WORKDIR}
echo 'Job \${SLURM_JOB_ID} on \$(hostname) — ${TASK}/${DIR_CAT}/${ABLATION}'
echo 'Python: \$(which python3)'
echo 'Start: \$(date -Iseconds)'
python3 ${SCRIPT} \
    --task ${TASK} \
    --category ${HARNESS_CAT} \
    --ablation ${ABLATION} \
    --preset full \
    --output-dir ${OUTDIR} \
    --seed ${SEED} \
    --no-verify-first \
    ${MODELS_FLAG}
echo 'Done: \$(date -Iseconds)'
"

    SUBMITTED=$((SUBMITTED + 1))
}

# ============================================================================
# 1. ML_TABULAR — all 15 models (3 tasks × 2 ablations = 6 jobs)
# ============================================================================
echo ""
echo "=== [1/3] ML_TABULAR (15 models) ==="
for TASK in "${TASKS[@]}"; do
    for ABL in "${ABLATIONS[@]}"; do
        submit_shard "${TASK}" "ml_tabular" "ml_tabular" "${ABL}" \
            "mlt_${TASK: -1}_${ABL:5:2}"
    done
done

# ============================================================================
# 2. AUTOFIT — redesigned v1/v2/v2e (3 tasks × 2 ablations = 6 jobs)
#    These internally fit base models so need more time.
# ============================================================================
echo ""
echo "=== [2/3] AUTOFIT (3 models: V1, V2, V2E) ==="
for TASK in "${TASKS[@]}"; do
    for ABL in "${ABLATIONS[@]}"; do
        # Remove old broken results
        local_dir="${BASE}/${TASK}/autofit/${ABL}"
        if [[ -f "${local_dir}/MANIFEST.json" ]]; then
            rm -f "${local_dir}/MANIFEST.json" "${local_dir}/metrics.json" \
                  "${local_dir}/predictions.parquet" 2>/dev/null || true
        fi
        submit_shard "${TASK}" "autofit" "autofit" "${ABL}" \
            "af_${TASK: -1}_${ABL:5:2}"
    done
done

# ============================================================================
# 3. RE-CHECK all other categories — re-run if incomplete
# ============================================================================
echo ""
echo "=== [3/3] RE-CHECK OTHER CATEGORIES ==="

# Transformer SOTA splits (with explicit --models)
TSA="PatchTST,iTransformer,TimesNet,TSMixer,Informer"
TSB="Autoformer,FEDformer,VanillaTransformer,TiDE,NBEATSx"
TSC="BiTCN,KAN,RMoK,SOFTS,StemGNN"

for TASK in "${TASKS[@]}"; do
    for ABL in "${ABLATIONS[@]}"; do
        # Statistical
        submit_shard "${TASK}" "statistical" "statistical" "${ABL}" \
            "st_${TASK: -1}_${ABL:5:2}"
        # Deep Classical
        submit_shard "${TASK}" "deep_classical" "deep_classical" "${ABL}" \
            "dc_${TASK: -1}_${ABL:5:2}"
        # Foundation
        submit_shard "${TASK}" "foundation" "foundation" "${ABL}" \
            "fn_${TASK: -1}_${ABL:5:2}"
        # Irregular
        submit_shard "${TASK}" "irregular" "irregular" "${ABL}" \
            "ir_${TASK: -1}_${ABL:5:2}"
        # Transformer SOTA A
        submit_shard "${TASK}" "transformer_sota" "transformer_sota_A" "${ABL}" \
            "tA_${TASK: -1}_${ABL:5:2}" "${TSA}"
        # Transformer SOTA B
        submit_shard "${TASK}" "transformer_sota" "transformer_sota_B" "${ABL}" \
            "tB_${TASK: -1}_${ABL:5:02}" "${TSB}"
        # Transformer SOTA C
        submit_shard "${TASK}" "transformer_sota" "transformer_sota_C" "${ABL}" \
            "tC_${TASK: -1}_${ABL:5:2}" "${TSC}"
    done
done

echo ""
echo "============================================================"
echo "SUMMARY: Submitted=${SUBMITTED}, Skipped=${SKIPPED} (already completed)"
echo "Monitor: squeue -u \$USER --sort=+i"
echo "Results: ${BASE}/"
echo "============================================================"
