#!/usr/bin/env bash
# =============================================================================
# MAXIMUM-SPEED Block 3 Benchmark Relaunch
#
# Strategy: exploit ALL available GPU nodes (24 × V100) on Iris HPC.
#
# transformer_sota (15 models): split into 3 sub-shards of 5 models each.
#   → 3 tasks × 2 ablations × 3 sub-shards = 18 GPU jobs
#   → Each sub-shard runs ~1h instead of ~4h
#
# autofit (3 models): run on GPU partition (underlying model may be GPU).
#   → 3 tasks × 2 ablations = 6 GPU jobs
#
# statistical task1 (already running, will complete soon)
# ml_tabular (already running on batch, 6 jobs)
# deep_classical task1 (already running on GPU, 2 jobs)
#
# Total new GPU jobs: 24 (within 24-node GPU partition capacity)
# =============================================================================
set -euo pipefail

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
STAMP="20260203_225620"
OUTBASE="${REPO}/runs/benchmarks/block3_${STAMP}_iris_full"
SCRIPT="${REPO}/scripts/run_block3_benchmark_shard.py"
MAMBA_EXE="/mnt/aiongpfs/users/npin/.local/bin/micromamba"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_edgar")

# Split 15 transformer_sota models into 3 groups of 5
TSPLIT_A="PatchTST,iTransformer,TimesNet,TSMixer,Informer"
TSPLIT_B="Autoformer,FEDformer,VanillaTransformer,TiDE,NBEATSx"
TSPLIT_C="BiTCN,KAN,RMoK,SOFTS,StemGNN"

GIT_HASH=$(cd "${REPO}" && git rev-parse --short HEAD)
echo "=== MAXIMUM-SPEED BENCHMARK RELAUNCH ==="
echo "Git: ${GIT_HASH}"
echo "Output: ${OUTBASE}"
echo ""

TOTAL=0

submit_gpu_shard() {
    local TASK="$1"
    local CAT="$2"        # actual --category arg (must be in CATEGORY_NAMES)
    local ABL="$3"
    local MODELS="$4"
    local JOBNAME="$5"
    local OUTDIR_SUFFIX="${6:-$CAT}"  # sub-dir name (can differ from CAT)

    local OUTDIR="${OUTBASE}/${TASK}/${OUTDIR_SUFFIX}/${ABL}"

    # Skip if already completed
    if [[ -f "${OUTDIR}/MANIFEST.json" ]]; then
        local status
        status=$(python3 -c "import json; print(json.load(open('${OUTDIR}/MANIFEST.json')).get('status','?'))" 2>/dev/null || echo "?")
        if [[ "$status" == "completed" ]]; then
            echo "SKIP ${JOBNAME} (already completed)"
            return
        fi
        # Remove old incomplete manifest so shard runs fresh
        rm -f "${OUTDIR}/MANIFEST.json" "${OUTDIR}/metrics.json" "${OUTDIR}/predictions.parquet"
    fi

    mkdir -p "${OUTDIR}"

    local MODELS_ARG=""
    if [[ -n "$MODELS" ]]; then
        MODELS_ARG="--models ${MODELS}"
    fi

    echo "SUBMIT ${JOBNAME} [${MODELS:-all}] → ${OUTDIR_SUFFIX}/${ABL}"
    sbatch \
        --job-name="${JOBNAME}" \
        --partition=gpu \
        --qos=normal \
        --account=yves.letraon \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=7 \
        --mem=180G \
        --gres=gpu:1 \
        --time=1-00:00:00 \
        --output="${OUTDIR}/slurm_%j.log" \
        --error="${OUTDIR}/slurm_%j.err" \
        --wrap="
            set -euo pipefail
            eval \"\$(${MAMBA_EXE} shell hook --shell bash)\"
            micromamba activate insider
            cd ${REPO}
            python ${SCRIPT} \
                --task ${TASK} \
                --category ${CAT} \
                --ablation ${ABL} \
                --preset full \
                --output-dir ${OUTDIR} \
                --no-verify-first \
                --seed 42 \
                ${MODELS_ARG}
        "
    TOTAL=$((TOTAL + 1))
}

# =====================================================================
# 1. transformer_sota: 3 sub-shards per (task, ablation) = 18 jobs
#    Use split sub-dirs: transformer_sota_A, _B, _C
#    Paper table generator reads all transformer_sota* dirs
# =====================================================================
echo "--- transformer_sota (18 GPU jobs, 5 models each) ---"
for task in "${TASKS[@]}"; do
    for abl in "${ABLATIONS[@]}"; do
        #                  TASK   CATEGORY             ABL  MODELS      JOBNAME                      OUTDIR_SUFFIX
        submit_gpu_shard "$task" "transformer_sota" "${abl}" \
            "${TSPLIT_A}" "b3ts_${task}_tsA_${abl}" "transformer_sota_A"

        submit_gpu_shard "$task" "transformer_sota" "${abl}" \
            "${TSPLIT_B}" "b3ts_${task}_tsB_${abl}" "transformer_sota_B"

        submit_gpu_shard "$task" "transformer_sota" "${abl}" \
            "${TSPLIT_C}" "b3ts_${task}_tsC_${abl}" "transformer_sota_C"
    done
done

# =====================================================================
# 2. autofit: 1 job per (task, ablation) on GPU = 6 jobs
# =====================================================================
echo ""
echo "--- autofit (6 GPU jobs) ---"
for task in "${TASKS[@]}"; do
    for abl in "${ABLATIONS[@]}"; do
        submit_gpu_shard "$task" "autofit" "${abl}" \
            "" "b3af_${task}_af_${abl}" "autofit"
    done
done

echo ""
echo "=== Total submitted: ${TOTAL} GPU jobs ==="
echo "Monitor: squeue -u \$USER | grep -E 'b3ts|b3af'"
echo ""
echo "Remaining running jobs (ml_tabular, deep_classical, statistical):"
echo "  squeue -u \$USER | grep -E 'b3f_|b3mx'"
