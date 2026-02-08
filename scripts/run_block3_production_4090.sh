#!/usr/bin/env bash
# =============================================================================
# Block 3 PRODUCTION Benchmark — 4090 Dual GPU
# KDD'26 Full Paper Grade — 44 models × 3 tasks × 2 ablations
#
# CHANGES from previous (smoke-grade) run:
#   - deep_models.py: NF defaults (max_steps 1000-3000), early stopping,
#     proper hidden sizes, robust scaler, validation via val_size=h
#   - traditional_ml.py: GBDTs at 2000 iterations, early stopping,
#     RF/ET at 500 trees, SVR/KNN subsampled to 100K
#   - irregular_models.py: GRU-D/SAITS actually train (epochs=50, GPU)
#   - Harness: runs ALL targets (not just first)
#
# GPU distribution:
#   GPU0: deep_classical (4) + transformer_sota_A (8)
#   GPU1: transformer_sota_B (7) + foundation (3) + irregular (2)
#   CPU:  ml_tabular (15) + statistical (5)  (runs in parallel)
#
# Usage:
#   conda activate insider
#   bash scripts/run_block3_production_4090.sh
#
# Output: runs/benchmarks/block3_20260203_225620_4090_production/
# =============================================================================
set -euo pipefail

STAMP="20260203_225620"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUTROOT="${REPO}/runs/benchmarks/block3_${STAMP}_4090_production"
PRESET="full"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "full")

# Transformer SOTA split across GPUs for balanced load
TSOTA_GPU0="PatchTST,iTransformer,TimesNet,TSMixer,Informer,Autoformer,FEDformer,VanillaTransformer"
TSOTA_GPU1="TiDE,NBEATSx,BiTCN,KAN,RMoK,SOFTS,StemGNN"

echo "================================================================"
echo "Block 3 PRODUCTION Benchmark — 4090 Dual GPU"
echo "================================================================"
echo "Stamp:     ${STAMP}"
echo "Output:    ${OUTROOT}"
echo "Preset:    ${PRESET}"
echo "Tasks:     ${TASKS[*]}"
echo "Ablations: ${ABLATIONS[*]}"
echo ""
echo "GPU0:  deep_classical(4) + transformer_sota_A(8)"
echo "GPU1:  transformer_sota_B(7) + foundation(3) + irregular(2)"
echo "CPU:   ml_tabular(15) + statistical(5)"
echo "================================================================"
date
echo ""

mkdir -p "${OUTROOT}/logs"

# Function: run one shard
run_shard() {
    local gpu_id=$1
    local task=$2
    local category=$3
    local ablation=$4
    local models_flag="$5"  # empty string or "--models X,Y,Z"
    local outdir="${OUTROOT}/${task}/${category}/${ablation}"
    local tag="${task}_${category}_${ablation}"
    local logfile="${OUTROOT}/logs/${tag}.log"

    mkdir -p "${outdir}"
    echo "[$(date '+%H:%M:%S')] GPU${gpu_id} START: ${tag}"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    python "${REPO}/scripts/run_block3_benchmark_shard.py" \
        --task "${task}" \
        --category "${category}" \
        --ablation "${ablation}" \
        --preset "${PRESET}" \
        --output-dir "${outdir}" \
        --seed 42 \
        ${models_flag} \
        2>&1 | tee "${logfile}"

    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] GPU${gpu_id} DONE ✓: ${tag}"
    else
        echo "[$(date '+%H:%M:%S')] GPU${gpu_id} FAIL ✗: ${tag} (exit ${rc})"
    fi
    return $rc
}

# ============================================================================
# GPU0: deep_classical (all) + transformer_sota (first 8)
# ============================================================================
run_gpu0() {
    echo "[GPU0] Starting deep_classical + transformer_sota_A..."
    for task in "${TASKS[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            # deep_classical — all 4 models
            run_shard 0 "${task}" "deep_classical" "${ablation}" "" || true
            # transformer_sota — first 8 models (PatchTST through VanillaTransformer)
            run_shard 0 "${task}" "transformer_sota" "${ablation}" "--models ${TSOTA_GPU0}" || true
        done
    done
    echo "[GPU0] ALL COMPLETE at $(date)"
}

# ============================================================================
# GPU1: transformer_sota (last 7) + foundation + irregular
# ============================================================================
run_gpu1() {
    echo "[GPU1] Starting transformer_sota_B + foundation + irregular..."
    for task in "${TASKS[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            # transformer_sota — last 7 models (TiDE through StemGNN)
            run_shard 1 "${task}" "transformer_sota" "${ablation}" "--models ${TSOTA_GPU1}" || true
            # foundation — Chronos, Moirai, TimesFM
            run_shard 1 "${task}" "foundation" "${ablation}" "" || true
            # irregular — GRU-D, SAITS
            run_shard 1 "${task}" "irregular" "${ablation}" "" || true
        done
    done
    echo "[GPU1] ALL COMPLETE at $(date)"
}

# ============================================================================
# CPU: ml_tabular + statistical (no GPU needed)
# ============================================================================
run_cpu() {
    echo "[CPU] Starting ml_tabular + statistical..."
    for task in "${TASKS[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            # ml_tabular — 15 models (CPU only)
            run_shard -1 "${task}" "ml_tabular" "${ablation}" "" || true
            # statistical — 5 models
            run_shard -1 "${task}" "statistical" "${ablation}" "" || true
        done
    done
    echo "[CPU] ALL COMPLETE at $(date)"
}

# ============================================================================
# Launch all three workers in parallel
# ============================================================================
echo ""
echo "Launching 3 workers (GPU0, GPU1, CPU) in parallel..."
echo "Logs: ${OUTROOT}/logs/"
echo ""

run_gpu0 > "${OUTROOT}/logs/gpu0_master.log" 2>&1 &
GPU0_PID=$!

run_gpu1 > "${OUTROOT}/logs/gpu1_master.log" 2>&1 &
GPU1_PID=$!

run_cpu  > "${OUTROOT}/logs/cpu_master.log"  2>&1 &
CPU_PID=$!

echo "GPU0 PID: ${GPU0_PID}"
echo "GPU1 PID: ${GPU1_PID}"
echo "CPU  PID: ${CPU_PID}"
echo ""
echo "Monitor progress:"
echo "  tail -f ${OUTROOT}/logs/gpu0_master.log"
echo "  tail -f ${OUTROOT}/logs/gpu1_master.log"
echo "  tail -f ${OUTROOT}/logs/cpu_master.log"
echo ""

# Wait for all three
wait ${GPU0_PID}
G0_EXIT=$?
wait ${GPU1_PID}
G1_EXIT=$?
wait ${CPU_PID}
CPU_EXIT=$?

echo ""
echo "================================================================"
echo "PRODUCTION BENCHMARK COMPLETE"
echo "  GPU0 exit:   ${G0_EXIT}  (deep_classical + transformer_sota_A)"
echo "  GPU1 exit:   ${G1_EXIT}  (transformer_sota_B + foundation + irregular)"
echo "  CPU  exit:   ${CPU_EXIT}  (ml_tabular + statistical)"
echo "  Output:      ${OUTROOT}"
echo "  Finished at: $(date)"
echo "================================================================"
