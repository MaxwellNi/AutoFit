#!/usr/bin/env bash
# =============================================================================
# Block 3 Full Benchmark — 4090 Dual GPU Launch Script
# 44 models × 3 tasks × 2 ablations = full KDD'26 benchmark
#
# Usage: bash scripts/run_block3_full_4090.sh
# Must run inside: conda activate insider
# Launches inside tmux session "b3_full"
# =============================================================================
set -euo pipefail

STAMP="20260203_225620"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUTROOT="${REPO}/runs/benchmarks/block3_${STAMP}_4090_final"
PRESET="full"

# RAM safety: 252 GB total, reserve 5% = ~12 GB → limit processes
TOTAL_RAM_GB=252
RESERVE_PCT=5
MAX_RAM_GB=$(( TOTAL_RAM_GB * (100 - RESERVE_PCT) / 100 ))

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "full")

# GPU0 categories (heavier: deep learning)
GPU0_CATS=("deep_classical" "transformer_sota" "foundation")
# GPU1 categories (lighter: CPU-bound mostly)
GPU1_CATS=("ml_tabular" "statistical" "irregular")

echo "=============================================="
echo "Block 3 Full Benchmark — 4090 Dual GPU"
echo "=============================================="
echo "Stamp:   ${STAMP}"
echo "Output:  ${OUTROOT}"
echo "Preset:  ${PRESET}"
echo "Tasks:   ${TASKS[*]}"
echo "Ablations: ${ABLATIONS[*]}"
echo "Max RAM: ${MAX_RAM_GB} GB (${RESERVE_PCT}% reserved)"
echo "GPU0:    ${GPU0_CATS[*]}"
echo "GPU1:    ${GPU1_CATS[*]}"
echo "=============================================="

mkdir -p "${OUTROOT}/logs"

# Function: run one category/task/ablation
run_shard() {
    local gpu_id=$1
    local task=$2
    local category=$3
    local ablation=$4
    local outdir="${OUTROOT}/${task}/${category}/${ablation}"
    local logfile="${OUTROOT}/logs/${task}_${category}_${ablation}.log"

    mkdir -p "${outdir}"
    echo "[$(date '+%H:%M:%S')] GPU${gpu_id} START: ${task}/${category}/${ablation}"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    python "${REPO}/scripts/run_block3_benchmark_shard.py" \
        --task "${task}" \
        --category "${category}" \
        --ablation "${ablation}" \
        --preset "${PRESET}" \
        --output-dir "${outdir}" \
        --seed 42 \
        2>&1 | tee "${logfile}"

    echo "[$(date '+%H:%M:%S')] GPU${gpu_id} DONE:  ${task}/${category}/${ablation}"
}

# ============================================================================
# GPU0 worker: deep_classical, transformer_sota, foundation
# ============================================================================
run_gpu0() {
    echo "[GPU0] Starting deep/transformer/foundation categories..."
    for task in "${TASKS[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            for cat in "${GPU0_CATS[@]}"; do
                run_shard 0 "${task}" "${cat}" "${ablation}"
            done
        done
    done
    echo "[GPU0] ALL COMPLETE at $(date)"
}

# ============================================================================
# GPU1 worker: ml_tabular, statistical, irregular
# ============================================================================
run_gpu1() {
    echo "[GPU1] Starting ml_tabular/statistical/irregular categories..."
    for task in "${TASKS[@]}"; do
        for ablation in "${ABLATIONS[@]}"; do
            for cat in "${GPU1_CATS[@]}"; do
                run_shard 1 "${task}" "${cat}" "${ablation}"
            done
        done
    done
    echo "[GPU1] ALL COMPLETE at $(date)"
}

# ============================================================================
# Launch: two background processes in current shell
# ============================================================================
echo ""
echo "Launching GPU0 and GPU1 workers in parallel..."
echo "Logs: ${OUTROOT}/logs/"
echo ""

run_gpu0 > "${OUTROOT}/logs/gpu0_master.log" 2>&1 &
GPU0_PID=$!

run_gpu1 > "${OUTROOT}/logs/gpu1_master.log" 2>&1 &
GPU1_PID=$!

echo "GPU0 PID: ${GPU0_PID}"
echo "GPU1 PID: ${GPU1_PID}"
echo ""
echo "Monitor progress:"
echo "  tail -f ${OUTROOT}/logs/gpu0_master.log"
echo "  tail -f ${OUTROOT}/logs/gpu1_master.log"
echo ""

# Wait for both
wait ${GPU0_PID}
G0_EXIT=$?
wait ${GPU1_PID}
G1_EXIT=$?

echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "  GPU0 exit: ${G0_EXIT}"
echo "  GPU1 exit: ${G1_EXIT}"
echo "  Output:    ${OUTROOT}"
echo "=============================================="
