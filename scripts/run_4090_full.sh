#!/bin/bash
# Full 4090 benchmark execution - 3 parallel workers
# GPU0: deep_classical + transformer_sota (half)
# GPU1: transformer_sota (other half) + foundation
# CPU : ml_tabular (remaining tasks)
#
# RAM budget: 252GB total, 3 workers × ~35GB = 105GB, leaving ~95GB (38%) free
set -e
cd /home/pni/projects/repo_root
mkdir -p logs

OUTBASE="runs/benchmarks/block3_20260203_225620_4090_standard"
STAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================================
# WORKER 1 (GPU0): deep_classical — all 3 tasks × 2 ablations
# ============================================================================
run_gpu0() {
    export CUDA_VISIBLE_DEVICES=0
    local LOG="logs/gpu0_deep_classical_${STAMP}.log"
    echo "[$(date)] GPU0: Starting deep_classical" | tee -a "$LOG"

    for TASK in task1_outcome task2_forecast task3_risk_adjust; do
        for ABL in core_only full; do
            OUTDIR="${OUTBASE}/${TASK}/deep_classical/${ABL}"
            mkdir -p "$OUTDIR"
            echo "[$(date)] GPU0: ${TASK}/deep_classical/${ABL}" | tee -a "$LOG"
            python scripts/run_block3_benchmark_shard.py \
                --preset standard \
                --task "$TASK" \
                --category deep_classical \
                --ablation "$ABL" \
                --output-dir "$OUTDIR" 2>&1 | tee -a "$LOG" || \
                echo "[$(date)] WARNING: ${TASK}/deep_classical/${ABL} failed" | tee -a "$LOG"
        done
    done

    # After deep_classical, run transformer_sota tasks 1-2 on GPU0
    for TASK in task1_outcome task2_forecast; do
        for ABL in core_only full; do
            OUTDIR="${OUTBASE}/${TASK}/transformer_sota/${ABL}"
            mkdir -p "$OUTDIR"
            echo "[$(date)] GPU0: ${TASK}/transformer_sota/${ABL}" | tee -a "$LOG"
            python scripts/run_block3_benchmark_shard.py \
                --preset standard \
                --task "$TASK" \
                --category transformer_sota \
                --ablation "$ABL" \
                --output-dir "$OUTDIR" 2>&1 | tee -a "$LOG" || \
                echo "[$(date)] WARNING: ${TASK}/transformer_sota/${ABL} failed" | tee -a "$LOG"
        done
    done

    echo "[$(date)] GPU0: ALL DONE" | tee -a "$LOG"
}

# ============================================================================
# WORKER 2 (GPU1): foundation + transformer_sota task3
# ============================================================================
run_gpu1() {
    export CUDA_VISIBLE_DEVICES=1
    local LOG="logs/gpu1_foundation_${STAMP}.log"
    echo "[$(date)] GPU1: Starting foundation" | tee -a "$LOG"

    for TASK in task1_outcome task2_forecast task3_risk_adjust; do
        for ABL in core_only full; do
            OUTDIR="${OUTBASE}/${TASK}/foundation/${ABL}"
            mkdir -p "$OUTDIR"
            echo "[$(date)] GPU1: ${TASK}/foundation/${ABL}" | tee -a "$LOG"
            python scripts/run_block3_benchmark_shard.py \
                --preset standard \
                --task "$TASK" \
                --category foundation \
                --ablation "$ABL" \
                --output-dir "$OUTDIR" 2>&1 | tee -a "$LOG" || \
                echo "[$(date)] WARNING: ${TASK}/foundation/${ABL} failed" | tee -a "$LOG"
        done
    done

    # After foundation, run transformer_sota task3 on GPU1
    for ABL in core_only full; do
        OUTDIR="${OUTBASE}/task3_risk_adjust/transformer_sota/${ABL}"
        mkdir -p "$OUTDIR"
        echo "[$(date)] GPU1: task3_risk_adjust/transformer_sota/${ABL}" | tee -a "$LOG"
        python scripts/run_block3_benchmark_shard.py \
            --preset standard \
            --task task3_risk_adjust \
            --category transformer_sota \
            --ablation "$ABL" \
            --output-dir "$OUTDIR" 2>&1 | tee -a "$LOG" || \
            echo "[$(date)] WARNING: task3/transformer_sota/${ABL} failed" | tee -a "$LOG"
    done

    echo "[$(date)] GPU1: ALL DONE" | tee -a "$LOG"
}

# ============================================================================
# WORKER 3 (CPU): ml_tabular — remaining tasks + statistical
# ============================================================================
run_cpu() {
    local LOG="logs/cpu_ml_tabular_${STAMP}.log"
    # Fast models only — skip SVR, KNN (too slow for 4.4M rows)
    local MODELS="Ridge,Lasso,ElasticNet,RandomForest,ExtraTrees,HistGradientBoosting,LightGBM,XGBoost,CatBoost,SeasonalNaive,MeanPredictor"
    echo "[$(date)] CPU: Starting ml_tabular (fast)" | tee -a "$LOG"

    for TASK in task2_forecast task3_risk_adjust; do
        for ABL in core_only full; do
            OUTDIR="${OUTBASE}/${TASK}/ml_tabular/${ABL}"
            mkdir -p "$OUTDIR"
            echo "[$(date)] CPU: ${TASK}/ml_tabular/${ABL}" | tee -a "$LOG"
            python scripts/run_block3_benchmark_shard.py \
                --preset standard \
                --task "$TASK" \
                --category ml_tabular \
                --models "$MODELS" \
                --ablation "$ABL" \
                --output-dir "$OUTDIR" 2>&1 | tee -a "$LOG" || \
                echo "[$(date)] WARNING: ${TASK}/ml_tabular/${ABL} failed" | tee -a "$LOG"
        done
    done

    echo "[$(date)] CPU: ALL DONE" | tee -a "$LOG"
}

# ============================================================================
# Launch all 3 workers in parallel
# ============================================================================
echo "==================================================================="
echo "  Block 3 Full Benchmark — 4090 (3 parallel workers)"
echo "  Started: $(date)"
echo "  GPU0: deep_classical + transformer_sota (tasks 1-2)"
echo "  GPU1: foundation + transformer_sota (task 3)"
echo "  CPU:  ml_tabular (tasks 2-3, fast models)"
echo "==================================================================="

run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

run_cpu &
PID_CPU=$!

echo "Worker PIDs: GPU0=$PID_GPU0, GPU1=$PID_GPU1, CPU=$PID_CPU"
echo "Logs: logs/gpu0_deep_classical_${STAMP}.log, logs/gpu1_foundation_${STAMP}.log, logs/cpu_ml_tabular_${STAMP}.log"

# Wait for all workers
wait $PID_GPU0
echo "[$(date)] GPU0 worker finished (exit=$?)"
wait $PID_GPU1
echo "[$(date)] GPU1 worker finished (exit=$?)"
wait $PID_CPU
echo "[$(date)] CPU worker finished (exit=$?)"

echo ""
echo "==================================================================="
echo "  Block 3 Full Benchmark COMPLETE — $(date)"
echo "==================================================================="

# Summary
echo ""
echo "=== Results Summary ==="
find "$OUTBASE" -name "metrics.json" -exec python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
if isinstance(data, list) and data:
    for r in data[:2]:
        m = r.get('model_name','?')
        mae = r.get('mae', 'N/A')
        cat = r.get('category','?')
        task = r.get('task','?')
        print(f'  {task}/{cat}: {m} MAE={mae:.0f}' if isinstance(mae, float) else f'  {task}/{cat}: {m} MAE={mae}')
" {} \; 2>/dev/null | head -40
