#!/bin/bash
# Foundation models on GPU
set -e
cd /home/pni/projects/repo_root

export CUDA_VISIBLE_DEVICES=${1:-0}
OUTBASE="runs/benchmarks/block3_20260203_225620_4090_standard"
TASKS="task1_outcome task2_forecast task3_risk_adjust"
ABLATIONS="core_only full"

echo "=== Foundation models on GPU $CUDA_VISIBLE_DEVICES ==="

for TASK in $TASKS; do
    for ABL in $ABLATIONS; do
        OUTDIR="$OUTBASE/$TASK/foundation/$ABL"
        mkdir -p "$OUTDIR"
        echo "[$(date)] Starting: $TASK / foundation / $ABL"
        
        python scripts/run_block3_benchmark_shard.py \
            --preset standard \
            --task $TASK \
            --category foundation \
            --ablation $ABL \
            --output-dir "$OUTDIR" 2>&1 || echo "WARNING: $TASK/$ABL failed"
        
        echo "[$(date)] Completed: $TASK / foundation / $ABL"
    done
done
echo "=== Foundation done ==="
