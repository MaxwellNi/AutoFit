#!/bin/bash
# Re-run the 5 models affected by n_series=1 bug (now fixed)
# Models: iTransformer, TSMixer, RMoK, SOFTS, StemGNN
#
# These models all require n_series parameter matching the actual number
# of entities in the panel. The original run had n_series=1 (hardcoded),
# causing fallback predictions for all of them.

set -e

OUTROOT="runs/benchmarks/block3_20260203_225620_4090_final"
MODELS="iTransformer,TSMixer,RMoK,SOFTS,StemGNN"
CATEGORY="transformer_sota"
PRESET="full"

cd /home/pni/projects/repo_root

echo "=========================================="
echo "N-series fix re-run: $MODELS"
echo "Output: $OUTROOT"
echo "Started: $(date)"
echo "=========================================="

# All tasks × ablations for transformer_sota
for TASK in task1_outcome task2_forecast task3_risk_adjust; do
    for ABL in core_only full; do
        SHARD_DIR="${OUTROOT}/${TASK}/${CATEGORY}/${ABL}"
        echo ""
        echo "[$(date +%H:%M:%S)] RE-RUN: ${TASK}/${CATEGORY}/${ABL}"
        
        CUDA_VISIBLE_DEVICES=0 python scripts/run_block3_benchmark_shard.py \
            --task "$TASK" \
            --category "$CATEGORY" \
            --models "$MODELS" \
            --ablation "$ABL" \
            --preset "$PRESET" \
            --output-dir "$SHARD_DIR" \
            2>&1 | tee -a "${OUTROOT}/logs/nseries_rerun.log"
    done
done

echo ""
echo "=========================================="
echo "N-series re-run COMPLETE: $(date)"
echo "=========================================="
