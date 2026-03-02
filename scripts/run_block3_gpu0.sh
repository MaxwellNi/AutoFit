#!/bin/bash
# Block 3 Benchmark - GPU0 Window Script
# 运行 GPU 类任务的前半部分（deep_classical + transformer_sota 的一半）
#
# 使用方法：在 tmux 的 gpu0 窗口中执行本脚本

set -e

cd ~/projects/repo_root
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

STAMP="20260203_225620"
RUN_ROOT="runs/benchmarks/block3_${STAMP}_4090_standard"

DEEP_MODELS="NBEATS,NHITS,TFT,DeepAR"
TRANSFORMER_MODELS="PatchTST,iTransformer"  # GPU0 跑一半

TASKS="task1_outcome task2_forecast task3_risk_adjust"
ABLATIONS="core_only full"

echo "=============================================="
echo "Block 3 GPU0 Benchmark"
echo "RUN_ROOT: ${RUN_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "开始时间: $(date)"
echo "=============================================="

# 先验证 freeze gate
python scripts/block3_verify_freeze.py --pointer docs/audits/FULL_SCALE_POINTER.yaml
if [ $? -ne 0 ]; then
    echo "FATAL: Freeze gate FAILED!"
    exit 1
fi

COUNT=0
TOTAL=$((3 * 2 * 2))  # 3 tasks * 2 categories * 2 ablations

for TASK in $TASKS; do
    for ABLATION in $ABLATIONS; do
        # Deep Classical
        COUNT=$((COUNT + 1))
        OUTDIR="${RUN_ROOT}/${TASK}/deep_classical/${ABLATION}"
        echo ""
        echo "[GPU0 $COUNT/$TOTAL] ${TASK} / deep_classical / ${ABLATION}"
        echo "输出: ${OUTDIR}"
        
        python scripts/run_block3_benchmark_shard.py \
            --preset standard \
            --task "${TASK}" \
            --category deep_classical \
            --ablation "${ABLATION}" \
            --models "${DEEP_MODELS}" \
            --output-dir "${OUTDIR}" \
            --seed 42
        
        # 验证输出
        if [ ! -f "${OUTDIR}/MANIFEST.json" ] || [ ! -f "${OUTDIR}/metrics.json" ]; then
            echo "FATAL: Missing MANIFEST.json or metrics.json in ${OUTDIR}"
            exit 1
        fi
        echo "[OK] ${OUTDIR}"
        
        # Transformer SOTA (前半)
        COUNT=$((COUNT + 1))
        OUTDIR="${RUN_ROOT}/${TASK}/transformer_sota_0/${ABLATION}"
        echo ""
        echo "[GPU0 $COUNT/$TOTAL] ${TASK} / transformer_sota_0 / ${ABLATION}"
        echo "输出: ${OUTDIR}"
        
        python scripts/run_block3_benchmark_shard.py \
            --preset standard \
            --task "${TASK}" \
            --category transformer_sota \
            --ablation "${ABLATION}" \
            --models "${TRANSFORMER_MODELS}" \
            --output-dir "${OUTDIR}" \
            --seed 42
        
        # 验证输出
        if [ ! -f "${OUTDIR}/MANIFEST.json" ] || [ ! -f "${OUTDIR}/metrics.json" ]; then
            echo "FATAL: Missing MANIFEST.json or metrics.json in ${OUTDIR}"
            exit 1
        fi
        echo "[OK] ${OUTDIR}"
    done
done

echo ""
echo "=============================================="
echo "GPU0 Benchmark 完成！"
echo "结束时间: $(date)"
echo "=============================================="
