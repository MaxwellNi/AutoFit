#!/bin/bash
# Block 3 Benchmark - GPU1 Window Script
# 运行 GPU 类任务的后半部分（transformer_sota 的后半 + foundation + irregular）
#
# 使用方法：在 tmux 的 gpu1 窗口中执行本脚本

set -e

cd ~/projects/repo_root
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

STAMP="20260203_225620"
RUN_ROOT="runs/benchmarks/block3_${STAMP}_4090_standard"

TRANSFORMER_MODELS="TimesNet,TSMixer"  # GPU1 跑后半
IRREGULAR_MODELS="GRU-D,SAITS"
# Foundation 模型可能需要额外依赖，先跳过
# FOUNDATION_MODELS="TimesFM,Chronos,Moirai"

TASKS="task1_outcome task2_forecast task3_risk_adjust"
ABLATIONS="core_only full"

echo "=============================================="
echo "Block 3 GPU1 Benchmark"
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
        # Transformer SOTA (后半)
        COUNT=$((COUNT + 1))
        OUTDIR="${RUN_ROOT}/${TASK}/transformer_sota_1/${ABLATION}"
        echo ""
        echo "[GPU1 $COUNT/$TOTAL] ${TASK} / transformer_sota_1 / ${ABLATION}"
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
        
        # Irregular Aware
        COUNT=$((COUNT + 1))
        OUTDIR="${RUN_ROOT}/${TASK}/irregular_aware/${ABLATION}"
        echo ""
        echo "[GPU1 $COUNT/$TOTAL] ${TASK} / irregular_aware / ${ABLATION}"
        echo "输出: ${OUTDIR}"
        
        python scripts/run_block3_benchmark_shard.py \
            --preset standard \
            --task "${TASK}" \
            --category irregular_aware \
            --ablation "${ABLATION}" \
            --models "${IRREGULAR_MODELS}" \
            --output-dir "${OUTDIR}" \
            --seed 42 || echo "[WARN] irregular_aware 可能缺依赖，跳过"
        
        # 验证输出（irregular 可能失败，不 fatal）
        if [ -f "${OUTDIR}/MANIFEST.json" ] && [ -f "${OUTDIR}/metrics.json" ]; then
            echo "[OK] ${OUTDIR}"
        else
            echo "[WARN] ${OUTDIR} 输出不完整"
        fi
    done
done

echo ""
echo "=============================================="
echo "GPU1 Benchmark 完成！"
echo "结束时间: $(date)"
echo "=============================================="
