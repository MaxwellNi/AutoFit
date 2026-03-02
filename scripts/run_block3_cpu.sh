#!/bin/bash
# Block 3 Benchmark - CPU Window Script
# 运行所有 CPU 类任务（statistical + ml_tabular）
#
# 使用方法：在 tmux 的 cpu 窗口中执行本脚本

set -e

cd ~/projects/repo_root

STAMP="20260203_225620"
RUN_ROOT="runs/benchmarks/block3_${STAMP}_4090_standard"

# 定义回归模型（排除分类器）
ML_MODELS="Ridge,Lasso,ElasticNet,SVR,KNN,RandomForest,ExtraTrees,HistGradientBoosting,LightGBM,XGBoost,CatBoost,MeanPredictor,SeasonalNaive"
STAT_MODELS="AutoARIMA,ETS,Theta,MSTL,SF_SeasonalNaive"

TASKS="task1_outcome task2_forecast task3_risk_adjust"
ABLATIONS="core_only full"

echo "=============================================="
echo "Block 3 CPU Benchmark"
echo "RUN_ROOT: ${RUN_ROOT}"
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
        # Statistical
        COUNT=$((COUNT + 1))
        OUTDIR="${RUN_ROOT}/${TASK}/statistical/${ABLATION}"
        echo ""
        echo "[$COUNT/$TOTAL] ${TASK} / statistical / ${ABLATION}"
        echo "输出: ${OUTDIR}"
        
        python scripts/run_block3_benchmark_shard.py \
            --preset standard \
            --task "${TASK}" \
            --category statistical \
            --ablation "${ABLATION}" \
            --models "${STAT_MODELS}" \
            --output-dir "${OUTDIR}" \
            --seed 42
        
        # 验证输出
        if [ ! -f "${OUTDIR}/MANIFEST.json" ] || [ ! -f "${OUTDIR}/metrics.json" ]; then
            echo "FATAL: Missing MANIFEST.json or metrics.json in ${OUTDIR}"
            exit 1
        fi
        echo "[OK] ${OUTDIR}"
        
        # ML Tabular
        COUNT=$((COUNT + 1))
        OUTDIR="${RUN_ROOT}/${TASK}/ml_tabular/${ABLATION}"
        echo ""
        echo "[$COUNT/$TOTAL] ${TASK} / ml_tabular / ${ABLATION}"
        echo "输出: ${OUTDIR}"
        
        python scripts/run_block3_benchmark_shard.py \
            --preset standard \
            --task "${TASK}" \
            --category ml_tabular \
            --ablation "${ABLATION}" \
            --models "${ML_MODELS}" \
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
echo "CPU Benchmark 完成！"
echo "结束时间: $(date)"
echo "=============================================="
