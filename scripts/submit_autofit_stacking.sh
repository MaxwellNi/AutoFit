#!/usr/bin/env bash
# Submit 6 autofit GPU jobs: 3 tasks × 2 ablations
# Uses the new stacking-based AutoFit (Moirai base + LightGBM residual)
set -euo pipefail

STAMP="20260203_225620"
BASE_DIR="runs/benchmarks/block3_${STAMP}_iris_full"

MAMBA_ROOT="/mnt/aiongpfs/users/npin/.local/bin/micromamba"
ENV_ACTIVATE="eval \"\$($MAMBA_ROOT shell hook --shell bash)\" && micromamba activate insider"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_edgar")

for TASK in "${TASKS[@]}"; do
    for ABLATION in "${ABLATIONS[@]}"; do
        OUT_DIR="${BASE_DIR}/${TASK}/autofit/${ABLATION}"
        JOB_NAME="af_${TASK:4:1}_${ABLATION:5:2}"
        
        sbatch \
            --job-name="${JOB_NAME}" \
            --partition=gpu \
            --gres=gpu:1 \
            --cpus-per-task=7 \
            --mem=180G \
            --time=1-00:00:00 \
            --account=yves.letraon \
            --qos=normal \
            --output="logs/autofit_${TASK}_${ABLATION}_%j.out" \
            --error="logs/autofit_${TASK}_${ABLATION}_%j.err" \
            --signal=B:TERM@120 \
            --wrap="${ENV_ACTIVATE} && cd /home/users/npin/repo_root && python scripts/run_block3_benchmark_shard.py \
                --task ${TASK} \
                --category autofit \
                --ablation ${ABLATION} \
                --preset full \
                --output-dir ${OUT_DIR} \
                --seed 42"
        
        echo "Submitted ${JOB_NAME}: ${TASK} / ${ABLATION}"
    done
done

echo "All 6 autofit stacking jobs submitted."
