#!/bin/bash
# Fast ml_tabular - skip slow models (SVR, KNN)
set -e
cd /home/pni/projects/repo_root

OUTBASE="runs/benchmarks/block3_20260203_225620_4090_standard"
TASKS="task1_outcome task2_forecast task3_risk_adjust"
ABLATIONS="core_only full"

# Fast models only (skip SVR, KNN, LogisticRegression)
MODELS="Ridge,Lasso,ElasticNet,RandomForest,ExtraTrees,HistGradientBoosting,LightGBM,XGBoost,CatBoost,SeasonalNaive,MeanPredictor"

echo "=== Fast ml_tabular benchmark ==="
echo "Models: $MODELS"

for TASK in $TASKS; do
    for ABL in $ABLATIONS; do
        OUTDIR="$OUTBASE/$TASK/ml_tabular/$ABL"
        mkdir -p "$OUTDIR"
        echo "[$(date)] Starting: $TASK / ml_tabular / $ABL"
        
        python scripts/run_block3_benchmark_shard.py \
            --preset standard \
            --task $TASK \
            --category ml_tabular \
            --models "$MODELS" \
            --ablation $ABL \
            --output-dir "$OUTDIR" || echo "WARNING: $TASK/$ABL failed"
        
        echo "[$(date)] Completed: $TASK / ml_tabular / $ABL"
    done
done
echo "=== All done ==="
