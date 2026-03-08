#!/usr/bin/env bash
#SBATCH --job-name=p9_mlT_t3_co
#SBATCH --account=npin
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=112G
#SBATCH --cpus-per-task=28
#SBATCH --output=/work/projects/eint/logs/phase9/p9_mlT_t3_co_%j.out
#SBATCH --error=/work/projects/eint/logs/phase9/p9_mlT_t3_co_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"; exit 2
fi
echo "============================================================"
echo "Phase 9 Fair Benchmark | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(python3 -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

echo "Task: task3_risk_adjust | Cat: ml_tabular | Abl: core_only | Models: LightGBM,XGBoost,CatBoost,RandomForest,ExtraTrees,HistGradientBoosting,Ridge,Lasso,ElasticNet,SVR,KNN,QuantileRegressor,MeanPredictor,SeasonalNaive,LightGBMTweedie,XGBoostPoisson,LogisticRegression,NegativeBinomialGLM"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task3_risk_adjust --category ml_tabular --ablation core_only \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task3_risk_adjust/ml_tabular/core_only --seed 42 \
    --no-verify-first --models LightGBM,XGBoost,CatBoost,RandomForest,ExtraTrees,HistGradientBoosting,Ridge,Lasso,ElasticNet,SVR,KNN,QuantileRegressor,MeanPredictor,SeasonalNaive,LightGBMTweedie,XGBoostPoisson,LogisticRegression,NegativeBinomialGLM
echo "Done: $(date -Iseconds)"
