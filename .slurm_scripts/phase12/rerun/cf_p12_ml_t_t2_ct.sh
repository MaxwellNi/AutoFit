#!/usr/bin/env bash
#SBATCH --job-name=cf_p12_ml_t_t2_ct
#SBATCH --account=christian.fisch
#SBATCH --partition=bigmem
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=320G
#SBATCH --cpus-per-task=28
#SBATCH --output=/work/projects/eint/logs/phase12/cf_p12_ml_t_t2_ct_%j.out
#SBATCH --error=/work/projects/eint/logs/phase12/cf_p12_ml_t_t2_ct_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -e
umask 002
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
# HF_HOME: cfisch uses own ~/.cache/huggingface
cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing"; exit 2
fi
echo "============================================================"
echo "Phase 12 Text Re-run | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${INSIDER_PY} -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

# Verify text embeddings exist
if [[ ! -f "runs/text_embeddings/text_embeddings.parquet" ]]; then
  echo "FATAL: text_embeddings.parquet not found — cannot run text ablation!"; exit 3
fi
echo "Text embeddings verified: $(ls -lh runs/text_embeddings/text_embeddings.parquet)"

echo "Task: task2_forecast | Cat: ml_tabular | Abl: core_text | Models: CatBoost,ExtraTrees,HistGradientBoosting,LightGBM,LightGBMTweedie,MeanPredictor,NegativeBinomialGLM,RandomForest,SeasonalNaive,XGBoost,XGBoostPoisson"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category ml_tabular --ablation core_text \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task2_forecast/ml_tabular/core_text --seed 42 \
    --no-verify-first --models CatBoost,ExtraTrees,HistGradientBoosting,LightGBM,LightGBMTweedie,MeanPredictor,NegativeBinomialGLM,RandomForest,SeasonalNaive,XGBoost,XGBoostPoisson
echo "Done: $(date -Iseconds)"
