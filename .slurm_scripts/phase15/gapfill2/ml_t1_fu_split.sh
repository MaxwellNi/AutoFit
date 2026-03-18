#!/bin/bash
#SBATCH --job-name=ml_t1_fu_sp
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:0
#SBATCH --output=/work/projects/eint/logs/phase15/ml_t1_fu_sp_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/ml_t1_fu_sp_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue
set -euo pipefail
umask 002
cd /home/users/npin/repo_root
INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
echo "ML t1_fu split | Job $SLURM_JOB_ID on $(hostname)"
echo "$(date -Iseconds) | Git: $(git rev-parse --short HEAD)"

# Run models ONE BY ONE to avoid 640G OOM from running all 10 at once
for MODEL in CatBoost ExtraTrees HistGradientBoosting LightGBM LightGBMTweedie MeanPredictor RandomForest SeasonalNaive XGBoost XGBoostPoisson; do
    echo "=== $(date -Iseconds) Starting $MODEL ==="
    $INSIDER_PY scripts/run_block3_benchmark_shard.py \
        --task task1_outcome --category ml_tabular --ablation full \
        --models "$MODEL" \
        --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/ml_tabular/full --seed 42 \
    || echo "WARN: $MODEL failed, continuing..."
    echo "=== $(date -Iseconds) Finished $MODEL ==="
done
echo "All ml_tabular models done"
