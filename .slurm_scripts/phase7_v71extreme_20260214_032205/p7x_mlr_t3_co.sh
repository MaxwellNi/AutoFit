#!/usr/bin/env bash
#SBATCH --job-name=p7x_mlr_t3_co
#SBATCH --account=yves.letraon
#SBATCH --partition=batch
#SBATCH --qos=iris-batch-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=112G
#SBATCH --cpus-per-task=28


#SBATCH --output=/work/projects/eint/logs/phase7_v71extreme_20260214_032205/p7x_mlr_t3_co_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_v71extreme_20260214_032205/p7x_mlr_t3_co_%j.err
#SBATCH --export=ALL
#SBATCH --signal=B:USR1@120

set -e
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(which python3)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

echo "Task: task3_risk_adjust | Category: ml_tabular | Ablation: core_only"
echo "Models: RandomForest,ExtraTrees,LightGBM,XGBoost"
echo "Preset: full | Seed: 42"
echo "Output: runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/ml_refs/task3_risk_adjust/ml_tabular/core_only"
echo "============================================================"

CMD=(
    python3 scripts/run_block3_benchmark_shard.py
    --task task3_risk_adjust
    --category ml_tabular
    --ablation core_only
    --preset full
    --output-dir runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/ml_refs/task3_risk_adjust/ml_tabular/core_only
    --seed 42
    --no-verify-first
)
if [[ -n "RandomForest,ExtraTrees,LightGBM,XGBoost" ]]; then
    CMD+=(--models "RandomForest,ExtraTrees,LightGBM,XGBoost")
fi
if [[ -n "/home/users/npin/repo_root/.slurm_scripts/phase7_v71extreme_20260214_032205/p7x_mlr_t3_co_model_kwargs.json" ]]; then
    CMD+=(--model-kwargs-file "/home/users/npin/repo_root/.slurm_scripts/phase7_v71extreme_20260214_032205/p7x_mlr_t3_co_model_kwargs.json")
fi
"${CMD[@]}"

echo "Done: $(date -Iseconds)"
