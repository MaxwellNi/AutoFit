#!/usr/bin/env bash
#SBATCH --job-name=hp_catboost_c17
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=1-12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/hp_grid_ml_tabular/hp_catboost_c17_%j.out
#SBATCH --error=/work/projects/eint/logs/hp_grid_ml_tabular/hp_catboost_c17_%j.err
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --signal=USR1@120

set -e

# ── Preemption handler ──
handle_preempt() {
    echo "PREEMPT: SIGUSR1 received at $(date -Iseconds)"
    wait "$HARNESS_PID" 2>/dev/null || true
    echo "PREEMPT: Requeue count: ${SLURM_RESTART_COUNT:-0}"
    exit 0
}
trap handle_preempt USR1

export CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/aiongpfs/projects/eint/vendor/Time-Series-Library:${PYTHONPATH:-}"
cd /mnt/aiongpfs/projects/eint/repo_root

INSIDER_PY="${CONDA_PREFIX}/bin/python3"
[[ -x "${INSIDER_PY}" ]] || { echo "FATAL: insider python missing"; exit 2; }

echo "============================================================"
echo "HP Grid Search | Job ${SLURM_JOB_ID} | Config #17"
echo "$(date -Iseconds) | $(hostname)"
echo "Model: CatBoost | Category: ml_tabular"
echo "Task: task1_outcome | Ablation: core_only"
echo "HP Config: {\"CatBoost\": {\"bagging_temperature\": 0, \"depth\": 6, \"iterations\": 500, \"l2_leaf_reg\": 10, \"learning_rate\": 0"
echo "============================================================"

python3 - <<'PY'
import sys, torch
assert "insider" in sys.executable, "not insider python"
assert sys.version_info >= (3, 11), "python >= 3.11 required"
assert torch.cuda.is_available(), "GPU required"
PY

${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category ml_tabular \
    --ablation core_only \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/hp_grid/ml_tabular/CatBoost/config_17 \
    --seed 42 \
    --no-verify-first \
    --models CatBoost \
    --model-kwargs-json '{"CatBoost": {"bagging_temperature": 0, "depth": 6, "iterations": 500, "l2_leaf_reg": 10, "learning_rate": 0.1}}' &

HARNESS_PID=$!
wait "$HARNESS_PID"

echo "Done: $(date -Iseconds)"
