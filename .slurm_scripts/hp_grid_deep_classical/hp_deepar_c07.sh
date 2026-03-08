#!/usr/bin/env bash
#SBATCH --job-name=hp_deepar_c07
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=1-12:00:00
#SBATCH --mem=384G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/hp_grid_deep_classical/hp_deepar_c07_%j.out
#SBATCH --error=/work/projects/eint/logs/hp_grid_deep_classical/hp_deepar_c07_%j.err
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
echo "HP Grid Search | Job ${SLURM_JOB_ID} | Config #7"
echo "$(date -Iseconds) | $(hostname)"
echo "Model: DeepAR | Category: deep_classical"
echo "Task: task1_outcome | Ablation: core_only"
echo "HP Config: {\"DeepAR\": {\"batch_size\": 32, \"hidden_size\": 64, \"input_size\": 90, \"learning_rate\": 0.001, \"max_steps\": 2000"
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
    --category deep_classical \
    --ablation core_only \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/hp_grid/deep_classical/DeepAR/config_07 \
    --seed 42 \
    --no-verify-first \
    --models DeepAR \
    --model-kwargs-json '{"DeepAR": {"batch_size": 32, "hidden_size": 64, "input_size": 90, "learning_rate": 0.001, "max_steps": 2000, "n_layers": 3}}' &

HARNESS_PID=$!
wait "$HARNESS_PID"

echo "Done: $(date -Iseconds)"
