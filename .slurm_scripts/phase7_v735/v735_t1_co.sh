#!/usr/bin/env bash
#SBATCH --job-name=v735_t1_co
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase7_v735/v735_t1_co_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_v735/v735_t1_co_%j.err
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --signal=USR1@120

set -e

handle_preempt() {
    echo "PREEMPT: SIGUSR1 received at $(date -Iseconds)"
    wait "$HARNESS_PID" 2>/dev/null || true
    exit 0
}
trap handle_preempt USR1

export CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /mnt/aiongpfs/projects/eint/repo_root

INSIDER_PY="${CONDA_PREFIX}/bin/python3"
[[ -x "${INSIDER_PY}" ]] || { echo "FATAL: insider python missing"; exit 2; }

echo "============================================================"
echo "V7.3.5 Exact Oracle Champion | Job ${SLURM_JOB_ID}"
echo "$(date -Iseconds) | $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Task: task1_outcome | Ablation: core_only | Model: AutoFitV735"
echo "============================================================"

${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category autofit \
    --ablation core_only \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_only \
    --seed 42 \
    --no-verify-first \
    --models AutoFitV735 \
    --enable-global-dedup &

HARNESS_PID=$!
wait "$HARNESS_PID"

echo "Done: $(date -Iseconds)"
