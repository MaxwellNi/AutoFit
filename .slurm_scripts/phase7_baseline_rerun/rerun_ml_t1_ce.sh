#!/usr/bin/env bash
#SBATCH --job-name=rerun_ml_t1_ce
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=1-00:00:00
#SBATCH --mem=384G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase7_baseline_rerun/rerun_ml_t1_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_baseline_rerun/rerun_ml_t1_ce_%j.err
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
echo "Baseline Re-run | Job ${SLURM_JOB_ID}"
echo "$(date -Iseconds) | $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Task: task1_outcome | Category: ml_tabular | Ablation: core_edgar"
echo "Purpose: Fix EDGAR feature mismatch (104→105 features)"
echo "============================================================"

${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category ml_tabular \
    --ablation core_edgar \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar \
    --seed 42 \
    --no-verify-first \
    --enable-global-dedup &

HARNESS_PID=$!
wait "$HARNESS_PID"

echo "Done: $(date -Iseconds)"
