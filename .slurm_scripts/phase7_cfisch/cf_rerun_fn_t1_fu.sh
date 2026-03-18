#!/usr/bin/env bash
#SBATCH --job-name=cf_rerun_fn_t1_fu
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase7_cfisch/cf_rerun_fn_t1_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_cfisch/cf_rerun_fn_t1_fu_%j.err
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

# Pull latest code (oracle fixes 46d173a+)
git pull --ff-only origin main || echo "WARN: git pull failed"

INSIDER_PY="${CONDA_PREFIX}/bin/python3"
[[ -x "${INSIDER_PY}" ]] || { echo "FATAL: insider python missing"; exit 2; }

echo "============================================================"
echo "Baseline Re-run (cfisch) | Job ${SLURM_JOB_ID}"
echo "$(date -Iseconds) | $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Task: task1_outcome | Category: foundation | Ablation: full"
echo "============================================================"

${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py     --task task1_outcome     --category foundation     --ablation full     --preset full     --output-dir runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full     --seed 42     --no-verify-first     --enable-global-dedup &

HARNESS_PID=$!
wait "$HARNESS_PID"

echo "Done: $(date -Iseconds)"
