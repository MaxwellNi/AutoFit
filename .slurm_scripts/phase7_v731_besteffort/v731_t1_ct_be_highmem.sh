#!/usr/bin/env bash
#SBATCH --job-name=v731_t1_ct_hm
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu,hopper
#SBATCH --qos=besteffort
#SBATCH --time=2-00:00:00
#SBATCH --mem=450G
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/work/projects/eint/logs/phase7_v731/v731_t1_ct_hm_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_v731/v731_t1_ct_hm_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e

handle_preempt() {
    echo "PREEMPT: SIGUSR1 received at $(date -Iseconds) — saving & requeueing"
    wait "$HARNESS_PID" 2>/dev/null || true
    echo "PREEMPT: Job will be automatically requeued by SLURM --requeue"
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
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Partition: ${SLURM_JOB_PARTITION} | QOS: besteffort | MEM: 450G (highmem)"
echo "Requeue count: ${SLURM_RESTART_COUNT:-0}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

python3 - <<'PY'
import sys, torch
if "insider" not in sys.executable: raise SystemExit("FATAL: not insider")
if sys.version_info < (3, 11): raise SystemExit("FATAL: python >= 3.11 required")
if not torch.cuda.is_available(): raise SystemExit("FATAL: GPU required")
PY

${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"

echo "Task: task1_outcome | Ablation: core_text | Model: AutoFitV731 (highmem resub)"
echo "============================================================"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category autofit \
    --ablation core_text \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_text \
    --seed 42 \
    --no-verify-first \
    --models AutoFitV731 &

HARNESS_PID=$!
wait "$HARNESS_PID"
echo "Done: $(date -Iseconds)"
