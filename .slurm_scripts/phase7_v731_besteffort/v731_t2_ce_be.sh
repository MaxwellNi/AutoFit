#!/usr/bin/env bash
#SBATCH --job-name=v731_t2_ce_be
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu,l40s,hopper
#SBATCH --qos=besteffort
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/work/projects/eint/logs/phase7_v731/v731_t2_ce_be_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_v731/v731_t2_ce_be_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e

# ── Handle SIGUSR1 for checkpoint-before-preemption ──
# The benchmark harness already handles USR1 (saves partial metrics.json),
# but we also need to ensure requeue happens cleanly at the shell level.
handle_preempt() {
    echo "PREEMPT: SIGUSR1 received at $(date -Iseconds) — saving & requeueing"
    # Give harness time to handle the signal and flush
    wait "$HARNESS_PID" 2>/dev/null || true
    echo "PREEMPT: Job will be automatically requeued by SLURM --requeue"
    exit 0
}
trap handle_preempt USR1

# ── Environment setup ──
export CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /mnt/aiongpfs/projects/eint/repo_root

INSIDER_PY="${CONDA_PREFIX}/bin/python3"
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing or non-executable: ${INSIDER_PY}"
  exit 2
fi

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Partition: ${SLURM_JOB_PARTITION}  |  QOS: besteffort (preemptible)"
echo "Requeue count: ${SLURM_RESTART_COUNT:-0}"
echo "Python: $(which python3)"
echo "PythonV: $(python3 -V)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA:   $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"
echo "============================================================"

python3 - <<'PY'
import sys, torch
print("sys.executable:", sys.executable)
if "insider" not in sys.executable:
    raise SystemExit("FATAL: runtime python is not insider")
if sys.version_info < (3, 11):
    raise SystemExit(f"FATAL: python >={3}.{11} required, got {sys.version_info}")
if not torch.cuda.is_available():
    raise SystemExit("FATAL: GPU required but torch.cuda.is_available()=False")
PY

${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"

echo "Task: task2_forecast | Category: autofit | Ablation: core_edgar"
echo "Models: AutoFitV731"
echo "Preset: full | Seed: 42"
echo "Output: runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/autofit/core_edgar"
echo "============================================================"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast \
    --category autofit \
    --ablation core_edgar \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/autofit/core_edgar \
    --seed 42 \
    --no-verify-first \
    --models AutoFitV731 &

HARNESS_PID=$!
wait "$HARNESS_PID"

echo "Done: $(date -Iseconds)"
