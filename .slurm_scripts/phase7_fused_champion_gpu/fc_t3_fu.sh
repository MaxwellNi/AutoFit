#!/usr/bin/env bash
#SBATCH --job-name=fc_t3_fu
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=384G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:volta:1
#SBATCH --output=/work/projects/eint/logs/phase7_fused_champion/fc_t3_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_fused_champion/fc_t3_fu_%j.err
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --signal=USR1@120

set -e

# ── Preemption handler (断点续跑) ──
handle_preempt() {
    echo "PREEMPT: SIGUSR1 received at $(date -Iseconds) — saving partial results"
    wait "$HARNESS_PID" 2>/dev/null || true
    echo "PREEMPT: Requeue count: ${SLURM_RESTART_COUNT:-0}"
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
echo "FusedChampion V7.3.2 | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Partition: ${SLURM_JOB_PARTITION} | QOS: iris-gpu-long"
echo "Python: $(which python3) | $(python3 -V)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>/dev/null || echo N/A)"
echo "============================================================"

python3 - <<'PY'
import sys, torch
if "insider" not in sys.executable:
    raise SystemExit("FATAL: not insider python")
if sys.version_info < (3, 11):
    raise SystemExit("FATAL: python >= 3.11 required")
if not torch.cuda.is_available():
    raise SystemExit("FATAL: GPU required")
PY

${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"

echo "Task: task3_risk_adjust | Ablation: full | Model: FusedChampion"
echo "Output: runs/benchmarks/block3_20260203_225620_phase7/task3_risk_adjust/autofit/full"
echo "============================================================"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task3_risk_adjust \
    --category autofit \
    --ablation full \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task3_risk_adjust/autofit/full \
    --seed 42 \
    --no-verify-first \
    --models FusedChampion &

HARNESS_PID=$!
wait "$HARNESS_PID"

echo "Done: $(date -Iseconds)"
