#!/usr/bin/env bash
# Phase 7 TSLib SOTA: 14 models × 3 tasks × 4 ablations = 12 shards
# Each shard runs ALL 14 TSLib models for one (task, ablation) combo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/work/projects/eint/logs/phase7_tslib"
mkdir -p "${LOG_DIR}"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_text" "core_edgar" "full")
TASK_SHORT=("t1" "t2" "t3")
ABL_SHORT=("co" "ct" "ce" "fu")

MODELS="TimeFilter,WPMixer,MultiPatchFormer,MSGNet,PAttn,MambaSimple,Koopa,FreTS,Crossformer,MICN,SegRNN,NonstationaryTransformer,FiLM,SCINet"

for ti in 0 1 2; do
  for ai in 0 1 2 3; do
    TASK="${TASKS[$ti]}"
    ABL="${ABLATIONS[$ai]}"
    TS="${TASK_SHORT[$ti]}"
    AS="${ABL_SHORT[$ai]}"
    JOB_NAME="tslib_${TS}_${AS}"
    SCRIPT="${SCRIPT_DIR}/${JOB_NAME}.sh"

    cat > "${SCRIPT}" <<SLURM
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=3-00:00:00
#SBATCH --mem=384G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --signal=USR1@120

set -e

# ── Preemption handler ──
handle_preempt() {
    echo "PREEMPT: SIGUSR1 received at \$(date -Iseconds)"
    wait "\$HARNESS_PID" 2>/dev/null || true
    echo "PREEMPT: Requeue count: \${SLURM_RESTART_COUNT:-0}"
    exit 0
}
trap handle_preempt USR1

export CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/aiongpfs/projects/eint/vendor/Time-Series-Library:\${PYTHONPATH:-}"
cd /mnt/aiongpfs/projects/eint/repo_root

INSIDER_PY="\${CONDA_PREFIX}/bin/python3"
[[ -x "\${INSIDER_PY}" ]] || { echo "FATAL: insider python missing"; exit 2; }

echo "============================================================"
echo "TSLib SOTA Benchmark | Job \${SLURM_JOB_ID}"
echo "\$(date -Iseconds) | \$(hostname) | GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Task: ${TASK} | Ablation: ${ABL} | Models: ${MODELS}"
echo "============================================================"

python3 - <<'PY'
import sys, torch
assert "insider" in sys.executable, "not insider python"
assert sys.version_info >= (3, 11), "python >= 3.11 required"
assert torch.cuda.is_available(), "GPU required"
PY

\${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:\${SLURM_JOB_NAME}"

"\${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \\
    --task ${TASK} \\
    --category tslib_sota \\
    --ablation ${ABL} \\
    --preset full \\
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/${TASK}/tslib_sota/${ABL} \\
    --seed 42 \\
    --no-verify-first \\
    --models ${MODELS} &

HARNESS_PID=\$!
wait "\$HARNESS_PID"

echo "Done: \$(date -Iseconds)"
SLURM

    chmod +x "${SCRIPT}"
    echo "Generated: ${SCRIPT}"
  done
done

echo ""
echo "=== Submitting 12 TSLib SLURM jobs ==="
for f in "${SCRIPT_DIR}"/tslib_*.sh; do
  echo "sbatch ${f}"
  sbatch "${f}"
  sleep 1
done

echo ""
echo "=== All TSLib jobs submitted ==="
squeue -u npin --format="%.10i %.20j %.8T %.10M %.6D %R" | head -30
