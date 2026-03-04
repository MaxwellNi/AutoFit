#!/usr/bin/env bash
# ============================================================================
# V7.3.3 NF-Native Adaptive Champion — SLURM Submission Script
# Generates and submits 12 jobs (3 tasks × 4 ablations)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/work/projects/eint/logs/phase7_v733"
OUTPUT_BASE="runs/benchmarks/block3_20260203_225620_phase7"
WORK_DIR="/mnt/aiongpfs/projects/eint/repo_root"
CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
MODEL_NAME="AutoFitV733"

# Use gpu partition (V100, proven stable)
PARTITION="gpu"
QOS="iris-gpu-long"
WALL="3-00:00:00"
MEM="192G"
CPUS=8

mkdir -p "${LOG_DIR}"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_text" "core_edgar" "full")
TASK_SHORT=("t1" "t2" "t3")
ABL_SHORT=("co" "ct" "ce" "fu")

echo "============================================================"
echo "V7.3.3 NF-Native Adaptive Champion — Submitting 12 jobs"
echo "Partition: ${PARTITION} | QOS: ${QOS} | Wall: ${WALL}"
echo "Memory: ${MEM} | CPUs: ${CPUS}"
echo "============================================================"

SUBMITTED=0
for ti in 0 1 2; do
    task="${TASKS[$ti]}"
    ts="${TASK_SHORT[$ti]}"
    for ai in 0 1 2 3; do
        abl="${ABLATIONS[$ai]}"
        as="${ABL_SHORT[$ai]}"
        
        JOB_NAME="v733_${ts}_${as}"
        SCRIPT="${SCRIPT_DIR}/v733_${ts}_${as}.sh"
        OUT_DIR="${OUTPUT_BASE}/${task}/autofit/${abl}"
        
        cat > "${SCRIPT}" << SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=christian.fisch
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --time=${WALL}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
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

export CONDA_PREFIX="${CONDA_PREFIX}"
export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
cd ${WORK_DIR}

INSIDER_PY="\${CONDA_PREFIX}/bin/python3"
[[ -x "\${INSIDER_PY}" ]] || { echo "FATAL: insider python missing"; exit 2; }

echo "============================================================"
echo "V7.3.3 NF-Native Adaptive Champion | Job \${SLURM_JOB_ID}"
echo "\$(date -Iseconds) | \$(hostname) | GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Task: ${task} | Ablation: ${abl} | Model: ${MODEL_NAME}"
echo "============================================================"

python3 - <<'PY'
import sys, torch
assert "insider" in sys.executable, "not insider python"
assert sys.version_info >= (3, 11), "python >= 3.11 required"
assert torch.cuda.is_available(), "GPU required"
PY

\${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:\${SLURM_JOB_NAME}"

"\${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \\
    --task ${task} \\
    --category autofit \\
    --ablation ${abl} \\
    --preset full \\
    --output-dir ${OUT_DIR} \\
    --seed 42 \\
    --no-verify-first \\
    --models ${MODEL_NAME} &

HARNESS_PID=\$!
wait "\$HARNESS_PID"

echo "Done: \$(date -Iseconds)"
SLURM_EOF
        
        chmod +x "${SCRIPT}"
        
        # Submit
        if [[ "${1:-}" == "--dry-run" ]]; then
            echo "  [DRY-RUN] ${JOB_NAME}: ${SCRIPT}"
        else
            JOB_ID=$(sbatch "${SCRIPT}" 2>&1 | grep -oP '\d+')
            echo "  SUBMITTED: ${JOB_NAME} → Job ${JOB_ID}"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
done

echo ""
echo "Total: ${SUBMITTED} jobs submitted"
echo "Monitor: squeue -u \$USER --name=v733"
echo "Logs: ${LOG_DIR}/"
