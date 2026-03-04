#!/usr/bin/env bash
# ============================================================================
# Phase 7: New NF Baseline Models — SLURM Submission Script
# 5 models (GRU, LSTM, TCN, MLP, DilatedRNN) × 3 tasks × 4 ablations = 60 jobs
# Sharded by model to avoid excessive job count
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/work/projects/eint/logs/phase7_new_baselines"
OUTPUT_BASE="runs/benchmarks/block3_20260203_225620_phase7"
WORK_DIR="/mnt/aiongpfs/projects/eint/repo_root"
CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"

# Use gpu partition (V100, proven stable)
PARTITION="gpu"
QOS="iris-gpu-long"
WALL="3-00:00:00"
MEM="192G"
CPUS=8

mkdir -p "${LOG_DIR}"

MODELS=("GRU" "LSTM" "TCN" "MLP" "DilatedRNN")
TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_text" "core_edgar" "full")
TASK_SHORT=("t1" "t2" "t3")
ABL_SHORT=("co" "ct" "ce" "fu")

echo "============================================================"
echo "Phase 7: New NF Baselines — ${#MODELS[@]} models × ${#TASKS[@]} tasks × ${#ABLATIONS[@]} ablations = $((${#MODELS[@]} * ${#TASKS[@]} * ${#ABLATIONS[@]})) jobs"
echo "Models: ${MODELS[*]}"
echo "Partition: ${PARTITION} | QOS: ${QOS} | Wall: ${WALL}"
echo "Memory: ${MEM} | CPUs: ${CPUS}"
echo "============================================================"

SUBMITTED=0
for model in "${MODELS[@]}"; do
    model_lower=$(echo "${model}" | tr '[:upper:]' '[:lower:]')
    for ti in 0 1 2; do
        task="${TASKS[$ti]}"
        ts="${TASK_SHORT[$ti]}"
        for ai in 0 1 2 3; do
            abl="${ABLATIONS[$ai]}"
            as="${ABL_SHORT[$ai]}"
            
            JOB_NAME="nb_${model_lower}_${ts}_${as}"
            SCRIPT="${SCRIPT_DIR}/${JOB_NAME}.sh"
            OUT_DIR="${OUTPUT_BASE}/${task}/deep_classical/${abl}"
            
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
echo "New Baseline: ${model} | Job \${SLURM_JOB_ID}"
echo "\$(date -Iseconds) | \$(hostname) | GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Task: ${task} | Ablation: ${abl} | Model: ${model}"
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
    --category deep_classical \\
    --ablation ${abl} \\
    --preset full \\
    --output-dir ${OUT_DIR} \\
    --seed 42 \\
    --no-verify-first \\
    --models ${model} &

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
done

echo ""
echo "Total: ${SUBMITTED} jobs submitted"
echo "Monitor: squeue -u \$USER --name=nb_"
echo "Logs: ${LOG_DIR}/"
