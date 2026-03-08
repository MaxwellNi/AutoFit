#!/usr/bin/env bash
# Generate V735 SLURM scripts — 12 jobs (3 tasks × 4 ablations)
set -e

SCRIPT_DIR="/home/users/npin/repo_root/.slurm_scripts/phase7_v735"
LOG_DIR="/work/projects/eint/logs/phase7_v735"
OUT_ROOT="runs/benchmarks/block3_20260203_225620_phase7"
ACCOUNT="christian.fisch"

# Find the last baseline re-run job for each ablation to chain V735 after it
# V735 core_only/core_text can run immediately (no baseline re-run needed)
# V735 core_edgar/full should run AFTER baseline re-runs complete

for task_full in task1_outcome task2_forecast task3_risk_adjust; do
    case $task_full in
        task1_outcome)     ts="t1" ;;
        task2_forecast)    ts="t2" ;;
        task3_risk_adjust) ts="t3" ;;
    esac

    for abl in core_only core_text core_edgar full; do
        case $abl in
            core_only)  as="co" ;;
            core_text)  as="ct" ;;
            core_edgar) as="ce" ;;
            full)       as="fu" ;;
        esac

        job_name="v735_${ts}_${as}"
        out_dir="${OUT_ROOT}/${task_full}/autofit/${abl}"

        cat > "${SCRIPT_DIR}/${job_name}.sh" << SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=384G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --signal=USR1@120

set -e

handle_preempt() {
    echo "PREEMPT: SIGUSR1 received at \$(date -Iseconds)"
    wait "\$HARNESS_PID" 2>/dev/null || true
    exit 0
}
trap handle_preempt USR1

export CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
cd /mnt/aiongpfs/projects/eint/repo_root

INSIDER_PY="\${CONDA_PREFIX}/bin/python3"
[[ -x "\${INSIDER_PY}" ]] || { echo "FATAL: insider python missing"; exit 2; }

echo "============================================================"
echo "V7.3.5 Exact Oracle Champion | Job \${SLURM_JOB_ID}"
echo "\$(date -Iseconds) | \$(hostname) | GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Task: ${task_full} | Ablation: ${abl} | Model: AutoFitV735"
echo "============================================================"

\${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:\${SLURM_JOB_NAME}"

"\${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \\
    --task ${task_full} \\
    --category autofit \\
    --ablation ${abl} \\
    --preset full \\
    --output-dir ${out_dir} \\
    --seed 42 \\
    --no-verify-first \\
    --models AutoFitV735 \\
    --enable-global-dedup &

HARNESS_PID=\$!
wait "\$HARNESS_PID"

echo "Done: \$(date -Iseconds)"
SLURM_EOF
        chmod +x "${SCRIPT_DIR}/${job_name}.sh"
    done
done

echo "Generated 12 V735 SLURM scripts in ${SCRIPT_DIR}"
ls -1 "${SCRIPT_DIR}"/v735_*.sh
