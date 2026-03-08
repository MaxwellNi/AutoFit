#!/usr/bin/env bash
# Generate SLURM scripts to re-run ALL baseline categories for core_edgar + full
# This fixes the EDGAR feature mismatch (104→105 features after bab5a51)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/work/projects/eint/logs/phase7_baseline_rerun"
OUT_ROOT="runs/benchmarks/block3_20260203_225620_phase7"
ACCOUNT="christian.fisch"

mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

# Categories to re-run (all non-autofit categories)
CATEGORIES=(deep_classical foundation ml_tabular statistical irregular transformer_sota)
TASKS=(task1_outcome task2_forecast task3_risk_adjust)
ABLATIONS=(core_edgar full)

# Category-specific settings
declare -A CAT_TIME CAT_MEM CAT_GPU
CAT_TIME[deep_classical]="1-00:00:00"
CAT_TIME[foundation]="2-00:00:00"
CAT_TIME[ml_tabular]="1-00:00:00"
CAT_TIME[statistical]="1-00:00:00"
CAT_TIME[irregular]="1-00:00:00"
CAT_TIME[transformer_sota]="2-00:00:00"

CAT_MEM[deep_classical]="384G"
CAT_MEM[foundation]="384G"
CAT_MEM[ml_tabular]="256G"
CAT_MEM[statistical]="256G"
CAT_MEM[irregular]="384G"
CAT_MEM[transformer_sota]="384G"

CAT_GPU[deep_classical]="gpu:1"
CAT_GPU[foundation]="gpu:1"
CAT_GPU[ml_tabular]="gpu:1"
CAT_GPU[statistical]="gpu:1"
CAT_GPU[irregular]="gpu:1"
CAT_GPU[transformer_sota]="gpu:1"

# Short names for job IDs
declare -A CAT_SHORT
CAT_SHORT[deep_classical]="dc"
CAT_SHORT[foundation]="fn"
CAT_SHORT[ml_tabular]="ml"
CAT_SHORT[statistical]="st"
CAT_SHORT[irregular]="ir"
CAT_SHORT[transformer_sota]="ts"

declare -A TASK_SHORT
TASK_SHORT[task1_outcome]="t1"
TASK_SHORT[task2_forecast]="t2"
TASK_SHORT[task3_risk_adjust]="t3"

declare -A ABL_SHORT
ABL_SHORT[core_edgar]="ce"
ABL_SHORT[full]="fu"

count=0
for cat in "${CATEGORIES[@]}"; do
    for task in "${TASKS[@]}"; do
        for abl in "${ABLATIONS[@]}"; do
            cs="${CAT_SHORT[$cat]}"
            ts="${TASK_SHORT[$task]}"
            as="${ABL_SHORT[$abl]}"
            job_name="rerun_${cs}_${ts}_${as}"
            out_dir="${OUT_ROOT}/${task}/${cat}/${abl}"
            
            cat > "${SCRIPT_DIR}/${job_name}.sh" << SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=${CAT_TIME[$cat]}
#SBATCH --mem=${CAT_MEM[$cat]}
#SBATCH --cpus-per-task=8
#SBATCH --gres=${CAT_GPU[$cat]}
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
echo "Baseline Re-run | Job \${SLURM_JOB_ID}"
echo "\$(date -Iseconds) | \$(hostname) | GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Task: ${task} | Category: ${cat} | Ablation: ${abl}"
echo "Purpose: Fix EDGAR feature mismatch (104→105 features)"
echo "============================================================"

\${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:\${SLURM_JOB_NAME}"

"\${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \\
    --task ${task} \\
    --category ${cat} \\
    --ablation ${abl} \\
    --preset full \\
    --output-dir ${out_dir} \\
    --seed 42 \\
    --no-verify-first \\
    --enable-global-dedup &

HARNESS_PID=\$!
wait "\$HARNESS_PID"

echo "Done: \$(date -Iseconds)"
SLURM_EOF
            chmod +x "${SCRIPT_DIR}/${job_name}.sh"
            count=$((count + 1))
        done
    done
done

echo "Generated ${count} SLURM scripts in ${SCRIPT_DIR}"
echo ""
echo "To submit all (respecting MaxJobsPU=4 limit):"
echo "  cd ${SCRIPT_DIR} && bash submit_all.sh"
