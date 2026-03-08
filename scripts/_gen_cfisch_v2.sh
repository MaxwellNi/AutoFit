#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/work/projects/eint/logs/phase8_cf"
SLURM_DIR="/work/projects/eint/slurm_cf"

mkdir -p "$LOG_DIR" "$SLURM_DIR"

gen() {
    local JN=$1 TIME=$2 MEM=$3 TASK=$4 CAT=$5 ABL=$6 MODELS=$7
    local OUTDIR="runs/benchmarks/block3_20260203_225620_phase7/${TASK}/${CAT}/${ABL}"
    local MA=""
    [[ -n "$MODELS" ]] && MA="--models $MODELS"

    # Use quoted heredoc to preserve $ literally
    cat > "${SLURM_DIR}/${JN}.sh" <<'PART1'
#!/usr/bin/env bash
PART1

    # SBATCH headers (expand variables)
    cat >> "${SLURM_DIR}/${JN}.sh" <<PART2
#SBATCH --job-name=${JN}
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:volta:1
#SBATCH --output=${LOG_DIR}/${JN}_%j.out
#SBATCH --error=${LOG_DIR}/${JN}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
PART2

    # Body (quoted heredoc — no expansion)
    cat >> "${SLURM_DIR}/${JN}.sh" <<'PART3'

set -e
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
export MAMBA_EXE=/mnt/aiongpfs/users/npin/.local/bin/micromamba
eval "$($MAMBA_EXE shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"
[[ ! -x "${INSIDER_PY}" ]] && echo "FATAL: no insider python" && exit 2
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(python3 -V)"
PART3

    # Command line (expand variables for task/category/ablation)
    cat >> "${SLURM_DIR}/${JN}.sh" <<PART4

echo "Task: ${TASK} | Cat: ${CAT} | Abl: ${ABL} | Models: ${MODELS:-ALL}"
"\${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \\
    --task ${TASK} --category ${CAT} --ablation ${ABL} \\
    --preset full --output-dir ${OUTDIR} --seed 42 \\
    --no-verify-first ${MA}
echo "Done: \$(date -Iseconds)"
PART4

    chmod 755 "${SLURM_DIR}/${JN}.sh"
    echo "  GEN ${JN}.sh"
}

DC="GRU,LSTM,TCN,MLP,DilatedRNN"

for TASK in task1_outcome task2_forecast task3_risk_adjust; do
    case $TASK in
        task1_outcome) ABLS="core_only core_text core_edgar full"; TA=t1;;
        task2_forecast) ABLS="core_only core_text core_edgar full"; TA=t2;;
        task3_risk_adjust) ABLS="core_only core_edgar full"; TA=t3;;
    esac
    for ABL in $ABLS; do
        case $ABL in core_only) AA=co;; core_text) AA=ct;; core_edgar) AA=ce;; full) AA=fu;; esac
        gen "cf_dcB_${TA}_${AA}" "2-00:00:00" "256G" "$TASK" "deep_classical" "$ABL" "$DC"
    done
done

for TASK in task1_outcome task2_forecast task3_risk_adjust; do
    case $TASK in
        task1_outcome) ABLS="core_only core_text core_edgar full"; TA=t1;;
        task2_forecast) ABLS="core_only core_text core_edgar full"; TA=t2;;
        task3_risk_adjust) ABLS="core_only core_edgar full"; TA=t3;;
    esac
    for ABL in $ABLS; do
        case $ABL in core_only) AA=co;; core_text) AA=ct;; core_edgar) AA=ce;; full) AA=fu;; esac
        gen "cf_fmTF_${TA}_${AA}" "1-12:00:00" "256G" "$TASK" "foundation" "$ABL" "TimesFM"
    done
done

echo ""
echo "Total: $(ls ${SLURM_DIR}/cf_*.sh | wc -l) scripts"
