#!/bin/bash
###############################################################################
# Resubmit ONLY foundation + irregular shards (12 shards total)
#
# These shards produced garbage mean-fallback predictions because:
#   - Foundation: chronos / uni2ts were not installed → silent np.mean fallback
#   - Irregular: pypots was not installed → silent np.mean fallback
#   - TimesFM: REMOVED (requires Python <3.12, lingvo incompatible)
#
# Now fixed:
#   - chronos-forecasting 2.2.2 installed
#   - uni2ts 2.0.0 installed (Moirai)
#   - pypots 1.1 installed (GRU-D, SAITS)
#   - deep_models.py: hard-fail on missing packages (no silent fallback)
#   - irregular_models.py: hard-fail on missing PyPOTS
#
# Matrix: 3 tasks × 2 ablations × 2 categories = 12 GPU shards
# Cost: ~12 × 15 min = ~3 GPU-hours (V100)
###############################################################################
set -euo pipefail

STAMP="20260203_225620"
REPO="/home/users/npin/repo_root"
OUTBASE="${REPO}/runs/benchmarks/block3_${STAMP}_iris"
SLURMLOG="${REPO}/runs/slurm"
MAMBA_EXE="/mnt/aiongpfs/users/npin/.local/bin/micromamba"
PRESET="standard"
GPU_TIME="04:00:00"

mkdir -p "${SLURMLOG}"

TASKS="task1_outcome task2_forecast task3_risk_adjust"
ABLATIONS="core_only core_edgar"
CATEGORIES="foundation irregular"

TOTAL=0
JOBS=""

for TASK in ${TASKS}; do
    for ABL in ${ABLATIONS}; do
        for CAT in ${CATEGORIES}; do
            OUTDIR="${OUTBASE}/${TASK}/${CAT}/${ABL}"
            JOBNAME="b3r_${TASK}_${CAT}_${ABL}"
            JOBNAME="${JOBNAME:0:30}"

            # Remove stale results from mean-fallback run
            if [[ -d "${OUTDIR}" ]]; then
                echo "  Removing stale results: ${OUTDIR}"
                rm -rf "${OUTDIR}"
            fi
            mkdir -p "${OUTDIR}"

            SBATCH_SCRIPT=$(mktemp /tmp/b3_rerun_XXXXXX.sbatch)
            cat > "${SBATCH_SCRIPT}" << HEREDOC
#!/bin/bash -l
#SBATCH --job-name=${JOBNAME}
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time=${GPU_TIME}
#SBATCH --gres=gpu:1
#SBATCH --output=${SLURMLOG}/${JOBNAME}_%j.out
#SBATCH --error=${SLURMLOG}/${JOBNAME}_%j.err

set -euo pipefail

echo "=== Block 3 RERUN — ${CAT} ==="
echo "Task:      ${TASK}"
echo "Category:  ${CAT}"
echo "Ablation:  ${ABL}"
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname -s)"
echo "Started:   \$(date -Iseconds)"
echo "====================="

cd ${REPO}

# Verify packages are importable before wasting GPU time
python -c "
import chronos; print('chronos OK')
from uni2ts.model.moirai import MoiraiModule; print('uni2ts/Moirai OK')
from pypots.imputation import GRUD, SAITS; print('pypots OK')
import torch; print(f'torch {torch.__version__} CUDA:{torch.cuda.is_available()}')
"

python scripts/run_block3_benchmark_shard.py \\
    --task ${TASK} \\
    --category ${CAT} \\
    --ablation ${ABL} \\
    --preset ${PRESET} \\
    --output-dir "${OUTDIR}" \\
    --no-verify-first \\
    --seed 42

echo "Shard completed: \$(date -Iseconds)"
HEREDOC

            JOB_ID=$(sbatch "${SBATCH_SCRIPT}" 2>&1 | grep -oP '\d+$')
            echo "  Submitted: ${JOBNAME} → Job ${JOB_ID} (gpu, ${GPU_TIME})"
            JOBS="${JOBS} ${JOB_ID}"
            TOTAL=$((TOTAL + 1))
            rm -f "${SBATCH_SCRIPT}"
        done
    done
done

echo ""
echo "============================================================"
echo "Resubmitted ${TOTAL} shards (foundation + irregular only)"
echo "Job IDs: ${JOBS}"
echo "Monitor:  squeue -u \$USER"
echo "Output:   ${OUTBASE}"
echo "============================================================"
