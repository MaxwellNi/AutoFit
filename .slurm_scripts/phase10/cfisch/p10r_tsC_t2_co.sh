#!/usr/bin/env bash
#SBATCH --job-name=p10r_tsC_t2_co
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase10/p10r_tsC_t2_co_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/p10r_tsC_t2_co_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --dependency=afterany:5217484:5217485:5217486:5217488:5217489:5217490:5217492:5217493:5217550:5217551:5217552:5217554:5217555:5217556:5217558:5217559

set -e
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"; exit 2
fi
echo "============================================================"
echo "tsC Re-run (data-loss recovery) | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(python3 -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

echo "Task: task2_forecast | Abl: core_only | Models: ETSformer,LightTS,Pyraformer,Reformer"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category tslib_sota --ablation core_only \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task2_forecast/tslib_sota/core_only --seed 42 \
    --no-verify-first --models ETSformer,LightTS,Pyraformer,Reformer
echo "Done: $(date -Iseconds)"
