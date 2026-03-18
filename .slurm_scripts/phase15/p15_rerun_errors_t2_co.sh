#!/usr/bin/env bash
#SBATCH --job-name=p15_fix_t2_co
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase15/p15_fix_t2_co_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/p15_fix_t2_co_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -e
umask 002
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"

echo "Phase 15 RERUN (fixed models) | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Git: $(git rev-parse --short HEAD)"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category tslib_sota --ablation core_only \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task2_forecast/tslib_sota/core_only --seed 42 \
    --no-verify-first --models DeformableTST,DUET,FilterTS,PathFormer,SEMPO
echo "Done: $(date -Iseconds)"
