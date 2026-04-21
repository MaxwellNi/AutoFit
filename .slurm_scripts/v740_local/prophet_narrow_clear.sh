#!/bin/bash
#SBATCH --job-name=v740_prop_std
#SBATCH --account=npin
#SBATCH --partition=bigmem
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=/work/projects/eint/logs/v740_local/%x_%j.out
#SBATCH --error=/work/projects/eint/logs/v740_local/%x_%j.err
#SBATCH --signal=USR1@120
#SBATCH --requeue
set -euo pipefail
umask 002
_requeue_handler() { echo "$(date -Iseconds) USR1: requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; }
trap _requeue_handler USR1
cd /home/users/npin/repo_root
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/run_block3_benchmark_shard.py \
  --task task2_forecast \
  --category statistical \
  --models Prophet \
  --model-kwargs-json '{"Prophet":{"min_history":20,"max_entities_fit":16}}' \
  --ablation core_only \
  --target-filter funding_raised_usd \
  --horizons-filter 30 \
  --preset standard \
  --max-entities 16 \
  --max-rows 1600 \
  --output-dir runs/benchmarks/block3_phase9_localclear_20260328/prophet_standard_h30
