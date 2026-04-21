#!/bin/bash
#SBATCH --job-name=v740_units_clr
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=0-04:00:00
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
export BLOCK3_UNITS_REPO="$HOME/.cache/block3_optional_repos/UniTS"
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/run_block3_benchmark_shard.py \
  --task task2_forecast \
  --category transformer_sota \
  --models UniTS \
  --model-kwargs-json '{"UniTS":{"input_size":60,"patch_len":10,"stride":10,"d_model":64,"prompt_num":4,"e_layers":2,"n_heads":4,"max_epochs":5,"batch_size":16,"max_covariates":8,"max_entities":1000,"max_windows":12000}}' \
  --ablation core_edgar \
  --target-filter funding_raised_usd \
  --horizons-filter 30 \
  --preset standard \
  --max-entities 16 \
  --max-rows 1600 \
  --output-dir runs/benchmarks/block3_phase9_localclear_20260330/units_funding_h30
