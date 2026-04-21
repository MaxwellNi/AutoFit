#!/bin/bash
#SBATCH --job-name=v740_tpfn26r_inv
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
export BLOCK3_TABPFN_VENDOR="${BLOCK3_TABPFN_VENDOR:-/home/users/npin/.cache/block3_optional_pydeps/py312_tabpfn_latest}"
export BLOCK3_TABPFN_REGRESSOR_MODEL_PATH="${BLOCK3_TABPFN_REGRESSOR_MODEL_PATH:-/home/users/npin/.cache/huggingface/hub/models--Prior-Labs--tabpfn_2_6/snapshots/6b03c759e5136df1faf03668cd7744f541d5ba31/tabpfn-v2.6-regressor-v2.6_default.ckpt}"
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/run_block3_benchmark_shard.py \
  --task task2_forecast \
  --category ml_tabular \
  --models TabPFNRegressor \
  --ablation core_edgar \
  --target-filter investors_count \
  --horizons-filter 14 \
  --preset standard \
  --max-entities 16 \
  --max-rows 1600 \
  --output-dir runs/benchmarks/block3_phase9_localclear_20260328/tabpfn2_6_regressor_investors
