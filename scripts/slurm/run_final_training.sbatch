#!/bin/bash
#SBATCH --job-name=autofit_final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -e
mkdir -p runs/slurm

source ~/.bashrc
conda activate insider
export PYTHONPATH=src

RUN_DIR="runs/auto_fit/final_${SLURM_JOB_ID}"
python scripts/run_auto_fit.py \
  --offers_path data/raw/offers \
  --output_dir "${RUN_DIR}" \
  --budget_epochs 2 5 \
  --final_epochs 50 \
  --device cuda
