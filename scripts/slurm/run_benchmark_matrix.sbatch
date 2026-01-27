#!/bin/bash
#SBATCH --job-name=autofit_benchmark
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

python scripts/run_full_benchmark.py \
  --offers_path data/raw/offers \
  --device cuda
