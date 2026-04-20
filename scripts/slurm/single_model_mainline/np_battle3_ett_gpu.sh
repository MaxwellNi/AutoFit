#!/bin/bash
#SBATCH --job-name=smm_battle3_ett
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/work/projects/eint/logs/%x_%j.out
#SBATCH --error=/work/projects/eint/logs/%x_%j.err

set -euo pipefail
umask 002
mkdir -p /work/projects/eint/logs

REPO_ROOT=/work/projects/eint/repo_root
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3

export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

echo "=== Battle 3: ETT/Weather Universal Test ==="

"$INSIDER_PY" scripts/run_battle3_ett_adapter.py \
  --datasets ETTh1 ETTm1 \
  --pred-len 96 192 336 720 \
  --seq-len 336 \
  --output-dir runs/benchmarks/single_model_mainline_localclear_20260420/battle3_ett_weather

echo "=== Battle 3 DONE ==="
