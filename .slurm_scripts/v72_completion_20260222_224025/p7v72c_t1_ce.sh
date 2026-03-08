#!/usr/bin/env bash
#SBATCH --job-name=p7v72c_t1_ce
#SBATCH --account=yves.letraon
#SBATCH --partition=batch
#SBATCH --qos=iris-batch-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=112G
#SBATCH --cpus-per-task=28
#SBATCH --output=/work/projects/eint/logs/v72_completion_20260222_224025/p7v72c_t1_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/v72_completion_20260222_224025/p7v72c_t1_ce_%j.err
#SBATCH --export=ALL

set -euo pipefail
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root

python3 scripts/run_block3_benchmark_shard.py \
  --task task1_outcome \
  --category autofit \
  --ablation core_edgar \
  --models AutoFitV72 \
  --preset full \
  --output-dir runs/benchmarks/block3_20260203_225620_phase7_v72_completion_20260222_224025/task1_outcome/autofit/core_edgar \
  --seed 42 \
  --no-verify-first
