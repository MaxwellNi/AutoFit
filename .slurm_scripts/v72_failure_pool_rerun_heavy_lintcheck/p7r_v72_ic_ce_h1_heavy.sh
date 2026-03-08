#!/usr/bin/env bash
#SBATCH --job-name=p7r_v72_ic_ce_h1_heavy
#SBATCH --account=yves.letraon
#SBATCH --partition=batch
#SBATCH --qos=iris-batch-long
#SBATCH --time=4-00:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=24
#SBATCH --output=/work/projects/eint/logs/v72_failure_pool_rerun_heavy_lintcheck/p7r_v72_ic_ce_h1_heavy_%j.out
#SBATCH --error=/work/projects/eint/logs/v72_failure_pool_rerun_heavy_lintcheck/p7r_v72_ic_ce_h1_heavy_%j.err
#SBATCH --export=ALL
#SBATCH --signal=B:USR1@120

set -euo pipefail

export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root


echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Failure pool rerun heavy | h=1"
echo "Output: runs/benchmarks/block3_20260203_225620_phase7_v72_failure_pool_rerun_heavy_lintcheck/task1_outcome/autofit/core_edgar/h1_rerun_heavy"
echo "============================================================"

python3 scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category autofit \
    --ablation core_edgar \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7_v72_failure_pool_rerun_heavy_lintcheck/task1_outcome/autofit/core_edgar/h1_rerun_heavy \
    --seed 42 \
    --no-verify-first \
    --models AutoFitV71,AutoFitV72 \
    --target-filter investors_count \
    --horizons-filter 1
