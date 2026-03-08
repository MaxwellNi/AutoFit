#!/usr/bin/env bash
#SBATCH --job-name=bm_st_t1_ce
#SBATCH --account=christian.fisch
#SBATCH --partition=bigmem
#SBATCH --qos=iris-bigmem-long
#SBATCH --time=1-00:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=14
#SBATCH --output=/work/projects/eint/logs/cfisch_bigmem/bm_st_t1_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/cfisch_bigmem/bm_st_t1_ce_%j.err
#SBATCH --export=ALL
#SBATCH --requeue

set -e
export CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /mnt/aiongpfs/projects/eint/repo_root
git pull --ff-only origin main || echo "WARN: git pull failed"

PY="$CONDA_PREFIX/bin/python3"
echo "=== bm_st_t1_ce | Job $SLURM_JOB_ID | $(date -Iseconds) | bigmem ==="
$PY scripts/assert_block3_execution_contract.py --entrypoint "slurm:$SLURM_JOB_NAME"

"$PY" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category statistical \
    --ablation core_edgar \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar \
    --seed 42 \
    --no-verify-first \
    --enable-global-dedup

echo "Done: $(date -Iseconds)"
