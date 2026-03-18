#!/usr/bin/env bash
#SBATCH --job-name=cf_v735_t2_ct
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/cfisch_batch2/cf_v735_t2_ct_%j.out
#SBATCH --error=/work/projects/eint/logs/cfisch_batch2/cf_v735_t2_ct_%j.err
#SBATCH --export=ALL
#SBATCH --requeue

set -e
export CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /mnt/aiongpfs/projects/eint/repo_root
git pull --ff-only origin main || echo "WARN: git pull failed"

PY="$CONDA_PREFIX/bin/python3"
echo "=== cf_v735_t2_ct | Job $SLURM_JOB_ID | $(date -Iseconds) ==="
$PY scripts/assert_block3_execution_contract.py --entrypoint "slurm:$SLURM_JOB_NAME"

"$PY" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast \
    --category autofit \
    --ablation core_text \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/autofit/core_text \
    --seed 42 \
    --no-verify-first \
    --models AutoFitV735 \
    --enable-global-dedup

echo "Done: $(date -Iseconds)"
