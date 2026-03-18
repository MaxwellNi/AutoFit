#!/usr/bin/env bash
#SBATCH --job-name=cf_gf_fm_t1_cos2
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase14/cf_gf_fm_t1_cos2_%j.out
#SBATCH --error=/work/projects/eint/logs/phase14/cf_gf_fm_t1_cos2_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -e
umask 002
INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
# HF_HOME uses default (cfisch own ~/.cache/huggingface)
cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then echo "FATAL: python missing"; exit 2; fi
echo "GapFill FM | Job ${SLURM_JOB_ID} on $(hostname) | gpu/normal | $(date -Iseconds)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Models: Chronos2,TTM"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category foundation --ablation core_text \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/foundation/core_only_seed2 --seed 42 \
    --no-verify-first --models Chronos2,TTM
echo "Done: $(date -Iseconds)"
