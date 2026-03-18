#!/usr/bin/env bash
#SBATCH --job-name=gf_fm_t1_cos2
#SBATCH --account=npin
#SBATCH --partition=l40s
#SBATCH --qos=iris-snt
#SBATCH --time=2-00:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase13/gf_fm_t1_cos2_%j.out
#SBATCH --error=/work/projects/eint/logs/phase13/gf_fm_t1_cos2_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -e
umask 002
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/home/users/npin/.cache/huggingface"
cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then echo "FATAL: python missing"; exit 2; fi
echo "GapFill | Job ${SLURM_JOB_ID} on $(hostname) | l40s/iris-snt | $(date -Iseconds)"
echo "Models: Chronos2,TTM"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category foundation --ablation core_only_seed2 \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/foundation/core_only_seed2 --seed 42 \
    --no-verify-first --models Chronos2,TTM
echo "Done: $(date -Iseconds)"
