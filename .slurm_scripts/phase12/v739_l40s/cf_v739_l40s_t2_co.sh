#!/usr/bin/env bash
#SBATCH --job-name=cf_v739_l40s_t2_co
#SBATCH --account=christian.fisch
#SBATCH --partition=l40s
#SBATCH --qos=iris-snt
#SBATCH --time=2-00:00:00
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase12/cf_v739_l40s_t2_co_%j.out
#SBATCH --error=/work/projects/eint/logs/phase12/cf_v739_l40s_t2_co_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/home/users/npin/.cache/huggingface"
cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then echo "FATAL: python missing"; exit 2; fi
echo "Phase 12 V739 L40S | Job ${SLURM_JOB_ID} on $(hostname) | $(date -Iseconds)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category autofit --ablation core_only \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task2_forecast/autofit/core_only --seed 42 \
    --no-verify-first --models AutoFitV739
echo "Done: $(date -Iseconds)"
