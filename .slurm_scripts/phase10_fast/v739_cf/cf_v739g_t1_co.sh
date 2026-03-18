#!/usr/bin/env bash
#SBATCH --job-name=cf_v739g_t1_co
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase10/cf_v739g_t1_co_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/cf_v739g_t1_co_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/home/users/npin/.cache/huggingface"
cd /work/projects/eint/repo_root
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"; exit 2
fi
echo "V739 cfisch iris-gpu-long | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

echo "Task: task1_outcome | Abl: core_only | Models: AutoFitV739"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category autofit --ablation core_only \
    --preset full --output-dir runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_only --seed 42 \
    --no-verify-first --models AutoFitV739
echo "Done: $(date -Iseconds)"
