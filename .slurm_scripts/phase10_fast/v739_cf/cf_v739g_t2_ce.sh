#!/usr/bin/env bash
#SBATCH --job-name=cf_v739g_t2_ce
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase10/cf_v739g_t2_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/cf_v739g_t2_ce_%j.err
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

echo "Task: task2_forecast | Abl: core_edgar | Models: AutoFitV739"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category autofit --ablation core_edgar \
    --preset full --output-dir runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_edgar --seed 42 \
    --no-verify-first --models AutoFitV739
echo "Done: $(date -Iseconds)"
