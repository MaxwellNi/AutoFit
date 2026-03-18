#!/usr/bin/env bash
#SBATCH --job-name=cf_v739_t3_fu
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase13/cf_v739_t3_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase13/cf_v739_t3_fu_%j.err
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
echo "V739 Blitz | Job ${SLURM_JOB_ID} on $(hostname) | gpu/normal | $(date -Iseconds)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task3_risk_adjust --category autofit --ablation full \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task3_risk_adjust/autofit/full --seed 42 \
    --no-verify-first --models AutoFitV739
echo "Done: $(date -Iseconds)"
