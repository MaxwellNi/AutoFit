#!/usr/bin/env bash
#SBATCH --job-name=cf_v739r2_t1_ct
#SBATCH --account=christian.fisch
#SBATCH --partition=l40s
#SBATCH --qos=iris-snt
#SBATCH --time=2-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase10/cf_v739r2_t1_ct_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/cf_v739r2_t1_ct_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
umask 002
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/home/users/npin/.cache/huggingface"
cd /work/projects/eint/repo_root

echo "V739 cfisch RETRY2 (300G) | Job ${SLURM_JOB_ID} on $(hostname) | l40s/iris-snt"
echo "Task: task1_outcome | Abl: core_text | Models: AutoFitV739"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
  --task task1_outcome --category autofit --ablation core_text \
  --models AutoFitV739 --preset full \
  --output-dir runs/benchmarks/block3_phase10/v739
