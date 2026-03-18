#!/bin/bash -l
#SBATCH --job-name=cf_gen_text_emb_l40s
#SBATCH --partition=l40s
#SBATCH --qos=iris-snt
#SBATCH --account=christian.fisch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=0-06:00:00
#SBATCH --output=/work/projects/eint/logs/phase12/cf_text_emb_l40s_%j.out
#SBATCH --error=/work/projects/eint/logs/phase12/cf_text_emb_l40s_%j.err
# -----------------------------------------------------------------
# Text embeddings on L40S via cfisch (iris-snt QOS)
# -----------------------------------------------------------------
set -euo pipefail
echo "=== Text Embedding Generation (cfisch/l40s) ==="
echo "Node: $(hostname) | GPU: $(nvidia-smi -L 2>/dev/null | head -1) | Date: $(date) | Job: ${SLURM_JOB_ID:-local}"

INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/home/users/npin/.cache/huggingface"
cd /work/projects/eint/repo_root
mkdir -p runs/text_embeddings /work/projects/eint/logs/phase12

$INSIDER_PY scripts/generate_text_embeddings.py \
    --model Alibaba-NLP/gte-Qwen2-1.5B-instruct \
    --batch-size 128 \
    --max-length 512 \
    --max-chars 2048 \
    --pca-dim 64 \
    --output runs/text_embeddings/ \
    --device cuda

echo "=== Done at $(date) ==="
ls -lh runs/text_embeddings/
