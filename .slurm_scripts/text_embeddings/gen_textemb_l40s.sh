#!/bin/bash -l
#SBATCH --job-name=gen_textemb
#SBATCH --partition=l40s
#SBATCH --qos=iris-snt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=/work/projects/eint/logs/text_emb/textemb_l40s_%j.out
#SBATCH --error=/work/projects/eint/logs/text_emb/textemb_l40s_%j.err
# -----------------------------------------------------------------
# Generate text embeddings using GTE-Qwen2-1.5B-instruct on L40S
# Model: 1.5B params, ~3.3GB VRAM at FP16, 1536-dim → PCA to 64
# Data: 5.77M rows offers_text, deduped → ~unique texts
# Expected: 30-60 min
# -----------------------------------------------------------------
set -euo pipefail
umask 002

echo "=== Text Embedding Generation (L40S) ==="
echo "Node: $(hostname) | GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Date: $(date) | Job: ${SLURM_JOB_ID:-local}"

PYTHON="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
cd /home/users/npin/repo_root

mkdir -p runs/text_embeddings /work/projects/eint/logs/text_emb

echo "--- Verifying model cache ---"
ls -lh ~/.cache/huggingface/hub/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/snapshots/*/model-00001-of-00002.safetensors 2>/dev/null && echo "Model cached OK" || echo "Model needs download"

echo "--- Starting embedding generation ---"
$PYTHON scripts/generate_text_embeddings.py \
    --model Alibaba-NLP/gte-Qwen2-1.5B-instruct \
    --batch-size 128 \
    --max-length 512 \
    --max-chars 2048 \
    --pca-dim 64 \
    --output runs/text_embeddings/ \
    --device cuda

echo "--- Verifying output ---"
ls -lh runs/text_embeddings/
echo "Finished at: $(date)"
