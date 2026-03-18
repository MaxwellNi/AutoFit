#!/bin/bash -l
#SBATCH --job-name=gen_text_emb
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm_logs/text_emb_%j.out
#SBATCH --error=slurm_logs/text_emb_%j.err
# -----------------------------------------------------------------
# Generate LLM text embeddings using GTE-Qwen2-1.5B-instruct
# on ULHPC Iris GPU node (V100 32GB)
#
# Model: Alibaba-NLP/gte-Qwen2-1.5B-instruct
# - LLM-native decoder embedding (Qwen2 backbone)
# - MTEB 67.16, 1536-dim, Apache-2.0
# - ~3.3GB VRAM at FP16 → fits easily on V100
#
# Expected runtime: ~30-60 min for 5.77M rows (deduped)
# -----------------------------------------------------------------

set -euo pipefail

echo "=== Text Embedding Generation ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi -L 2>/dev/null || echo 'no nvidia-smi')"
echo "Date: $(date)"
echo "Job:  ${SLURM_JOB_ID:-local}"

# Activate environment
export PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin:$PATH"
PYTHON="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"

cd /home/users/npin/repo_root

# Ensure output directory
mkdir -p runs/text_embeddings slurm_logs

# Check if model needs downloading
echo "--- Checking model cache ---"
$PYTHON -c "
from huggingface_hub import snapshot_download
import os
cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
print(f'HF cache: {cache_dir}')
try:
    path = snapshot_download('Alibaba-NLP/gte-Qwen2-1.5B-instruct', local_files_only=True)
    print(f'Model already cached at: {path}')
except Exception:
    print('Model not cached, will download...')
    path = snapshot_download('Alibaba-NLP/gte-Qwen2-1.5B-instruct')
    print(f'Model downloaded to: {path}')
"

echo "--- Starting embedding generation ---"
$PYTHON scripts/generate_text_embeddings.py \
    --model Alibaba-NLP/gte-Qwen2-1.5B-instruct \
    --batch-size 64 \
    --max-length 512 \
    --max-chars 2048 \
    --pca-dim 64 \
    --output runs/text_embeddings/ \
    --device cuda

echo "--- Done ---"
echo "Finished at: $(date)"
ls -lh runs/text_embeddings/
