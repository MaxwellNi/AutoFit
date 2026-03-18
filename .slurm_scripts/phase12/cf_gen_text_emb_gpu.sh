#!/bin/bash -l
#SBATCH --job-name=cf_gen_text_emb
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=christian.fisch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=0-06:00:00
#SBATCH --output=/work/projects/eint/logs/phase12/cf_text_emb_%j.out
#SBATCH --error=/work/projects/eint/logs/phase12/cf_text_emb_%j.err
# -----------------------------------------------------------------
# Generate LLM text embeddings via cfisch account
# Uses GTE-Qwen2-1.5B-instruct, 256G RAM (Arrow OOM fix applied)
# cfisch: NO micromamba, use direct INSIDER_PY
# -----------------------------------------------------------------
set -euo pipefail
echo "=== Text Embedding Generation (cfisch/gpu) ==="
echo "Node: $(hostname) | GPU: $(nvidia-smi -L 2>/dev/null | head -1) | Date: $(date) | Job: ${SLURM_JOB_ID:-local}"

INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root
mkdir -p runs/text_embeddings /work/projects/eint/logs/phase12

# Copy HF model cache from npin if not available for cfisch
export HF_HOME="/home/users/npin/.cache/huggingface"

$INSIDER_PY scripts/generate_text_embeddings.py \
    --model Alibaba-NLP/gte-Qwen2-1.5B-instruct \
    --batch-size 64 \
    --max-length 512 \
    --max-chars 2048 \
    --pca-dim 64 \
    --output runs/text_embeddings/ \
    --device cuda

echo "=== Done at $(date) ==="
ls -lh runs/text_embeddings/
