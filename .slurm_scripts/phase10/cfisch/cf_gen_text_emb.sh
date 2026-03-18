#!/usr/bin/env bash
#SBATCH --job-name=cf_gen_text_emb
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase10/cf_gen_text_emb_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/cf_gen_text_emb_%j.err
#SBATCH --export=ALL

set -e
INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root
echo "Text Embedding Generation (cfisch) | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${INSIDER_PY} -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
"${INSIDER_PY}" scripts/generate_text_embeddings.py
echo "Done: $(date -Iseconds)"
