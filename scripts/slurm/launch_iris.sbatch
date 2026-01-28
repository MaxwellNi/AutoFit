#!/bin/bash
#SBATCH --job-name=b11_v2_iris
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --array=8-11
#SBATCH --output=/work/projects/eint/logs/b11_v2_iris_%A_%a.out

set -euo pipefail

cd /home/users/npin/repo_root
source /work/projects/eint/envs/.micromamba/etc/profile.d/micromamba.sh
micromamba activate insider

STAMP="$(cat runs/backups/current_audit_stamp.txt)"
ORCH_DIR="runs/orchestrator/${STAMP}"

python scripts/verify_artifacts.py \
  --artifacts_json "${ORCH_DIR}/ARTIFACTS.json" \
  --require_edgar

CUDA_VISIBLE_DEVICES=0 python scripts/run_matrix_entry.py \
  --matrix_json "${ORCH_DIR}/b11_v2_matrix.json" \
  --index "${SLURM_ARRAY_TASK_ID}"
