#!/bin/bash
# Submit with:
#   sbatch jobs/public_pack_full_family.sh
# Optional overrides:
#   PUBLIC_PACK_MODEL=LightGTS sbatch jobs/public_pack_full_family.sh
#   PUBLIC_PACK_PRESET=first_wave_forecasting PUBLIC_PACK_MODEL= sbatch jobs/public_pack_full_family.sh

#SBATCH --job-name=pubpack_full
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-6
#SBATCH --output=/home/users/npin/repo_root/runs/slurm/pubpack_full_%A_%a.out
#SBATCH --error=/home/users/npin/repo_root/runs/slurm/pubpack_full_%A_%a.err

set -euo pipefail

FAMILIES=(ecl ett exchange ili solar traffic weather)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FAMILY="${FAMILIES[$TASK_ID]}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MODEL="${PUBLIC_PACK_MODEL:-SAMformer}"
PRESET="${PUBLIC_PACK_PRESET:-first_wave_entrants}"
MAX_COVARIATES="${PUBLIC_PACK_MAX_COVARIATES:-8}"
N_BOOTSTRAP="${PUBLIC_PACK_N_BOOTSTRAP:-0}"
PYTHON_BIN="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"

cd /home/users/npin/repo_root
mkdir -p runs/slurm runs/audits

OUT_DIR="runs/audits/public_pack_full_${MODEL:-preset}_${FAMILY}_${STAMP}_${SLURM_JOB_ID:-local}"

echo "public-pack full family shard"
echo "family=${FAMILY}"
echo "model=${MODEL:-<preset:$PRESET>}"
echo "preset=${PRESET}"
echo "output=${OUT_DIR}"
echo "started=$(date -Iseconds)"

ARGS=(
  scripts/run_public_pack_first_wave.py
  --pack long_horizon_core
  --family "${FAMILY}"
  --preset "${PRESET}"
  --max-covariates "${MAX_COVARIATES}"
  --n-bootstrap "${N_BOOTSTRAP}"
  --output-dir "${OUT_DIR}"
)

if [[ -n "${MODEL}" ]]; then
  ARGS+=(--model "${MODEL}")
fi

"${PYTHON_BIN}" "${ARGS[@]}"

echo "finished=$(date -Iseconds)"