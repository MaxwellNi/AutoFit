#!/bin/bash
#SBATCH --job-name=smm_np_tgdyn
#SBATCH --account=christian.fisch
#SBATCH --partition=bigmem
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=189G
#SBATCH --time=0-12:00:00
#SBATCH --output=/work/projects/eint/logs/%x_%j.out
#SBATCH --error=/work/projects/eint/logs/%x_%j.err
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -euo pipefail
umask 002

mkdir -p /work/projects/eint/logs

_requeue_handler() {
  echo "[single-model-mainline] caught USR1, requeueing ${SLURM_JOB_ID}" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap _requeue_handler USR1

REPO_ROOT=/work/projects/eint/repo_root
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
RUN_ROOT=runs/benchmarks/single_model_mainline_localclear_20260413
LABEL=mainline_track_gate_dynamic_np_20260413

export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export BLOCK3_CANONICAL_REPO_ROOT="$REPO_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

mkdir -p "$RUN_ROOT/$LABEL"

"$INSIDER_PY" scripts/analyze_mainline_investors_track_gate.py \
  --mode dynamic \
  --dynamic-horizons 14 \
  --dynamic-entity-limit 4 \
  --output-json "$RUN_ROOT/$LABEL/report.json"