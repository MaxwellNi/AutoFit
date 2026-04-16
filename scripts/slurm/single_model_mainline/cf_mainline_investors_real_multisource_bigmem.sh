#!/bin/bash
#SBATCH --job-name=smm_cf_invrms
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

REPO_ROOT=${REPO_ROOT:-/work/projects/eint/repo_root}
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
RUN_ROOT=${RUN_ROOT:-runs/benchmarks/single_model_mainline_localclear_20260415}
LABEL=${LABEL:-mainline_investors_real_multisource_cf_bigmem_20260415}
ENTITY_SELECTION=${ENTITY_SELECTION:-dynamic_test}
ENTITY_LIMIT=${ENTITY_LIMIT:-4}
MAX_ROWS_PER_ABLATION=${MAX_ROWS_PER_ABLATION:-1000}
EXTRA_ARGS=${EXTRA_ARGS:-}

export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export BLOCK3_CANONICAL_REPO_ROOT="$REPO_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

mkdir -p "$RUN_ROOT/$LABEL"

CMD=(
  "$INSIDER_PY"
  scripts/analyze_mainline_investors_real_multisource_surface.py
  --entity-selection "$ENTITY_SELECTION"
  --entity-limit "$ENTITY_LIMIT"
  --max-rows-per-ablation "$MAX_ROWS_PER_ABLATION"
  --output-json "$RUN_ROOT/$LABEL/report.json"
)

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARGS_ARR[@]}")
fi

CMD+=("$@")

"${CMD[@]}"