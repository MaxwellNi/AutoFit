#!/bin/bash
#SBATCH --job-name=v743_cf_swp
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=150G
#SBATCH --time=0-12:00:00
#SBATCH --output=/work/projects/eint/logs/v743_local/%x_%j.out
#SBATCH --error=/work/projects/eint/logs/v743_local/%x_%j.err
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -euo pipefail
umask 002

mkdir -p /work/projects/eint/logs/v743_local

_requeue_handler() {
  echo "[v743-local] caught USR1, requeueing ${SLURM_JOB_ID}" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap _requeue_handler USR1

REPO_ROOT=/work/projects/eint/repo_root
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3

export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export BLOCK3_CANONICAL_REPO_ROOT="$REPO_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

RUN_ROOT=runs/benchmarks/v743_localclear_20260407/v743_keyslice_sweep_cf_20260407
mkdir -p "$RUN_ROOT"

run_case() {
  local label="$1"
  shift
  local out_json="$RUN_ROOT/${label}.json"
  local out_log="$RUN_ROOT/${label}.stdout"
  if [[ -s "$out_json" ]]; then
    echo "[v743-sweep] skip existing $label"
    return 0
  fi
  echo "[v743-sweep] start $label $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  "$INSIDER_PY" scripts/analyze_v742_financing_hypothesis.py "$@" --output-json "$out_json" > "$out_log" 2>&1
  echo "[v743-sweep] done $label $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
}

run_case investors_h1_current \
  --task task2_forecast --ablation core_edgar --target investors_count --horizon 1 \
  --max-entities 12 --max-rows 1200 --max-epochs 2

run_case investors_h7_current \
  --task task2_forecast --ablation core_edgar --target investors_count --horizon 7 \
  --max-entities 12 --max-rows 1400 --max-epochs 2

run_case binary_h14_current \
  --task task1_outcome --ablation core_edgar --target is_funded --horizon 14 \
  --max-entities 12 --max-rows 1200 --max-epochs 2

run_case binary_h14_light \
  --task task1_outcome --ablation core_edgar --target is_funded --horizon 14 \
  --max-entities 12 --max-rows 1200 --max-epochs 2 \
  --v743-consistency-strength 0.04 --v743-auxiliary-strength 0.06 --v743-scaffold-strength 0.00

run_case funding_h7_current \
  --task task1_outcome --ablation core_only --target funding_raised_usd --horizon 7 \
  --max-entities 12 --max-rows 1200 --max-epochs 2

run_case funding_h7_light \
  --task task1_outcome --ablation core_only --target funding_raised_usd --horizon 7 \
  --max-entities 12 --max-rows 1200 --max-epochs 2 \
  --v743-consistency-strength 0.04 --v743-auxiliary-strength 0.06 --v743-scaffold-strength 0.00