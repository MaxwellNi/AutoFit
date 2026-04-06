#!/bin/bash

set -euo pipefail
umask 002

_requeue_handler() {
  echo "[v740-priority] caught USR1, requeueing ${SLURM_JOB_ID}" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap _requeue_handler USR1

export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
REPO_ROOT=/home/users/npin/repo_root
VARIANT="${1:?variant required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT environment variable required}"

mkdir -p "$RUN_ROOT"
cd "$REPO_ROOT"

run_shared112() {
  local label="$1"
  shift
  local out_dir="$RUN_ROOT/$label"
  local summary_md="$RUN_ROOT/${label}.md"
  local surface_json="$RUN_ROOT/${label}_surface.json"

  echo "[v740-priority] variant=$VARIANT label=$label start $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  "$INSIDER_PY" scripts/run_v740_shared112_champion_loop.py \
    --models v740_alpha,incumbent \
    --profile quick \
    --output-dir "$out_dir" \
    --summary-md "$summary_md" \
    --surface-json "$surface_json" \
    --skip-existing \
    "$@"
  echo "[v740-priority] variant=$VARIANT label=$label done $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
}

run_multi_variant() {
  local label="$1"
  shift
  local job_root="$RUN_ROOT/$label"

  mkdir -p "$job_root"

  run_one() {
    local variant_name="$1"
    shift
    local out_dir="$job_root/$variant_name"
    local summary_md="$job_root/${variant_name}.md"
    local surface_json="$job_root/${variant_name}_surface.json"

    echo "[v740-priority] label=$label variant=$variant_name start $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    "$INSIDER_PY" scripts/run_v740_shared112_champion_loop.py \
      --models v740_alpha,incumbent \
      --profile quick \
      --output-dir "$out_dir" \
      --summary-md "$summary_md" \
      --surface-json "$surface_json" \
      --skip-existing \
      "$@"
    echo "[v740-priority] label=$label variant=$variant_name done $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  }

  case "$label" in
    v740_investors_h1_count_gate_20260405)
      run_one baseline --target investors_count --horizon 1
      run_one nojump --target investors_count --horizon 1 --disable-count-jump
      run_one noanchor --target investors_count --horizon 1 --disable-count-anchor
      run_one anchor_high_nojump --target investors_count --horizon 1 --disable-count-jump --count-anchor-strength 1.20
      run_one route_off_nojump --target investors_count --horizon 1 --disable-target-routing --disable-count-jump
      ;;
    v740_investors_h1_count_gate_v2_20260405)
      run_one baseline_v2 --target investors_count --horizon 1
      run_one no_sparsity_gate --target investors_count --horizon 1 --disable-count-sparsity-gate
      run_one high_sparsity --target investors_count --horizon 1 --count-sparsity-gate-strength 0.90
      run_one high_sparsity_nojump --target investors_count --horizon 1 --disable-count-jump --count-anchor-strength 1.10 --count-sparsity-gate-strength 0.90
      run_one route_off_high_sparsity --target investors_count --horizon 1 --disable-target-routing --disable-count-jump --count-sparsity-gate-strength 0.90
      ;;
    v740_investors_task2_h1_occurrence_probe_20260405)
      run_one baseline_task2_occ --task task2_forecast --target investors_count --horizon 1
      run_one no_sparsity_gate --task task2_forecast --target investors_count --horizon 1 --disable-count-sparsity-gate
      run_one route_off_nojump --task task2_forecast --target investors_count --horizon 1 --disable-target-routing --disable-count-jump
      ;;
    *)
      echo "Unknown multi-variant label: $label" >&2
      exit 1
      ;;
  esac
}

case "$VARIANT" in
  112_inv)
    run_shared112 \
      v740_shared112_investors_routed_loop_20260405 \
      --target investors_count \
      --target-route-experts 3 \
      --count-anchor-strength 0.70 \
      --count-jump-strength 0.30
    ;;
  112_bin)
    run_shared112 \
      v740_shared112_binary_routed_guard_20260405 \
      --target is_funded \
      --target-route-experts 3 \
      --count-anchor-strength 0.70 \
      --count-jump-strength 0.30
    ;;
  112_invh1)
    run_shared112 \
      v740_shared112_investors_routed_h1_probe_20260405 \
      --target investors_count \
      --horizon 1 \
      --max-cells 12 \
      --target-route-experts 3 \
      --count-anchor-strength 0.70 \
      --count-jump-strength 0.30
    ;;
  inv_gate)
    run_multi_variant v740_investors_h1_count_gate_20260405
    ;;
  inv_g2)
    run_multi_variant v740_investors_h1_count_gate_v2_20260405
    ;;
  inv_t2occ)
    run_multi_variant v740_investors_task2_h1_occurrence_probe_20260405
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 1
    ;;
esac