#!/usr/bin/env bash
# =============================================================================
# Relaunch statistical + deep_classical shards after MSTL/NBEATS fix (6179d41)
#
# Bugs fixed:
#   1. MSTL predict(h=30) ValueError when entity series < season_length=30
#      -> silently fell back to constant mean(training_y)
#   2. NBEATS h=1 degenerate forecast (h_nf=1, val_size=1)
#      -> NF produced constant predictions for all entities at h=1
#
# Scope: 12 shards (5 statistical + 5 deep_classical completed, 2 cancelled)
# NOT affected: ml_tabular, transformer_sota, foundation, irregular
# =============================================================================
set -euo pipefail

STAMP="20260203_225620"
VARIANT="TRAIN_WIDE_FINAL"
OUTBASE="runs/benchmarks/block3_${STAMP}_iris_full"
PRESET="full"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GIT_HASH="$(git rev-parse --short HEAD)"
echo "=== MSTL/NBEATS Fix Relaunch ==="
echo "  Git: ${GIT_HASH}"
echo "  Output: ${OUTBASE}"
echo ""

# ---- Invalidate old completed MANIFESTs ----
echo "--- Invalidating old MANIFEST.json for statistical + deep_classical ---"
for task in task1_outcome task2_forecast task3_risk_adjust; do
  for cat in statistical deep_classical; do
    for ablation in core_only core_edgar; do
      mf="${OUTBASE}/${task}/${cat}/${ablation}/MANIFEST.json"
      if [ -f "$mf" ]; then
        mv "$mf" "${mf}.mstl_nbeats_invalid"
        echo "  Invalidated: ${task}/${cat}/${ablation}"
      fi
    done
  done
done
echo ""

# ---- SLURM template ----
BATCH_COMMON="--account=yves.letraon --qos=normal --time=2-00:00:00 --mem=100G -c 14"
GPU_COMMON="--account=yves.letraon --qos=normal --time=24:00:00 --mem=128G -c 14 --gres=gpu:1 -p gpu"
ACTIVATE='eval "$(/mnt/aiongpfs/users/npin/.local/bin/micromamba shell hook --shell bash)" && micromamba activate insider'

submit_shard() {
  local task="$1" cat="$2" ablation="$3" partition="$4"
  local jobname="b3mx_${task}_${cat}_${ablation}"
  local outdir="${OUTBASE}/${task}/${cat}/${ablation}"
  
  local slurm_opts
  if [ "$partition" = "gpu" ]; then
    slurm_opts="$GPU_COMMON"
  else
    slurm_opts="$BATCH_COMMON -p batch"
  fi

  sbatch $slurm_opts \
    --job-name="$jobname" \
    --output="slurm_logs/${jobname}_%j.out" \
    --error="slurm_logs/${jobname}_%j.err" \
    --wrap="${ACTIVATE} && cd ${REPO_ROOT} && python scripts/run_block3_benchmark_shard.py \
      --task ${task} --category ${cat} --ablation ${ablation} \
      --preset ${PRESET} --output-dir ${outdir}"
  echo "  Submitted: ${task}/${cat}/${ablation} (${partition})"
}

mkdir -p slurm_logs

echo "--- Submitting statistical shards (batch) ---"
for task in task1_outcome task2_forecast task3_risk_adjust; do
  for ablation in core_only core_edgar; do
    submit_shard "$task" "statistical" "$ablation" "batch"
  done
done

echo ""
echo "--- Submitting deep_classical shards (gpu) ---"
for task in task1_outcome task2_forecast task3_risk_adjust; do
  for ablation in core_only core_edgar; do
    submit_shard "$task" "deep_classical" "$ablation" "gpu"
  done
done

echo ""
echo "=== Submitted 12 shards (6 statistical + 6 deep_classical) ==="
echo "Monitor: squeue -u \$USER"
