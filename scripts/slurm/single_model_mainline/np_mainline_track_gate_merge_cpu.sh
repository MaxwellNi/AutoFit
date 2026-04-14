#!/bin/bash
#SBATCH --job-name=smm_np_tgmerge
#SBATCH --account=npin
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=/work/projects/eint/logs/%x_%j.out
#SBATCH --error=/work/projects/eint/logs/%x_%j.err

set -euo pipefail
umask 002

mkdir -p /work/projects/eint/logs

REPO_ROOT=/work/projects/eint/repo_root
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
RUN_ROOT=runs/benchmarks/single_model_mainline_localclear_20260413
OFFICIAL_LABEL=mainline_track_gate_official_cf_20260413
DYNAMIC_LABEL=mainline_track_gate_dynamic_np_20260413
LABEL=mainline_track_gate_full_np_20260413

export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export BLOCK3_CANONICAL_REPO_ROOT="$REPO_ROOT"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

mkdir -p "$RUN_ROOT/$LABEL"

"$INSIDER_PY" scripts/analyze_mainline_investors_track_gate.py \
  --mode merge \
  --official-input-json "$RUN_ROOT/$OFFICIAL_LABEL/report.json" \
  --dynamic-input-json "$RUN_ROOT/$DYNAMIC_LABEL/report.json" \
  --output-json "$RUN_ROOT/$LABEL/report.json"