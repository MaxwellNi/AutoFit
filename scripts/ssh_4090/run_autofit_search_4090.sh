#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="runs/auto_fit/local_${RUN_ID}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/run_auto_fit.py \
  --offers_path data/raw/offers \
  --output_dir "${RUN_DIR}" \
  --budget_epochs 2 5 10 \
  --final_epochs 10 \
  --device cuda
