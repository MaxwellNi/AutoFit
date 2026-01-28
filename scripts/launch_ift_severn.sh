#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(cat "${ROOT}/runs/backups/current_audit_stamp.txt")"
ORCH_DIR="${ROOT}/runs/orchestrator/${STAMP}"
LOG_DIR="${ORCH_DIR}/logs"
mkdir -p "${LOG_DIR}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insider

python "${ROOT}/scripts/verify_artifacts.py" \
  --artifacts_json "${ORCH_DIR}/ARTIFACTS.json" \
  --require_edgar |& tee "${LOG_DIR}/verify_artifacts_ift_severn.log"

# Run indexes 0-3 on 3090 using two GPUs concurrently
CUDA_VISIBLE_DEVICES=0 python "${ROOT}/scripts/run_matrix_entry.py" \
  --matrix_json "${ORCH_DIR}/b11_v2_matrix.json" --index 0 \
  |& tee "${LOG_DIR}/run_0.log" &
CUDA_VISIBLE_DEVICES=1 python "${ROOT}/scripts/run_matrix_entry.py" \
  --matrix_json "${ORCH_DIR}/b11_v2_matrix.json" --index 1 \
  |& tee "${LOG_DIR}/run_1.log" &
wait

CUDA_VISIBLE_DEVICES=0 python "${ROOT}/scripts/run_matrix_entry.py" \
  --matrix_json "${ORCH_DIR}/b11_v2_matrix.json" --index 2 \
  |& tee "${LOG_DIR}/run_2.log" &
CUDA_VISIBLE_DEVICES=1 python "${ROOT}/scripts/run_matrix_entry.py" \
  --matrix_json "${ORCH_DIR}/b11_v2_matrix.json" --index 3 \
  |& tee "${LOG_DIR}/run_3.log" &
wait
