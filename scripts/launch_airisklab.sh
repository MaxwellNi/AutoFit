#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(cat "${ROOT}/runs/backups/current_audit_stamp.txt")"
ORCH_DIR="${ROOT}/runs/orchestrator/${STAMP}"
LOG_DIR="${ORCH_DIR}/logs"
mkdir -p "${LOG_DIR}"

source /home/pni/miniforge3/etc/profile.d/conda.sh
conda activate /home/pni/miniforge3/envs/insider

python "${ROOT}/scripts/verify_artifacts.py" \
  --artifacts_json "${ORCH_DIR}/ARTIFACTS.json" \
  --require_edgar |& tee "${LOG_DIR}/verify_artifacts_airisklab.log"

# Run indexes 4-7 on 4090 using two GPUs concurrently
CUDA_VISIBLE_DEVICES=0 python "${ROOT}/scripts/run_matrix_entry.py" \
  --matrix_json "${ORCH_DIR}/b11_v2_matrix.json" --index 4 \
  |& tee "${LOG_DIR}/run_4.log" &
CUDA_VISIBLE_DEVICES=1 python "${ROOT}/scripts/run_matrix_entry.py" \
  --matrix_json "${ORCH_DIR}/b11_v2_matrix.json" --index 5 \
  |& tee "${LOG_DIR}/run_5.log" &
wait

CUDA_VISIBLE_DEVICES=0 python "${ROOT}/scripts/run_matrix_entry.py" \
  --matrix_json "${ORCH_DIR}/b11_v2_matrix.json" --index 6 \
  |& tee "${LOG_DIR}/run_6.log" &
CUDA_VISIBLE_DEVICES=1 python "${ROOT}/scripts/run_matrix_entry.py" \
  --matrix_json "${ORCH_DIR}/b11_v2_matrix.json" --index 7 \
  |& tee "${LOG_DIR}/run_7.log" &
wait
