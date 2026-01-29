#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MATRIX_JSON=""
START_IDX=""
END_IDX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix)
      MATRIX_JSON="$2"
      shift 2
      ;;
    --start)
      START_IDX="$2"
      shift 2
      ;;
    --end)
      END_IDX="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -n "${MATRIX_JSON}" ]]; then
  MATRIX_JSON="$(cd "$(dirname "${MATRIX_JSON}")" && pwd)/$(basename "${MATRIX_JSON}")"
  ORCH_DIR="$(cd "$(dirname "${MATRIX_JSON}")" && pwd)"
else
  STAMP="$(cat "${ROOT}/runs/backups/current_audit_stamp.txt")"
  ORCH_DIR="${ROOT}/runs/orchestrator/${STAMP}"
  MATRIX_JSON="${ORCH_DIR}/b11_v2_matrix.json"
fi
LOG_DIR="${ORCH_DIR}/logs"
mkdir -p "${LOG_DIR}"
START_IDX="${START_IDX:-6}"
END_IDX="${END_IDX:-11}"

source /home/pni/miniforge3/etc/profile.d/conda.sh
conda activate /home/pni/miniforge3/envs/insider

python "${ROOT}/scripts/verify_artifacts.py" \
  --artifacts_json "${ORCH_DIR}/ARTIFACTS.json" \
  --require_edgar |& tee "${LOG_DIR}/verify_artifacts_4090.log"

idx="${START_IDX}"
while [[ "${idx}" -le "${END_IDX}" ]]; do
  idx2=$((idx + 1))
  CUDA_VISIBLE_DEVICES=0 python "${ROOT}/scripts/run_matrix_entry.py" \
    --matrix_json "${MATRIX_JSON}" --index "${idx}" \
    |& tee "${LOG_DIR}/run_${idx}.log" &
  if [[ "${idx2}" -le "${END_IDX}" ]]; then
    CUDA_VISIBLE_DEVICES=1 python "${ROOT}/scripts/run_matrix_entry.py" \
      --matrix_json "${MATRIX_JSON}" --index "${idx2}" \
      |& tee "${LOG_DIR}/run_${idx2}.log" &
  fi
  wait
  idx=$((idx + 2))
done
