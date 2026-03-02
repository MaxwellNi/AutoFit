#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

INDEX="${1:-${INDEX:-}}"
if [ -z "${INDEX}" ]; then
  echo "Usage: $0 <index>  (or set INDEX env var)" >&2
  exit 2
fi

STAMP_FILE="runs/backups/current_audit_stamp.txt"
if [ ! -f "${STAMP_FILE}" ]; then
  echo "ERROR: ${STAMP_FILE} not found." >&2
  exit 2
fi

STAMP="$(cat "${STAMP_FILE}")"
ORCH_DIR="runs/orchestrator/${STAMP}"

python scripts/verify_artifacts.py \
  --artifacts_json "${ORCH_DIR}/ARTIFACTS.json" \
  --require_edgar

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/run_matrix_entry.py \
  --matrix_json "${ORCH_DIR}/b11_v2_matrix.json" \
  --index "${INDEX}"
