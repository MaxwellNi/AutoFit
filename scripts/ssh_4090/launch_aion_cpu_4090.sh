#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

STAMP_FILE="runs/backups/current_audit_stamp.txt"
if [ ! -f "${STAMP_FILE}" ]; then
  echo "ERROR: ${STAMP_FILE} not found." >&2
  exit 2
fi

STAMP="$(cat "${STAMP_FILE}")"
ORCH_DIR="runs/orchestrator/${STAMP}"

python scripts/collect_results.py --runs_dir runs/benchmarks --output_dir "${ORCH_DIR}/collected"
python scripts/make_paper_tables.py --bench_root runs/benchmarks --output_dir "${ORCH_DIR}/paper_tables"
