#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

MODE="${1:-monitor}"
WIDE_STAMP="${WIDE_STAMP:-20260203_225620}"
INTERVAL_SEC="${INTERVAL_SEC:-900}"
WIDE_ANALYSIS="${WIDE_ANALYSIS:-runs/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}}"

check_outputs() {
  local step4=0 step5=0 step6=0 step7=0 step8=0 step9=0 step10=0 pointer_ok=0

  if [ -f "runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_core_daily.parquet" ] && \
     [ -f "runs/offers_core_full_daily_wide_${WIDE_STAMP}/MANIFEST.json" ]; then
    step4=1
  fi
  if [ -d "runs/offers_core_full_daily_wide_${WIDE_STAMP}/snapshots_index" ]; then
    step5=1
  fi
  if [ -d "runs/edgar_feature_store_full_daily_wide_${WIDE_STAMP}/edgar_features" ]; then
    step6=1
  fi
  if [ -f "runs/multiscale_full_wide_${WIDE_STAMP}/MANIFEST.json" ]; then
    step7=1
  fi
  if [ -f "${WIDE_ANALYSIS}/column_manifest.json" ] && [ -f "${WIDE_ANALYSIS}/column_manifest.md" ]; then
    step8=1
  fi
  if [ -f "${WIDE_ANALYSIS}/raw_cardinality_coverage_wide_${WIDE_STAMP}.json" ] && \
     [ -f "${WIDE_ANALYSIS}/raw_cardinality_coverage_wide_${WIDE_STAMP}.md" ]; then
    step9=1
  fi
  if [ -f "${WIDE_ANALYSIS}/freeze_candidates.json" ] && [ -f "${WIDE_ANALYSIS}/freeze_candidates.md" ]; then
    step10=1
  fi
  if [ -f "docs/audits/FULL_SCALE_POINTER.yaml" ] && \
     grep -q "${WIDE_STAMP}" "docs/audits/FULL_SCALE_POINTER.yaml"; then
    pointer_ok=1
  fi

  echo "Steps: 4=${step4} 5=${step5} 6=${step6} 7=${step7} 8=${step8} 9=${step9} 10=${step10} pointer=${pointer_ok}"

  if [ "${step4}" -eq 1 ] && [ "${step5}" -eq 1 ] && [ "${step6}" -eq 1 ] && \
     [ "${step7}" -eq 1 ] && [ "${step8}" -eq 1 ] && [ "${step9}" -eq 1 ] && \
     [ "${step10}" -eq 1 ] && [ "${pointer_ok}" -eq 1 ]; then
    return 0
  fi
  return 1
}

if [ "${MODE}" = "--run" ] || [ "${MODE}" = "run" ]; then
  echo "Running pipeline locally (WIDE_STAMP=${WIDE_STAMP})"
  bash scripts/run_wide_freeze_from_daily.sh
  check_outputs
  exit $?
fi

echo "Monitoring locally every ${INTERVAL_SEC}s (WIDE_STAMP=${WIDE_STAMP})"
while true; do
  if check_outputs; then
    echo "All steps complete."
    exit 0
  fi
  sleep "${INTERVAL_SEC}"
done
