#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-pni@4090}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/pni/projects/repo_root}"
WIDE_STAMP="${WIDE_STAMP:-20260203_225620}"

RSYNC_OPTS="${RSYNC_OPTS:--av --partial --progress}"

run_dir="/work/projects/eint/runs"

sync_dir() {
  local src="$1"
  local dest="$2"
  if [ -e "${src}" ]; then
    echo "Sync: ${src} -> ${REMOTE}:${dest}"
    rsync ${RSYNC_OPTS} "${src%/}/" "${REMOTE}:${dest%/}/"
  else
    echo "Skip (missing): ${src}"
  fi
}

sync_file() {
  local src="$1"
  local dest="$2"
  if [ -f "${src}" ]; then
    echo "Sync: ${src} -> ${REMOTE}:${dest}"
    rsync ${RSYNC_OPTS} "${src}" "${REMOTE}:${dest}"
  else
    echo "Skip (missing): ${src}"
  fi
}

ssh "${REMOTE}" "mkdir -p \"${REMOTE_ROOT}/runs\" \"${REMOTE_ROOT}/docs/audits\" \"${REMOTE_ROOT}/configs\""

# Core outputs
sync_dir "${run_dir}/offers_core_full_snapshot_wide_${WIDE_STAMP}" "${REMOTE_ROOT}/runs/offers_core_full_snapshot_wide_${WIDE_STAMP}"
sync_dir "${run_dir}/offers_core_full_daily_wide_${WIDE_STAMP}" "${REMOTE_ROOT}/runs/offers_core_full_daily_wide_${WIDE_STAMP}"
sync_dir "${run_dir}/edgar_feature_store_full_daily_wide_${WIDE_STAMP}" "${REMOTE_ROOT}/runs/edgar_feature_store_full_daily_wide_${WIDE_STAMP}"
sync_dir "${run_dir}/multiscale_full_wide_${WIDE_STAMP}" "${REMOTE_ROOT}/runs/multiscale_full_wide_${WIDE_STAMP}"

# Orchestrator analysis
sync_dir "${run_dir}/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}" "${REMOTE_ROOT}/runs/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}"

# Inputs required by audit steps
sync_dir "${run_dir}/selections/b11_v2_canonical" "${REMOTE_ROOT}/runs/selections/b11_v2_canonical"
sync_dir "${run_dir}/offers_text_v1_20260129_073037_full" "${REMOTE_ROOT}/runs/offers_text_v1_20260129_073037_full"

# Configs and audit docs
sync_file "configs/column_contract_wide.yaml" "${REMOTE_ROOT}/configs/column_contract_wide.yaml"
sync_dir "docs/audits" "${REMOTE_ROOT}/docs/audits"
