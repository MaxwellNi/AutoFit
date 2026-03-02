#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-4090}"
REMOTE_USER="${REMOTE_USER:-pni}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/pni/projects/repo_root}"
LOCAL_ROOT="${LOCAL_ROOT:-$(pwd)}"
LOCAL_RUNS="${LOCAL_RUNS:-${LOCAL_ROOT}/runs}"
WIDE_STAMP="${WIDE_STAMP:-20260203_225620}"
SSH_PORT="${SSH_PORT:-22}"

RSYNC_OPTS="${RSYNC_OPTS:--av --partial --progress}"

sync_dir() {
  local src="$1"
  local dest="$2"
  if [ -e "${src}" ]; then
    echo "Sync: ${src} -> ${REMOTE}:${dest}"
    rsync ${RSYNC_OPTS} -e "ssh -p ${SSH_PORT}" "${src%/}/" "${REMOTE}:${dest%/}/"
  else
    echo "Skip (missing): ${src}"
  fi
}

sync_file() {
  local src="$1"
  local dest="$2"
  if [ -f "${src}" ]; then
    echo "Sync: ${src} -> ${REMOTE}:${dest}"
    rsync ${RSYNC_OPTS} -e "ssh -p ${SSH_PORT}" "${src}" "${REMOTE}:${dest}"
  else
    echo "Skip (missing): ${src}"
  fi
}

ssh -p "${SSH_PORT}" "${REMOTE}" "mkdir -p \"${REMOTE_ROOT}/runs\" \"${REMOTE_ROOT}/docs/audits\" \"${REMOTE_ROOT}/configs\""

# Core outputs
sync_dir "${LOCAL_RUNS}/offers_core_full_snapshot_wide_${WIDE_STAMP}" "${REMOTE_ROOT}/runs/offers_core_full_snapshot_wide_${WIDE_STAMP}"
sync_dir "${LOCAL_RUNS}/offers_core_full_daily_wide_${WIDE_STAMP}" "${REMOTE_ROOT}/runs/offers_core_full_daily_wide_${WIDE_STAMP}"
sync_dir "${LOCAL_RUNS}/edgar_feature_store_full_daily_wide_${WIDE_STAMP}" "${REMOTE_ROOT}/runs/edgar_feature_store_full_daily_wide_${WIDE_STAMP}"
sync_dir "${LOCAL_RUNS}/multiscale_full_wide_${WIDE_STAMP}" "${REMOTE_ROOT}/runs/multiscale_full_wide_${WIDE_STAMP}"

# Orchestrator analysis
sync_dir "${LOCAL_RUNS}/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}" "${REMOTE_ROOT}/runs/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}"

# Inputs required by audit steps
sync_dir "${LOCAL_RUNS}/selections/b11_v2_canonical" "${REMOTE_ROOT}/runs/selections/b11_v2_canonical"
sync_dir "${LOCAL_RUNS}/offers_text_v1_20260129_073037_full" "${REMOTE_ROOT}/runs/offers_text_v1_20260129_073037_full"

# Configs and audit docs
sync_file "configs/column_contract_wide.yaml" "${REMOTE_ROOT}/configs/column_contract_wide.yaml"
sync_dir "docs/audits" "${REMOTE_ROOT}/docs/audits"
