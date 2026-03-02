#!/usr/bin/env bash
set -euo pipefail

MAC_HOST="${MAC_HOST:?Set MAC_HOST (e.g. my-mac.local or IP)}"
MAC_USER="${MAC_USER:?Set MAC_USER (e.g. alice)}"
MAC="${MAC_USER}@${MAC_HOST}"
MAC_ROOT="${MAC_ROOT:-/Users/${MAC_USER}/projects/repo_root}"
WIDE_STAMP="${WIDE_STAMP:-20260203_225620}"
SSH_PORT="${SSH_PORT:-22}"

RSYNC_OPTS="${RSYNC_OPTS:--av --partial --progress}"

RUNS_ROOT="/work/projects/eint/runs"

sync_dir() {
  local src="$1"
  local dest="$2"
  if [ -e "${src}" ]; then
    echo "Sync: ${src} -> ${MAC}:${dest}"
    rsync ${RSYNC_OPTS} -e "ssh -p ${SSH_PORT}" "${src%/}/" "${MAC}:${dest%/}/"
  else
    echo "Skip (missing): ${src}"
  fi
}

sync_file() {
  local src="$1"
  local dest="$2"
  if [ -f "${src}" ]; then
    echo "Sync: ${src} -> ${MAC}:${dest}"
    rsync ${RSYNC_OPTS} -e "ssh -p ${SSH_PORT}" "${src}" "${MAC}:${dest}"
  else
    echo "Skip (missing): ${src}"
  fi
}

ssh -p "${SSH_PORT}" "${MAC}" "mkdir -p \"${MAC_ROOT}/runs\" \"${MAC_ROOT}/docs/audits\" \"${MAC_ROOT}/configs\""

# Core outputs
sync_dir "${RUNS_ROOT}/offers_core_full_snapshot_wide_${WIDE_STAMP}" "${MAC_ROOT}/runs/offers_core_full_snapshot_wide_${WIDE_STAMP}"
sync_dir "${RUNS_ROOT}/offers_core_full_daily_wide_${WIDE_STAMP}" "${MAC_ROOT}/runs/offers_core_full_daily_wide_${WIDE_STAMP}"
sync_dir "${RUNS_ROOT}/edgar_feature_store_full_daily_wide_${WIDE_STAMP}" "${MAC_ROOT}/runs/edgar_feature_store_full_daily_wide_${WIDE_STAMP}"
sync_dir "${RUNS_ROOT}/multiscale_full_wide_${WIDE_STAMP}" "${MAC_ROOT}/runs/multiscale_full_wide_${WIDE_STAMP}"

# Orchestrator analysis
sync_dir "${RUNS_ROOT}/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}" "${MAC_ROOT}/runs/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}"

# Inputs required by audit steps
sync_dir "${RUNS_ROOT}/selections/b11_v2_canonical" "${MAC_ROOT}/runs/selections/b11_v2_canonical"
sync_dir "${RUNS_ROOT}/offers_text_v1_20260129_073037_full" "${MAC_ROOT}/runs/offers_text_v1_20260129_073037_full"

# Configs and audit docs
sync_file "configs/column_contract_wide.yaml" "${MAC_ROOT}/configs/column_contract_wide.yaml"
sync_dir "docs/audits" "${MAC_ROOT}/docs/audits"
