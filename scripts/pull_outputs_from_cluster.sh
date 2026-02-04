#!/usr/bin/env bash
set -euo pipefail

# Run this script on your Mac (or any machine that can SSH to the cluster).

# ULHPC login endpoint uses non-default SSH port 8022.
CLUSTER_HOST="${CLUSTER_HOST:-access-iris.uni.lu}"
CLUSTER_USER="${CLUSTER_USER:-npin}"
CLUSTER="${CLUSTER_USER}@${CLUSTER_HOST}"
CLUSTER_REPO_ROOT="${CLUSTER_REPO_ROOT:-/home/users/npin/repo_root}"
CLUSTER_RUNS_ROOT="${CLUSTER_RUNS_ROOT:-/work/projects/eint/runs}"

LOCAL_ROOT="${LOCAL_ROOT:-$(pwd)}"
LOCAL_RUNS="${LOCAL_RUNS:-${LOCAL_ROOT}/runs}"
WIDE_STAMP="${WIDE_STAMP:-20260203_225620}"
SSH_PORT="${SSH_PORT:-8022}"
SSH_EXTRA_OPTS="${SSH_EXTRA_OPTS:-}"

RSYNC_OPTS="${RSYNC_OPTS:--av --partial --progress}"

mkdir -p "${LOCAL_RUNS}" "${LOCAL_ROOT}/docs/audits" "${LOCAL_ROOT}/configs"

sync_dir() {
  local src="$1"
  local dest="$2"
  echo "Pull: ${CLUSTER}:${src} -> ${dest}"
  rsync ${RSYNC_OPTS} -e "ssh -p ${SSH_PORT} ${SSH_EXTRA_OPTS}" "${CLUSTER}:${src%/}/" "${dest%/}/"
}

sync_file() {
  local src="$1"
  local dest="$2"
  echo "Pull: ${CLUSTER}:${src} -> ${dest}"
  rsync ${RSYNC_OPTS} -e "ssh -p ${SSH_PORT} ${SSH_EXTRA_OPTS}" "${CLUSTER}:${src}" "${dest}"
}

# Core outputs
sync_dir "${CLUSTER_RUNS_ROOT}/offers_core_full_snapshot_wide_${WIDE_STAMP}" "${LOCAL_RUNS}/offers_core_full_snapshot_wide_${WIDE_STAMP}"
sync_dir "${CLUSTER_RUNS_ROOT}/offers_core_full_daily_wide_${WIDE_STAMP}" "${LOCAL_RUNS}/offers_core_full_daily_wide_${WIDE_STAMP}"
sync_dir "${CLUSTER_RUNS_ROOT}/edgar_feature_store_full_daily_wide_${WIDE_STAMP}" "${LOCAL_RUNS}/edgar_feature_store_full_daily_wide_${WIDE_STAMP}"
sync_dir "${CLUSTER_RUNS_ROOT}/multiscale_full_wide_${WIDE_STAMP}" "${LOCAL_RUNS}/multiscale_full_wide_${WIDE_STAMP}"

# Orchestrator analysis
sync_dir "${CLUSTER_RUNS_ROOT}/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}" "${LOCAL_RUNS}/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}"

# Inputs required by audit steps
sync_dir "${CLUSTER_RUNS_ROOT}/selections/b11_v2_canonical" "${LOCAL_RUNS}/selections/b11_v2_canonical"
sync_dir "${CLUSTER_RUNS_ROOT}/offers_text_v1_20260129_073037_full" "${LOCAL_RUNS}/offers_text_v1_20260129_073037_full"

# Configs and audit docs from cluster repo root
sync_file "${CLUSTER_REPO_ROOT}/configs/column_contract_wide.yaml" "${LOCAL_ROOT}/configs/column_contract_wide.yaml"
sync_dir "${CLUSTER_REPO_ROOT}/docs/audits" "${LOCAL_ROOT}/docs/audits"
