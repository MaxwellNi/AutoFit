#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(cat "${ROOT}/runs/backups/current_audit_stamp.txt")"
ORCH_DIR="${ROOT}/runs/orchestrator/${STAMP}"
mkdir -p "${ORCH_DIR}"

OFFERS_CORE="runs/offers_core_v2_20260127_043052/offers_core.parquet"
EDGAR_STAMP="$(cat "${ROOT}/runs/edgar_feature_store/latest.txt")"
EDGAR_DIR="runs/edgar_feature_store/${EDGAR_STAMP}/edgar_features"
SELECTION_DIR="runs/selections/b11_v2_canonical"

REPO_4090="/home/pni/projects/repo_root"
REPO_IRIS="/home/users/npin/repo_root"
REPO_AION="/home/users/npin/repo_root"

log="${ORCH_DIR}/SYNC_REPORT.txt"
echo "stamp=${STAMP}" | tee "${log}"

hash_dir() {
  local dir="$1"
  find "$dir" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
}

local_offers_sha=$(sha256sum "${ROOT}/${OFFERS_CORE}" | awk '{print $1}')
local_edgar_hash=$(hash_dir "${ROOT}/${EDGAR_DIR}")
local_sel_sha=$(sha256sum "${ROOT}/${SELECTION_DIR}/sampled_entities.json" | awk '{print $1}')
local_sel_hash=$(cat "${ROOT}/${SELECTION_DIR}/selection_hash.txt")

echo "local_offers_sha=${local_offers_sha}" | tee -a "${log}"
echo "local_edgar_hash=${local_edgar_hash}" | tee -a "${log}"
echo "local_selection_sha=${local_sel_sha}" | tee -a "${log}"
echo "local_selection_hash=${local_sel_hash}" | tee -a "${log}"

sync_one() {
  local host="$1"
  local repo="$2"
  echo "sync_host=${host} repo=${repo}" | tee -a "${log}"
  ssh "${host}" "mkdir -p ${repo}/$(dirname ${OFFERS_CORE}) ${repo}/${EDGAR_DIR} ${repo}/${SELECTION_DIR}"
  rsync -av "${ROOT}/${OFFERS_CORE}" "${host}:${repo}/${OFFERS_CORE}"
  rsync -av "${ROOT}/${EDGAR_DIR}/" "${host}:${repo}/${EDGAR_DIR}/"
  rsync -av "${ROOT}/${SELECTION_DIR}/" "${host}:${repo}/${SELECTION_DIR}/"

  remote_offers_sha=$(ssh "${host}" "sha256sum ${repo}/${OFFERS_CORE} | awk '{print \$1}'")
  remote_edgar_hash=$(ssh "${host}" "find ${repo}/${EDGAR_DIR} -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print \$1}'")
  remote_sel_sha=$(ssh "${host}" "sha256sum ${repo}/${SELECTION_DIR}/sampled_entities.json | awk '{print \$1}'")
  remote_sel_hash=$(ssh "${host}" "cat ${repo}/${SELECTION_DIR}/selection_hash.txt")

  echo "remote_offers_sha(${host})=${remote_offers_sha}" | tee -a "${log}"
  echo "remote_edgar_hash(${host})=${remote_edgar_hash}" | tee -a "${log}"
  echo "remote_selection_sha(${host})=${remote_sel_sha}" | tee -a "${log}"
  echo "remote_selection_hash(${host})=${remote_sel_hash}" | tee -a "${log}"

  if [ "${remote_offers_sha}" != "${local_offers_sha}" ]; then
    echo "FATAL: offers_core sha mismatch for ${host}" | tee -a "${log}"
    exit 2
  fi
  if [ "${remote_edgar_hash}" != "${local_edgar_hash}" ]; then
    echo "FATAL: edgar_dir hash mismatch for ${host}" | tee -a "${log}"
    exit 2
  fi
  if [ "${remote_sel_sha}" != "${local_sel_sha}" ]; then
    echo "FATAL: selection json sha mismatch for ${host}" | tee -a "${log}"
    exit 2
  fi
  if [ "${remote_sel_hash}" != "${local_sel_hash}" ]; then
    echo "FATAL: selection hash mismatch for ${host}" | tee -a "${log}"
    exit 2
  fi
}

sync_one 4090 "${REPO_4090}"
sync_one iris "${REPO_IRIS}"
sync_one aion "${REPO_AION}"

echo "SYNC OK" | tee -a "${log}"
