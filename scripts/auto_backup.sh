#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${ROOT}/runs/backups/${STAMP}"
mkdir -p "${BACKUP_DIR}"

paths=()
for p in configs docs scripts src test RUNBOOK.md DECISION.md pyproject.toml Makefile; do
  if [ -e "${ROOT}/${p}" ]; then
    paths+=("${p}")
  fi
done

# include all sanity reports (small, critical audit outputs)
for p in "${ROOT}"/runs/sanity_*; do
  if [ -e "${p}" ]; then
    paths+=("runs/$(basename "${p}")")
  fi
done

if [ ${#paths[@]} -eq 0 ]; then
  echo "No backup targets found; aborting." >&2
  exit 1
fi

tar -czf "${BACKUP_DIR}/repo_backup.tgz" -C "${ROOT}" "${paths[@]}" 2>/dev/null
echo "${BACKUP_DIR}/repo_backup.tgz" > "${ROOT}/runs/backups/latest.txt"

# Create git bundle (if repo initialized)
if [ -d "${ROOT}/.git" ]; then
  git -C "${ROOT}" bundle create "${BACKUP_DIR}/repo_bundle.bundle" --all
  echo "${BACKUP_DIR}/repo_bundle.bundle" > "${ROOT}/runs/backups/latest.txt"
fi

# Generate sha256 for essential files (tracked if git is available)
if [ -d "${ROOT}/.git" ]; then
  git -C "${ROOT}" ls-files | while read -r f; do
    if [ -f "${ROOT}/${f}" ]; then
      sha256sum "${ROOT}/${f}"
    fi
  done > "${BACKUP_DIR}/sha256_essential.txt"
else
  sha256sum "${BACKUP_DIR}/repo_backup.tgz" > "${BACKUP_DIR}/sha256_essential.txt"
fi

echo "Backup created: ${BACKUP_DIR}/repo_backup.tgz"
