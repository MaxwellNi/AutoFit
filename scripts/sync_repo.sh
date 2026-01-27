#!/bin/bash
set -e

SRC="${1:-.}"
DEST="${2:-/work/projects/eint/AutoFit-TS}"

echo "Syncing repo from ${SRC} to ${DEST}"
rsync -av \
  --exclude ".git" \
  --exclude "data/raw" \
  --exclude "runs" \
  "${SRC}/" "${DEST}/"

echo "Done."
