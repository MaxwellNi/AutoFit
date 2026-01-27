#!/bin/bash
set -e

SRC="${1:-runs}"
DEST="${2:-/work/projects/eint/runs}"

echo "Syncing results from ${SRC} to ${DEST}"
rsync -av "${SRC}/" "${DEST}/"

echo "Done."
