#!/bin/bash
# Prepare data sync from 4090 to Iris HPC
#
# This script generates rsync commands to sync the freeze data
# to the Iris cluster. Run on the 4090 machine.
#
# Usage:
#   ./prepare_iris_sync.sh [IRIS_USER@IRIS_HOST] [IRIS_DEST]
#
# Example:
#   ./prepare_iris_sync.sh pni@iris-login01 /scratch/users/pni/narrative

set -e

STAMP="20260203_225620"
IRIS_TARGET="${1:-pni@iris-login01}"
IRIS_DEST="${2:-/scratch/users/pni/narrative}"

echo "=============================================="
echo "Iris Data Sync Preparation"
echo "=============================================="
echo "Stamp: ${STAMP}"
echo "Target: ${IRIS_TARGET}:${IRIS_DEST}"
echo "=============================================="

# Directories to sync (read-only freeze data)
SYNC_DIRS=(
    "runs/offers_core_full_daily_wide_${STAMP}"
    "runs/offers_core_full_snapshot_wide_${STAMP}"
    "runs/offers_text_v1_20260129_073037_full"
    "runs/edgar_feature_store_full_daily_wide_${STAMP}"
    "runs/multiscale_full_wide_${STAMP}"
    "runs/orchestrator/20260129_073037/analysis/wide_${STAMP}"
    "docs/audits/FULL_SCALE_POINTER.yaml"
)

echo ""
echo "Directories to sync:"
for dir in "${SYNC_DIRS[@]}"; do
    if [ -e "$dir" ]; then
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  [OK] $dir ($SIZE)"
    else
        echo "  [MISSING] $dir"
    fi
done

echo ""
echo "Generated rsync commands:"
echo "=============================================="

for dir in "${SYNC_DIRS[@]}"; do
    if [ -e "$dir" ]; then
        # Create parent directory on remote
        PARENT_DIR=$(dirname "$dir")
        echo "ssh ${IRIS_TARGET} 'mkdir -p ${IRIS_DEST}/${PARENT_DIR}'"
        echo "rsync -avz --progress ${dir} ${IRIS_TARGET}:${IRIS_DEST}/${dir}"
        echo ""
    fi
done

echo "=============================================="
echo "To run all syncs, save this output and execute."
echo "=============================================="

# Also sync the repo code (excluding runs/)
echo ""
echo "# Sync repo code (excluding runs/)"
echo "rsync -avz --progress \\"
echo "    --exclude 'runs/' \\"
echo "    --exclude '.git/' \\"
echo "    --exclude '__pycache__/' \\"
echo "    --exclude '*.pyc' \\"
echo "    --exclude '.pytest_cache/' \\"
echo "    ~/projects/repo_root/ ${IRIS_TARGET}:${IRIS_DEST}/repo_root/"
