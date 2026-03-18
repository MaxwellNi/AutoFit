#!/usr/bin/env bash
# Phase 15 Rerun: Submit all fix11 + gap-fill reruns
set -e
cd "$(dirname "$0")"

echo "=== Phase 15 Fix11 Rerun (11 fixed models × 3 conditions) ==="
for f in rerun11/p15_fix11_*.sh; do
    echo "Submitting $f ..."
    sbatch "$f"
done

echo ""
echo "=== Gap-fill Rerun (10 models × 2 conditions, gpu partition) ==="
for f in gapfill/gf_ts_*.sh; do
    echo "Submitting $f ..."
    sbatch "$f"
done

echo ""
echo "All submitted. Check with: squeue -u npin"
