#!/usr/bin/env bash
# Phase 9 Fair Benchmark — Saturated Submission
# This script submits ALL Phase 9 jobs across both accounts.
# Run from npin account. cfisch jobs will be submitted via ssh iris-cf.
set -e
cd /home/users/npin/repo_root

echo "=== Phase 9 Saturated Submission ==="
echo "Git HEAD: $(git rev-parse --short HEAD)"
echo "Uncommitted changes:"
git diff --stat HEAD
echo ""

# Safety check: all code changes must be committed
if [[ -n "$(git diff HEAD -- scripts/ src/)" ]]; then
    echo "ERROR: Uncommitted changes in scripts/ or src/. Commit first!"
    exit 1
fi

echo ""
echo "=== Submitting npin GPU jobs ==="
for f in .slurm_scripts/phase9/p9_fmB_*.sh .slurm_scripts/phase9/p9_tsA_*.sh .slurm_scripts/phase9/p9_irN_*.sh .slurm_scripts/phase9/p9_af9_*.sh; do
    echo "  sbatch $f"
    sbatch "$f"
done

echo ""
echo "=== Submitting npin NF gap-fill (already pending) ==="
echo "  nfG jobs already in queue (5216571-5216573)"

echo ""
echo "=== Submitting cfisch GPU+batch jobs ==="
for f in .slurm_scripts/phase9/p9_tsB_*.sh .slurm_scripts/phase9/p9_mlT_*.sh; do
    echo "  ssh iris-cf sbatch $f"
    ssh iris-cf "cd /home/users/npin/repo_root && sbatch $f"
done

echo ""
echo "=== Submission complete ==="
squeue -u npin -u cfisch -o "%.10i %.30j %.8u %.8T" 2>/dev/null
