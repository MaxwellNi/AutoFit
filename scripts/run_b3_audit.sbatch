#!/bin/bash -l
# DEPRECATED LEGACY SURFACE
# This script references the retired V72 / early V73 truth-pack line and/or
# the archived pre-Phase-9 audit surface.
# It is kept for historical traceability and compatibility only.
# Do not use it for current Phase 9 / AutoFitV739 status, ranking, or execution decisions.
# Current truth docs: docs/CURRENT_SOURCE_OF_TRUTH.md and docs/PHASE9_V739_FACT_ALIGNMENT.md

#SBATCH --job-name=b3_audit
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/npin/repo_root/runs/slurm/b3_audit_%j.out
#SBATCH --error=/home/users/npin/repo_root/runs/slurm/b3_audit_%j.err
###############################################################################
# Block 3 Benchmark Audit — SLURM wrapper
# Runs the full 8-step audit and produces audit documents.
###############################################################################
set -euo pipefail

REPO="/home/users/npin/repo_root"
MAMBA_EXE="/mnt/aiongpfs/users/npin/.local/bin/micromamba"

echo "=== Block 3 Benchmark Audit ==="
echo "Job:    ${SLURM_JOB_ID}"
echo "Node:   $(hostname -s)"
echo "Start:  $(date -Iseconds)"
echo "==============================="

eval "$(${MAMBA_EXE} shell hook --shell bash)"
micromamba activate insider

cd "${REPO}"
python scripts/block3_benchmark_audit.py

echo "Audit completed: $(date -Iseconds)"
echo "Outputs:"
echo "  docs/audits/BLOCK3_BENCHMARK_AUDIT_20260209.md"
echo "  docs/audits/BLOCK3_EXPERIMENT_COVERAGE_MATRIX.md"
echo "  docs/audits/BLOCK3_AUDIT_FINDINGS_20260209.json"
