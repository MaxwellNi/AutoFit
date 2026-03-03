#!/usr/bin/env bash
# ============================================================
# Submit all 11 V7.3.2 besteffort jobs to gpu+l40s+hopper
# Run from cfisch OR npin account: cd repo_root
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /mnt/aiongpfs/projects/eint/repo_root

# Ensure log directory exists
mkdir -p /work/projects/eint/logs/phase7_v732

echo "============================================================"
echo "V7.3.2 Temporal Oracle Ensemble — Besteffort Submission"
echo "Partitions: gpu, l40s, hopper"
echo "QOS: besteffort (preemptible, auto-requeue)"
echo "Date: $(date -Iseconds)"
echo "============================================================"

SUBMITTED=0
FAILED=0

echo "Submitting v732_t1_co_be.sh..."
if sbatch "$SCRIPT_DIR/v732_t1_co_be.sh"; then
    ((SUBMITTED++))
else
    echo "  FAILED to submit v732_t1_co_be.sh"
    ((FAILED++))
fi

echo "Submitting v732_t1_ct_be.sh..."
if sbatch "$SCRIPT_DIR/v732_t1_ct_be.sh"; then
    ((SUBMITTED++))
else
    echo "  FAILED to submit v732_t1_ct_be.sh"
    ((FAILED++))
fi

echo "Submitting v732_t1_ce_be.sh..."
if sbatch "$SCRIPT_DIR/v732_t1_ce_be.sh"; then
    ((SUBMITTED++))
else
    echo "  FAILED to submit v732_t1_ce_be.sh"
    ((FAILED++))
fi

echo "Submitting v732_t1_fu_be.sh..."
if sbatch "$SCRIPT_DIR/v732_t1_fu_be.sh"; then
    ((SUBMITTED++))
else
    echo "  FAILED to submit v732_t1_fu_be.sh"
    ((FAILED++))
fi

echo "Submitting v732_t2_co_be.sh..."
if sbatch "$SCRIPT_DIR/v732_t2_co_be.sh"; then
    ((SUBMITTED++))
else
    echo "  FAILED to submit v732_t2_co_be.sh"
    ((FAILED++))
fi

echo "Submitting v732_t2_ct_be.sh..."
if sbatch "$SCRIPT_DIR/v732_t2_ct_be.sh"; then
    ((SUBMITTED++))
else
    echo "  FAILED to submit v732_t2_ct_be.sh"
    ((FAILED++))
fi

echo "Submitting v732_t2_ce_be.sh..."
if sbatch "$SCRIPT_DIR/v732_t2_ce_be.sh"; then
    ((SUBMITTED++))
else
    echo "  FAILED to submit v732_t2_ce_be.sh"
    ((FAILED++))
fi

echo "Submitting v732_t2_fu_be.sh..."
if sbatch "$SCRIPT_DIR/v732_t2_fu_be.sh"; then
    ((SUBMITTED++))
else
    echo "  FAILED to submit v732_t2_fu_be.sh"
    ((FAILED++))
fi

echo "Submitting v732_t3_co_be.sh..."
if sbatch "$SCRIPT_DIR/v732_t3_co_be.sh"; then
    ((SUBMITTED++))
else
    echo "  FAILED to submit v732_t3_co_be.sh"
    ((FAILED++))
fi

echo "Submitting v732_t3_ce_be.sh..."
if sbatch "$SCRIPT_DIR/v732_t3_ce_be.sh"; then
    ((SUBMITTED++))
else
    echo "  FAILED to submit v732_t3_ce_be.sh"
    ((FAILED++))
fi

echo "Submitting v732_t3_fu_be.sh..."
if sbatch "$SCRIPT_DIR/v732_t3_fu_be.sh"; then
    ((SUBMITTED++))
else
    echo "  FAILED to submit v732_t3_fu_be.sh"
    ((FAILED++))
fi

echo "============================================================"
echo "Submitted: ${SUBMITTED} / 11 jobs"
echo "Failed: ${FAILED}"
echo "Monitor: squeue -u $(whoami) --name=$(echo v732_*_be | tr ' ' ',')"
echo "============================================================"
