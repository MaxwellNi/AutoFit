#!/usr/bin/env bash
# ============================================================
# Submit all 11 V7.3.1 besteffort jobs to gpu+l40s+hopper
# Run this from cfisch account: ssh iris-cf, cd repo_root
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /mnt/aiongpfs/projects/eint/repo_root

# Ensure log directory exists
mkdir -p /work/projects/eint/logs/phase7_v731

echo "============================================================"
echo "V7.3.1 Besteffort Multi-Partition Submission"
echo "Partitions: gpu, l40s, hopper"
echo "QOS: besteffort (preemptible, auto-requeue)"
echo "Date: $(date -Iseconds)"
echo "============================================================"

# Cancel any existing PENDING V7.3.1 jobs from cfisch
echo ""
echo "Checking for existing V7.3.1 jobs..."
EXISTING=$(squeue -u cfisch --name="v731_t1_co,v731_t1_ct,v731_t1_ce,v731_t1_fu,v731_t2_co,v731_t2_ct,v731_t2_ce,v731_t2_fu,v731_t3_co,v731_t3_ce,v731_t3_fu" -h -o "%i" 2>/dev/null || true)
EXISTING_BE=$(squeue -u cfisch --name="v731_t1_co_be,v731_t1_ct_be,v731_t1_ce_be,v731_t1_fu_be,v731_t2_co_be,v731_t2_ct_be,v731_t2_ce_be,v731_t2_fu_be,v731_t3_co_be,v731_t3_ce_be,v731_t3_fu_be" -h -o "%i" 2>/dev/null || true)
ALL_EXISTING="${EXISTING} ${EXISTING_BE}"
ALL_EXISTING=$(echo "$ALL_EXISTING" | xargs)  # trim

if [[ -n "$ALL_EXISTING" ]]; then
    echo "Cancelling existing V7.3.1 jobs: $ALL_EXISTING"
    scancel $ALL_EXISTING 2>/dev/null || true
    sleep 2
else
    echo "No existing V7.3.1 jobs found."
fi

# Submit all 11 jobs
echo ""
echo "Submitting 11 V7.3.1 besteffort jobs..."
echo ""

SUBMITTED=0
for script in "${SCRIPT_DIR}"/v731_*_be.sh; do
    name=$(basename "$script" .sh)
    JOBID=$(sbatch "$script" 2>&1 | grep -oP '\d+')
    if [[ -n "$JOBID" ]]; then
        echo "  ✓ ${name}: JobID ${JOBID}"
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "  ✗ ${name}: FAILED to submit"
    fi
done

echo ""
echo "============================================================"
echo "Submitted: ${SUBMITTED}/11 jobs"
echo "QOS: besteffort (preemptible, auto-requeue on preemption)"
echo "Partitions: gpu, l40s, hopper (SLURM picks best available)"
echo ""
echo "Key features:"
echo "  • --requeue: auto-resubmit on preemption"
echo "  • --signal=USR1@120: 2min warning → checkpoint save"
echo "  • Harness resume: loads existing metrics.json, skips done combos"
echo "  • Multi-partition: SLURM picks gpu/l40s/hopper as available"
echo ""
echo "Monitor: squeue -u cfisch --format='%.10i %.12j %.8P %.6T %.10M %.6D %R'"
echo "============================================================"
