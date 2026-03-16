#!/usr/bin/env bash
# Submit Phase 15 error rerun jobs to hopper (besteffort) and l40s (iris-snt)
# Only submit AFTER the main Phase 15 jobs for t1_co, t1_ce, t2_co finish!
# These target the 5 models that failed on old code: DeformableTST, DUET, FilterTS, PathFormer, SEMPO

set -e
cd "$(dirname "$0")"

echo "WARNING: Only submit these AFTER main Phase 15 jobs 5253903/5253904/5253905 finish!"
echo "Check with: squeue -u npin | grep p15_new"
echo ""

# On gpu (safest, will queue behind current jobs)
sbatch p15_rerun_errors_t1_co.sh
sbatch p15_rerun_errors_t1_ce.sh
sbatch p15_rerun_errors_t2_co.sh

echo "Submitted 3 error-rerun jobs to gpu partition."
echo "For hopper (besteffort), uncomment below after verifying main jobs are done:"
echo ""
echo "# sbatch --partition=hopper --qos=besteffort --mem=256G p15_rerun_errors_t1_co.sh"
echo "# sbatch --partition=hopper --qos=besteffort --mem=256G p15_rerun_errors_t1_ce.sh"
echo "# sbatch --partition=hopper --qos=besteffort --mem=256G p15_rerun_errors_t2_co.sh"
