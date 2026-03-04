#!/usr/bin/env bash
# Submit all 12 FusedChampion V7.3.2 benchmark jobs
# QOS: iris-snt (SnT priority on l40s)
# Resources: 1×L40s GPU, 8 CPUs, 64GB per job

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting 12 FusedChampion V7.3.2 benchmark jobs..."
echo "Partition: l40s | QOS: iris-snt | CPUs: 8 | Mem: 64G | GPU: 1"
echo ""

for script in "$SCRIPT_DIR"/fc_t*.sh; do
    name=$(basename "$script" .sh)
    echo -n "  $name -> "
    sbatch "$script"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u $USER"
