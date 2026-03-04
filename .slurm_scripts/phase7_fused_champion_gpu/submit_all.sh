#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Submitting FusedChampion V7.3.2 (GPU partition, 192G)..."
echo "Partition: gpu | QOS: iris-gpu-long | CPUs: 8 | Mem: 192G | GPU: volta:1"
echo ""
for script in "$SCRIPT_DIR"/fc_t*.sh; do
    name=$(basename "$script" .sh)
    echo -n "  $name -> "
    sbatch "$script"
done
echo ""
echo "All jobs submitted. Monitor with: squeue -u $USER"
