#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMITTED=0
for script in "$DIR"/v734_*.sh; do
    JOB_ID=$(sbatch "$script" | awk '{print $4}')
    echo "Submitted $(basename "$script") → $JOB_ID"
    SUBMITTED=$((SUBMITTED + 1))
done
echo "Total submitted: $SUBMITTED jobs"
