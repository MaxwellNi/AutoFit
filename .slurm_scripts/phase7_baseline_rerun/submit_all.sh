#!/usr/bin/env bash
# Submit all baseline re-run jobs in dependency chains to respect QOS limits
# MaxJobsPU=4: submit 4 at a time, chain the rest with --dependency=afterany
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Collect all .sh scripts except this one and generate_scripts.sh
scripts=()
for f in rerun_*.sh; do
    [ -f "$f" ] && scripts+=("$f")
done

echo "Found ${#scripts[@]} scripts to submit"
echo ""

# Submit in waves of 4
wave=0
prev_ids=()
for ((i=0; i<${#scripts[@]}; i++)); do
    script="${scripts[$i]}"
    idx=$((i % 4))
    
    if [ $idx -eq 0 ] && [ $i -gt 0 ]; then
        wave=$((wave + 1))
        echo "--- Wave $wave (depends on previous wave) ---"
    fi
    
    if [ ${#prev_ids[@]} -gt 0 ] && [ $idx -eq 0 ]; then
        # Build dependency string from previous wave
        dep_str=$(IFS=:; echo "${prev_ids[*]}")
        prev_ids=()
    fi
    
    if [ $wave -eq 0 ]; then
        # First wave: no dependencies
        jobid=$(sbatch --parsable "$script")
    else
        # Subsequent waves: depend on corresponding slot from prev wave
        if [ $idx -lt ${#prev_ids[@]} ]; then
            jobid=$(sbatch --parsable --dependency=afterany:${prev_ids[$idx]} "$script")
        else
            jobid=$(sbatch --parsable "$script")
        fi
    fi
    
    echo "  Submitted $script → JobID=$jobid (wave=$wave, slot=$idx)"
    
    if [ $idx -eq 0 ] && [ $wave -gt 0 ]; then
        prev_ids=("$jobid")
    else
        prev_ids+=("$jobid")
    fi
done

echo ""
echo "Total submitted: ${#scripts[@]} jobs in $((wave + 1)) waves"
echo "Monitor: squeue -u \$USER"
