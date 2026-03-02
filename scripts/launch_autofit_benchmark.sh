#!/usr/bin/env bash
# =============================================================================
# Launch AutoFit v1/v2/v2E benchmark shards on Iris HPC.
#
# 6 shards: 3 tasks × 2 ablations (core_only, core_edgar)
# Uses batch partition (CPU-only, 128GB, 28 cores, 48h limit)
# AutoFit does meta-feature computation + routing + trains underlying model.
# If the underlying model is GPU-based (deep/transformer/foundation),
# it falls back to LightGBM (tabular expert always available).
# =============================================================================
set -euo pipefail

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
STAMP="20260203_225620"
OUTBASE="${REPO}/runs/benchmarks/block3_${STAMP}_iris_full"
SCRIPT="${REPO}/scripts/run_block3_benchmark_shard.py"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")
ABLATIONS=("core_only" "core_edgar")

echo "=== AutoFit Benchmark Launch ==="
echo "STAMP: ${STAMP}"
echo "Output: ${OUTBASE}"
echo ""

for task in "${TASKS[@]}"; do
    for abl in "${ABLATIONS[@]}"; do
        OUTDIR="${OUTBASE}/${task}/autofit/${abl}"
        JOBNAME="b3af_${task}_autofit_${abl}"

        # Skip if MANIFEST.json already says completed
        if [[ -f "${OUTDIR}/MANIFEST.json" ]]; then
            status=$(python3 -c "import json; print(json.load(open('${OUTDIR}/MANIFEST.json')).get('status','?'))" 2>/dev/null || echo "?")
            if [[ "$status" == "completed" ]]; then
                echo "SKIP ${JOBNAME} (already completed)"
                continue
            fi
        fi

        echo "SUBMIT ${JOBNAME}"
        sbatch \
            --job-name="${JOBNAME}" \
            --partition=batch \
            --qos=normal \
            --account=yves.letraon \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=28 \
            --mem=112G \
            --time=2-00:00:00 \
            --output="${OUTDIR}/slurm_%j.log" \
            --error="${OUTDIR}/slurm_%j.err" \
            --wrap="
                set -euo pipefail
                mkdir -p ${OUTDIR}
                eval \"\$(/mnt/aiongpfs/users/npin/.local/bin/micromamba shell hook --shell bash)\"
                micromamba activate insider
                cd ${REPO}
                python ${SCRIPT} \
                    --task ${task} \
                    --category autofit \
                    --ablation ${abl} \
                    --preset full \
                    --output-dir ${OUTDIR} \
                    --no-verify-first
            "
    done
done

echo ""
echo "=== All AutoFit shards submitted ==="
echo "Monitor: squeue -u \$USER | grep b3af"
