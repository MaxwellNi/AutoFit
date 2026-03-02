#!/usr/bin/env bash
# =============================================================================
# Phase 7: Full KDD'26 Benchmark — SLURM Mass Submission
#
# 67 models × 2 ablations (core_only + core_edgar) × task1_outcome
# Optimised resource allocation per category based on Phase 1 lessons:
#   - OOM fixes: larger memory per category
#   - Per-model isolation for heavy categories
#   - Resume support built into harness (metrics.json checkpoint)
#   - SIGTERM handler saves partial results before kill
#
# Cluster: ULHPC Iris — 24 GPU nodes (4×V100, 756GB RAM, 28c each)
#                      + 168 batch nodes (112GB RAM, 28c each)
#
# Usage:
#   bash scripts/phase7_submit_all.sh [--dry-run] [--task2] [--task3]
#
# Default: task1_outcome only (primary KDD table).
# Add --task2 / --task3 to also submit task2_forecast / task3_risk_adjust.
# =============================================================================
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
STAMP="20260203_225620"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUTROOT="${REPO}/runs/benchmarks/block3_${STAMP}_iris_phase7"
LOGDIR="/work/projects/eint/logs/phase7"
PRESET="full"
SEED=42
ACCOUNT="yves.letraon"
QOS="normal"

# Parse flags
DRY_RUN=false
TASKS=("task1_outcome")
for arg in "$@"; do
    case $arg in
        --dry-run)  DRY_RUN=true ;;
        --task2)    TASKS+=("task2_forecast") ;;
        --task3)    TASKS+=("task3_risk_adjust") ;;
    esac
done

ABLATIONS=("core_only" "core_edgar")

# ─── Model Groups ───────────────────────────────────────────────────────────
# ml_tabular: 15 models, CPU-only, batched
ML_TABULAR="LogisticRegression,Ridge,Lasso,ElasticNet,SVR,KNN,RandomForest,ExtraTrees,HistGradientBoosting,LightGBM,XGBoost,CatBoost,QuantileRegressor,SeasonalNaive,MeanPredictor"

# statistical: 5 models, CPU-only, batched
STATISTICAL="AutoARIMA,AutoETS,AutoTheta,MSTL,SF_SeasonalNaive"

# deep_classical: 4 models, GPU, batched
DEEP_CLASSICAL="NBEATS,NHITS,TFT,DeepAR"

# transformer_sota: 20 models, GPU, split into 4 groups of 5
TSOTA_A="PatchTST,iTransformer,TimesNet,TSMixer,Informer"
TSOTA_B="Autoformer,FEDformer,VanillaTransformer,TiDE,NBEATSx"
TSOTA_C="BiTCN,KAN,RMoK,SOFTS,StemGNN"
TSOTA_D="DLinear,NLinear,TimeMixer,TimeXer,TSMixerx"

# foundation: 11 models, GPU, per-model (heavyweight)
FOUNDATION_MODELS=(Chronos ChronosBolt Chronos2 Moirai MoiraiLarge Moirai2 Timer TimeMoE MOMENT LagLlama TimesFM)

# irregular: 2 models, GPU, batched
IRREGULAR="GRU-D,SAITS"

# autofit: 10 variants, CPU-heavy, per-model
AUTOFIT_MODELS=(V1 V2 V2E V3 V3E V3Max V4 V5 V6 V7)

# ─── SLURM Template ─────────────────────────────────────────────────────────
# Common preamble for all jobs
PREAMBLE="
set -euo pipefail
export MAMBA_ROOT_PREFIX=/work/projects/eint/envs/.micromamba
export PATH=\"\$HOME/.local/bin:\$PATH\"
eval \"\$(micromamba shell hook -s bash)\"
micromamba activate insider
export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}\"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd /work/projects/eint/repo_root
"

# ─── Helper Functions ────────────────────────────────────────────────────────

submit_count=0

submit_job() {
    # Args: job_name partition gpus cpus mem time_limit task category ablation models outdir_suffix
    local job_name="$1"
    local partition="$2"
    local gpus="$3"
    local cpus="$4"
    local mem="$5"
    local timelimit="$6"
    local task="$7"
    local category="$8"
    local ablation="$9"
    local models="${10}"
    local outdir_suffix="${11}"

    local outdir="${OUTROOT}/${task}/${outdir_suffix}/${ablation}"
    local logfile="${LOGDIR}/${job_name}_%j"

    # GPU resource string
    local gpu_flag=""
    if [[ "$gpus" -gt 0 ]]; then
        gpu_flag="#SBATCH -G ${gpus}"
    fi

    local script="#!/bin/bash -l
#SBATCH -J ${job_name}
#SBATCH -p ${partition}
#SBATCH -A ${ACCOUNT}
#SBATCH --qos=${QOS}
#SBATCH -N 1
${gpu_flag}
#SBATCH -c ${cpus}
#SBATCH --mem=${mem}
#SBATCH -t ${timelimit}
#SBATCH -o ${logfile}.out
#SBATCH -e ${logfile}.err
#SBATCH --signal=B:TERM@120

${PREAMBLE}

echo '======================================================================'
echo \"Phase 7 Benchmark: \${SLURM_JOB_NAME}\"
echo \"  Task:     ${task}\"
echo \"  Category: ${category}\"
echo \"  Ablation: ${ablation}\"
echo \"  Models:   ${models}\"
echo \"  Job ID:   \${SLURM_JOB_ID}\"
echo \"  Node:     \$(hostname)\"
echo \"  Start:    \$(date)\"
echo '======================================================================'

mkdir -p '${outdir}'

python scripts/run_block3_benchmark_shard.py \\
    --task '${task}' \\
    --category '${category}' \\
    --models '${models}' \\
    --ablation '${ablation}' \\
    --preset '${PRESET}' \\
    --output-dir '${outdir}' \\
    --seed ${SEED} \\
    --no-verify-first

echo \"[DONE] \$(date) — exit code \$?\"
"

    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch: ${job_name} | ${partition} | GPU=${gpus} | ${cpus}c | ${mem} | ${timelimit} | ${task}/${outdir_suffix}/${ablation}"
    else
        local jobid
        jobid=$(echo "$script" | sbatch --parsable)
        echo "[SUBMITTED] ${job_name} -> ${jobid} | ${partition} | GPU=${gpus} | ${cpus}c | ${mem} | ${timelimit}"
    fi
    submit_count=$((submit_count + 1))
}

# ─── Main ────────────────────────────────────────────────────────────────────

echo "============================================================"
echo " Phase 7 Full Benchmark — SLURM Mass Submission"
echo "============================================================"
echo " Stamp:     ${STAMP}"
echo " Output:    ${OUTROOT}"
echo " Tasks:     ${TASKS[*]}"
echo " Ablations: ${ABLATIONS[*]}"
echo " Preset:    ${PRESET}"
echo " Dry run:   ${DRY_RUN}"
echo "============================================================"

mkdir -p "${LOGDIR}"

for task in "${TASKS[@]}"; do
    echo ""
    echo "━━━ Task: ${task} ━━━"

    for abl in "${ABLATIONS[@]}"; do
        echo ""
        echo "  ── Ablation: ${abl} ──"

        # ── 1. ml_tabular (15 models, CPU-only, batched) ─────────────────
        # Phase 1 lesson: 32GB caused 28 failures. Allocate 100GB.
        submit_job "p7_mltab_${abl}" \
            batch 0 14 100G 12:00:00 \
            "$task" ml_tabular "$abl" \
            "$ML_TABULAR" ml_tabular

        # ── 2. statistical (5 models, CPU-only, batched) ─────────────────
        submit_job "p7_stat_${abl}" \
            batch 0 8 64G 6:00:00 \
            "$task" statistical "$abl" \
            "$STATISTICAL" statistical

        # ── 3. deep_classical (4 models, GPU, batched) ───────────────────
        submit_job "p7_deep_${abl}" \
            gpu 1 7 128G 12:00:00 \
            "$task" deep_classical "$abl" \
            "$DEEP_CLASSICAL" deep_classical

        # ── 4. transformer_sota (20 models, GPU, 4 groups of 5) ──────────
        submit_job "p7_tsA_${abl}" \
            gpu 1 7 128G 12:00:00 \
            "$task" transformer_sota "$abl" \
            "$TSOTA_A" transformer_sota_A

        submit_job "p7_tsB_${abl}" \
            gpu 1 7 128G 12:00:00 \
            "$task" transformer_sota "$abl" \
            "$TSOTA_B" transformer_sota_B

        submit_job "p7_tsC_${abl}" \
            gpu 1 7 128G 12:00:00 \
            "$task" transformer_sota "$abl" \
            "$TSOTA_C" transformer_sota_C

        submit_job "p7_tsD_${abl}" \
            gpu 1 7 128G 12:00:00 \
            "$task" transformer_sota "$abl" \
            "$TSOTA_D" transformer_sota_D

        # ── 5. foundation (11 models, GPU, per-model) ────────────────────
        # Phase 1 lesson: Chronos/Moirai are very slow. Per-model isolation
        # with high memory and 2-day time limit.
        for model in "${FOUNDATION_MODELS[@]}"; do
            submit_job "p7_fnd_${model}_${abl}" \
                gpu 1 7 180G 1-12:00:00 \
                "$task" foundation "$abl" \
                "$model" "foundation_${model}"
        done

        # ── 6. irregular (2 models, GPU, batched) ────────────────────────
        submit_job "p7_irreg_${abl}" \
            gpu 1 7 64G 6:00:00 \
            "$task" irregular "$abl" \
            "$IRREGULAR" irregular

        # ── 7. autofit (10 variants, CPU-heavy, per-model) ───────────────
        # Each AutoFit variant does internal 5-fold CV over 31 candidates.
        # Very CPU-heavy but no GPU needed.
        for model in "${AUTOFIT_MODELS[@]}"; do
            submit_job "p7_af_${model}_${abl}" \
                batch 0 14 100G 1-00:00:00 \
                "$task" autofit "$abl" \
                "$model" "autofit_${model}"
        done
    done
done

echo ""
echo "============================================================"
echo " Total jobs submitted: ${submit_count}"
echo " Output root: ${OUTROOT}"
echo " Logs: ${LOGDIR}"
echo "============================================================"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER -t PD,R --sort=S"
echo "  watch -n 30 'squeue -u \$USER | wc -l'"
echo ""
echo "Aggregate results after completion:"
echo "  python scripts/aggregate_block3_results.py --input-dir ${OUTROOT}"
