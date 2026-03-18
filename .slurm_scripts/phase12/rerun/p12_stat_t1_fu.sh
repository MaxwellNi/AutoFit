#!/usr/bin/env bash
#SBATCH --job-name=p12_stat_t1_fu
#SBATCH --account=npin
#SBATCH --partition=bigmem
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=320G
#SBATCH --cpus-per-task=28
#SBATCH --output=/work/projects/eint/logs/phase12/p12_stat_t1_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase12/p12_stat_t1_fu_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -e
umask 002
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"

if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing"; exit 2
fi
echo "============================================================"
echo "Phase 12 Text Re-run | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${INSIDER_PY} -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

# Verify text embeddings exist
if [[ ! -f "runs/text_embeddings/text_embeddings.parquet" ]]; then
  echo "FATAL: text_embeddings.parquet not found — cannot run text ablation!"; exit 3
fi
echo "Text embeddings verified: $(ls -lh runs/text_embeddings/text_embeddings.parquet)"

echo "Task: task1_outcome | Cat: statistical | Abl: full | Models: AutoARIMA,AutoCES,AutoETS,AutoTheta,CrostonClassic,CrostonOptimized,CrostonSBA,DynamicOptimizedTheta,HistoricAverage,Holt,HoltWinters,MSTL,Naive,SF_SeasonalNaive,WindowAverage"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category statistical --ablation full \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/statistical/full --seed 42 \
    --no-verify-first --models AutoARIMA,AutoCES,AutoETS,AutoTheta,CrostonClassic,CrostonOptimized,CrostonSBA,DynamicOptimizedTheta,HistoricAverage,Holt,HoltWinters,MSTL,Naive,SF_SeasonalNaive,WindowAverage
echo "Done: $(date -Iseconds)"
