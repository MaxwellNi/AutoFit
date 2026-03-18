#!/usr/bin/env bash
#SBATCH --job-name=w3r_sta_t1_ce
#SBATCH --account=christian.fisch
#SBATCH --partition=bigmem
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=28
#SBATCH --output=/work/projects/eint/logs/phase9/w3r_sta_t1_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase9/w3r_sta_t1_ce_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"; exit 2
fi
echo "============================================================"
echo "Phase 9 Fair Benchmark | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${INSIDER_PY} -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

echo "Task: task1_outcome | Cat: statistical | Abl: core_edgar | Models: AutoCES,CrostonClassic,CrostonOptimized,CrostonSBA,DynamicOptimizedTheta,HistoricAverage,Holt,HoltWinters,Naive,WindowAverage"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category statistical --ablation core_edgar \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/statistical/core_edgar --seed 42 \
    --no-verify-first --models AutoCES,CrostonClassic,CrostonOptimized,CrostonSBA,DynamicOptimizedTheta,HistoricAverage,Holt,HoltWinters,Naive,WindowAverage
echo "Done: $(date -Iseconds)"
