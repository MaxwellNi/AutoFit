#!/usr/bin/env bash
#SBATCH --job-name=cf_nbglm_t1_ce
#SBATCH --account=christian.fisch
#SBATCH --partition=bigmem
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=28
#SBATCH --output=/work/projects/eint/logs/phase11/cf_nbglm_t1_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase11/cf_nbglm_t1_ce_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"; exit 2
fi
echo "============================================================"
echo "NegBinGLM Gap-Fill (cfisch) | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${INSIDER_PY} -V)"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

echo "Task: task1_outcome | Cat: ml_tabular | Abl: core_edgar | Models: NegativeBinomialGLM"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category ml_tabular --ablation core_edgar \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/ml_tabular/core_edgar --seed 42 \
    --no-verify-first --models NegativeBinomialGLM
echo "Done: $(date -Iseconds)"
