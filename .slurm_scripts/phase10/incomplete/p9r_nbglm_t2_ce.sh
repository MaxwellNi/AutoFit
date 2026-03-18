#!/usr/bin/env bash
#SBATCH --job-name=p9r_nbglm_t2_ce
#SBATCH --account=npin
#SBATCH --partition=bigmem
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=28
#SBATCH --output=/work/projects/eint/logs/phase10/p9r_nbglm_t2_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/p9r_nbglm_t2_ce_%j.err
#SBATCH --export=ALL

set -e
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"

echo "Phase 9 gap-fill: NegBinGLM | task2_forecast/core_edgar | Job ${SLURM_JOB_ID}"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category ml_tabular --ablation core_edgar \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task2_forecast/ml_tabular/core_edgar --seed 42 \
    --no-verify-first --models NegativeBinomialGLM
echo "Done: $(date -Iseconds)"
