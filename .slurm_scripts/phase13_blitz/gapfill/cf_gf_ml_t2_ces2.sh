#!/usr/bin/env bash
#SBATCH --job-name=cf_gf_ml_t2_ces2
#SBATCH --account=christian.fisch
#SBATCH --partition=bigmem
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=14

#SBATCH --output=/work/projects/eint/logs/phase13/cf_gf_ml_t2_ces2_%j.out
#SBATCH --error=/work/projects/eint/logs/phase13/cf_gf_ml_t2_ces2_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -e
umask 002
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/home/users/npin/.cache/huggingface"
cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then echo "FATAL: python missing"; exit 2; fi
echo "GapFill | Job ${SLURM_JOB_ID} on $(hostname) | bigmem/normal | $(date -Iseconds)"
echo "Models: NegativeBinomialGLM"


"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category ml_tabular --ablation core_edgar_seed2 \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task2_forecast/ml_tabular/core_edgar_seed2 --seed 42 \
    --no-verify-first --models NegativeBinomialGLM
echo "Done: $(date -Iseconds)"
