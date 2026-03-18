#!/usr/bin/env bash
#SBATCH --job-name=cf_v737_t1_ce
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase10/cf_v737_t1_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/cf_v737_t1_ce_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root
echo "============================================================"
echo "Phase 10 V737 (cfisch) | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${INSIDER_PY} -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

echo "Task: task1_outcome | Cat: autofit | Abl: core_edgar | Models: AutoFitV737"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category autofit --ablation core_edgar \
    --preset full --output-dir runs/benchmarks/block3_phase10/v737/task1_outcome/autofit/core_edgar --seed 42 \
    --no-verify-first --models AutoFitV737
echo "Done: $(date -Iseconds)"
