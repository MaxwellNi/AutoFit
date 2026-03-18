#!/usr/bin/env bash
#SBATCH --job-name=p9r_reformer_t1_fu
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase10/p9r_reformer_t1_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/p9r_reformer_t1_fu_%j.err
#SBATCH --export=ALL

set -e
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"

echo "Phase 9 gap-fill: Reformer | task1_outcome/full | Job ${SLURM_JOB_ID}"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category tslib_sota --ablation full \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/tslib_sota/full --seed 42 \
    --no-verify-first --models Reformer
echo "Done: $(date -Iseconds)"
