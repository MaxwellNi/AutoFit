#!/usr/bin/env bash
#SBATCH --job-name=p15_fix11_t1_co
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase15/p15_fix11_t1_co_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/p15_fix11_t1_co_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -e
umask 002
INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then echo "FATAL: python missing"; exit 2; fi
echo "Phase 15 Fix11 Rerun | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Git: $(git rev-parse --short HEAD)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Models: DeformableTST,DUET,FilterTS,ModernTCN,PDF,PIR,PathFormer,SEMPO,SparseTSF,TimeRecipe,xPatch"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category tslib_sota --ablation core_only \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/tslib_sota/core_only --seed 42 \
    --no-verify-first --models DeformableTST,DUET,FilterTS,ModernTCN,PDF,PIR,PathFormer,SEMPO,SparseTSF,TimeRecipe,xPatch
echo "Done: $(date -Iseconds)"
