#!/usr/bin/env bash
#SBATCH --job-name=l40_fix11_t2_co
#SBATCH --account=npin
#SBATCH --partition=l40s
#SBATCH --qos=iris-snt
#SBATCH --time=2-00:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/work/projects/eint/logs/phase15/l40_fix11_t2_co_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/l40_fix11_t2_co_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -e
umask 002
INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/home/users/npin/.cache/huggingface"
cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then echo "FATAL: python missing"; exit 2; fi
echo "l40s job | ${SLURM_JOB_ID} on $(hostname) | 2026-03-17T01:38:07+01:00"
echo "Git: $(git rev-parse --short HEAD)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Models: DeformableTST,DUET,FilterTS,ModernTCN,PDF,PIR,PathFormer,SEMPO,SparseTSF,TimeRecipe,xPatch"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category tslib_sota --ablation core_only \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task2_forecast/tslib_sota/core_only --seed 42 \
    --no-verify-first --models DeformableTST,DUET,FilterTS,ModernTCN,PDF,PIR,PathFormer,SEMPO,SparseTSF,TimeRecipe,xPatch
echo "Done: $(date -Iseconds)"
