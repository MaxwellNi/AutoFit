#!/usr/bin/env bash
#SBATCH --job-name=hp_fix11_t1_ce
#SBATCH --account=npin
#SBATCH --partition=hopper
#SBATCH --qos=besteffort
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/work/projects/eint/logs/phase15/hp_fix11_t1_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/hp_fix11_t1_ce_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -e
umask 002
INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then echo "FATAL: python missing"; exit 2; fi
echo "Phase 15 Fix11 (Hopper) | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Git: $(git rev-parse --short HEAD)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category tslib_sota --ablation core_edgar \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/tslib_sota/core_edgar --seed 42 \
    --no-verify-first --models DeformableTST,DUET,FilterTS,ModernTCN,PDF,PIR,PathFormer,SEMPO,SparseTSF,TimeRecipe,xPatch
echo "Done: $(date -Iseconds)"
