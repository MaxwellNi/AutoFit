#!/usr/bin/env bash
#SBATCH --job-name=cf_gf_ts_t2_ces2
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase13/cf_gf_ts_t2_ces2_%j.out
#SBATCH --error=/work/projects/eint/logs/phase13/cf_gf_ts_t2_ces2_%j.err
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
echo "GapFill | Job ${SLURM_JOB_ID} on $(hostname) | gpu/normal | $(date -Iseconds)"
echo "Models: TimeFilter,MultiPatchFormer,MSGNet,PAttn,MambaSimple,Crossformer,ETSformer,LightTS,Pyraformer,Reformer"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category tslib_sota --ablation core_edgar_seed2 \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task2_forecast/tslib_sota/core_edgar_seed2 --seed 42 \
    --no-verify-first --models TimeFilter,MultiPatchFormer,MSGNet,PAttn,MambaSimple,Crossformer,ETSformer,LightTS,Pyraformer,Reformer
echo "Done: $(date -Iseconds)"
