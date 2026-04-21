#!/bin/bash
#SBATCH --job-name=g2_ac_t2_ce
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase15/g2_ac_t2_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/g2_ac_t2_ce_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue
set -euo pipefail
_requeue_handler() { echo "$(date -Iseconds) USR1: timeout approaching, requeuing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; }
trap _requeue_handler USR1
umask 002
cd /home/users/npin/repo_root
echo "AccelV2 ${SLURM_JOB_NAME} | Job $SLURM_JOB_ID on $(hostname)"
echo "$(date -Iseconds) | Git: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category tslib_sota --ablation core_edgar \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task2_forecast/tslib_sota/core_edgar --seed 42 \
    --models CARD,Crossformer,DUET,ETSformer,FiLM,FilterTS,FreTS,Fredformer,LightTS,MSGNet,MambaSimple,ModernTCN,NonstationaryTransformer,PDF,PIR,PAttn,Pyraformer,Reformer,SCINet,SRSNet,SegRNN,TimeRecipe,xPatch
