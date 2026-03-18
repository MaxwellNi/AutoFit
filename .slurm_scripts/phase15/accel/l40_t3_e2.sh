#!/bin/bash
#SBATCH --job-name=l40_ac_t3_e2
#SBATCH --account=npin
#SBATCH --partition=l40s
#SBATCH --qos=iris-snt
#SBATCH --time=2-00:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/work/projects/eint/logs/phase15/l40_ac_t3_e2_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/l40_ac_t3_e2_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue
set -euo pipefail
umask 002
cd /home/users/npin/repo_root
echo "Accel ${SLURM_JOB_NAME} | Job $SLURM_JOB_ID on $(hostname)"
echo "$(date -Iseconds) | Git: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/run_block3_benchmark_shard.py \
    --task task3_risk_adjust --category tslib_sota --ablation core_edgar_seed2 \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task3_risk_adjust/tslib_sota/core_edgar_seed2 --seed 42 \
    --models CARD,CFPT,Crossformer,DUET,DeformableTST,ETSformer,FiLM,FilterTS,FreTS,Fredformer,LightTS,MSGNet,MICN,MambaSimple,ModernTCN,MultiPatchFormer,NonstationaryTransformer,PDF,PIR,PAttn,PathFormer,Pyraformer,Reformer,SCINet,SEMPO,SRSNet,SegRNN,SparseTSF,TimeBridge,TimeFilter,TimePerceiver,TimeRecipe,xPatch
