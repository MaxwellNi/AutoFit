#!/bin/bash
#SBATCH --job-name=hpcf_ac_t1_ct
#SBATCH --account=christian.fisch
#SBATCH --partition=hopper
#SBATCH --qos=besteffort
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=/work/projects/eint/logs/phase15/hpcf_ac_t1_ct_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/hpcf_ac_t1_ct_%j.err
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
    --task task1_outcome --category tslib_sota --ablation core_text \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/tslib_sota/core_text --seed 42 \
    --models CARD,CFPT,Crossformer,DUET,DeformableTST,ETSformer,FiLM,FilterTS,FreTS,Fredformer,LightTS,MSGNet,MICN,MambaSimple,ModernTCN,MultiPatchFormer,NonstationaryTransformer,PDF,PIR,PAttn,PathFormer,Pyraformer,Reformer,SCINet,SEMPO,SRSNet,SegRNN,SparseTSF,TimeBridge,TimeFilter,TimePerceiver,TimeRecipe,xPatch
