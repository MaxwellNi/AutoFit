#!/usr/bin/env bash
#SBATCH --job-name=p15_new_t3_ce
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase15/p15_new_t3_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/p15_new_t3_ce_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -e
umask 002
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"

if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing"; exit 2
fi
echo "============================================================"
echo "Phase 15 TSLib New Models | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${INSIDER_PY} -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

echo "Task: task3_risk_adjust | Cat: tslib_sota | Abl: core_edgar | Models: CARD,CFPT,DeformableTST,DUET,FiLM,FilterTS,FreTS,Fredformer,MICN,ModernTCN,NonstationaryTransformer,PDF,PIR,PathFormer,SCINet,SEMPO,SRSNet,SegRNN,SparseTSF,TimeBridge,TimePerceiver,TimeRecipe,xPatch"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task3_risk_adjust --category tslib_sota --ablation core_edgar \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task3_risk_adjust/tslib_sota/core_edgar --seed 42 \
    --no-verify-first --models CARD,CFPT,DeformableTST,DUET,FiLM,FilterTS,FreTS,Fredformer,MICN,ModernTCN,NonstationaryTransformer,PDF,PIR,PathFormer,SCINet,SEMPO,SRSNet,SegRNN,SparseTSF,TimeBridge,TimePerceiver,TimeRecipe,xPatch
echo "Done: $(date -Iseconds)"
