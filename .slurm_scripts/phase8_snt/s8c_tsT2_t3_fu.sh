#!/usr/bin/env bash
#SBATCH --job-name=s8c_tsT2_t3_fu
#SBATCH --account=yves.letraon
#SBATCH --partition=gpu
#SBATCH --qos=iris-snt-long
#SBATCH --time=3-00:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:volta:1
#SBATCH --output=/work/projects/eint/logs/phase8_snt/s8c_tsT2_t3_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase8_snt/s8c_tsT2_t3_fu_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"; exit 2
fi
echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(python3 -V) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

echo "Task: task3_risk_adjust | Cat: tslib_sota | Abl: full | Models: FreTS,MICN,SegRNN,NonstationaryTransformer,FiLM,SCINet"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task3_risk_adjust --category tslib_sota --ablation full \
    --preset full --output-dir runs/benchmarks/block3_20260203_225620_phase7/task3_risk_adjust/tslib_sota/full --seed 42 \
    --no-verify-first --models FreTS,MICN,SegRNN,NonstationaryTransformer,FiLM,SCINet
echo "Done: $(date -Iseconds)"
