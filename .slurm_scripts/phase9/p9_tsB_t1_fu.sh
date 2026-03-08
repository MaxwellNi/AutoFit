#!/usr/bin/env bash
#SBATCH --job-name=p9_tsB_t1_fu
#SBATCH --account=cfisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=320G
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase9/p9_tsB_t1_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase9/p9_tsB_t1_fu_%j.err
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
echo "Phase 9 Fair Benchmark | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(python3 -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

echo "Task: task1_outcome | Cat: tslib_sota | Abl: full | Models: Koopa,FreTS,MICN,SegRNN,NonstationaryTransformer,FiLM,SCINet"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category tslib_sota --ablation full \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/tslib_sota/full --seed 42 \
    --no-verify-first --models Koopa,FreTS,MICN,SegRNN,NonstationaryTransformer,FiLM,SCINet
echo "Done: $(date -Iseconds)"
