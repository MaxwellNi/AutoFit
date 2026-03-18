#!/usr/bin/env bash
#SBATCH --job-name=p9_tsA_t2_ce
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase9/p9_tsA_t2_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase9/p9_tsA_t2_ce_%j.err
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

echo "Task: task2_forecast | Cat: tslib_sota | Abl: core_edgar | Models: TimeFilter,WPMixer,MultiPatchFormer,MSGNet,PAttn,MambaSimple,Crossformer"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category tslib_sota --ablation core_edgar \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task2_forecast/tslib_sota/core_edgar --seed 42 \
    --no-verify-first --models TimeFilter,WPMixer,MultiPatchFormer,MSGNet,PAttn,MambaSimple,Crossformer
echo "Done: $(date -Iseconds)"
