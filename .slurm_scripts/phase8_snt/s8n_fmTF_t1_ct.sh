#!/usr/bin/env bash
#SBATCH --job-name=s8n_fmTF_t1_ct
#SBATCH --account=yves.letraon
#SBATCH --partition=gpu
#SBATCH --qos=iris-snt-long
#SBATCH --time=1-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:volta:1
#SBATCH --output=/work/projects/eint/logs/phase8_snt/s8n_fmTF_t1_ct_%j.out
#SBATCH --error=/work/projects/eint/logs/phase8_snt/s8n_fmTF_t1_ct_%j.err
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

echo "Task: task1_outcome | Cat: foundation | Abl: core_text | Models: TimesFM"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category foundation --ablation core_text \
    --preset full --output-dir runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/core_text --seed 42 \
    --no-verify-first --models TimesFM
echo "Done: $(date -Iseconds)"
