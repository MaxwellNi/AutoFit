#!/usr/bin/env bash
#SBATCH --job-name=sat_fmTF_t2_fu
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=1-12:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:volta:1
#SBATCH --output=/work/projects/eint/logs/phase8_sat/sat_fmTF_t2_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase8_sat/sat_fmTF_t2_fu_%j.err
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

echo "Task: task2_forecast | Cat: foundation | Abl: full | Models: TimesFM"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category foundation --ablation full \
    --preset full --output-dir runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/foundation/full --seed 42 \
    --no-verify-first --models TimesFM
echo "Done: $(date -Iseconds)"
