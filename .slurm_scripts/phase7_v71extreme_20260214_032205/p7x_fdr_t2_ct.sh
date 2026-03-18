#!/usr/bin/env bash
#SBATCH --job-name=p7x_fdr_t2_ct
#SBATCH --account=yves.letraon
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:volta:1
#SBATCH --constraint=volta32
#SBATCH --output=/work/projects/eint/logs/phase7_v71extreme_20260214_032205/p7x_fdr_t2_ct_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_v71extreme_20260214_032205/p7x_fdr_t2_ct_%j.err
#SBATCH --export=ALL
#SBATCH --signal=B:USR1@120

set -e
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(which python3)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

echo "Task: task2_forecast | Category: foundation | Ablation: core_text"
echo "Models: Chronos2,Moirai2,TimesFM"
echo "Preset: full | Seed: 42"
echo "Output: runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/foundation_refs/task2_forecast/foundation/core_text"
echo "============================================================"

CMD=(
    python3 scripts/run_block3_benchmark_shard.py
    --task task2_forecast
    --category foundation
    --ablation core_text
    --preset full
    --output-dir runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/foundation_refs/task2_forecast/foundation/core_text
    --seed 42
    --no-verify-first
)
if [[ -n "Chronos2,Moirai2,TimesFM" ]]; then
    CMD+=(--models "Chronos2,Moirai2,TimesFM")
fi
if [[ -n "" ]]; then
    CMD+=(--model-kwargs-file "")
fi
"${CMD[@]}"

echo "Done: $(date -Iseconds)"
