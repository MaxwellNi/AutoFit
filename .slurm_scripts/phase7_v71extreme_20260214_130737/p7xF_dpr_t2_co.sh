#!/usr/bin/env bash
#SBATCH --job-name=p7xF_dpr_t2_co
#SBATCH --account=yves.letraon
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:volta:1
#SBATCH --constraint=volta32
#SBATCH --output=/work/projects/eint/logs/phase7_v71extreme_20260214_130737/p7xF_dpr_t2_co_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_v71extreme_20260214_130737/p7xF_dpr_t2_co_%j.err
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

echo "Task: task2_forecast | Category: deep_classical | Ablation: core_only"
echo "Models: NBEATS,NHITS,TFT,DeepAR"
echo "Preset: full | Seed: 42"
echo "Output: runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737/full/deep_refs/task2_forecast/deep_classical/core_only"
echo "============================================================"

CMD=(
    python3 scripts/run_block3_benchmark_shard.py
    --task task2_forecast
    --category deep_classical
    --ablation core_only
    --preset full
    --output-dir runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737/full/deep_refs/task2_forecast/deep_classical/core_only
    --seed 42
    --no-verify-first
)
if [[ -n "NBEATS,NHITS,TFT,DeepAR" ]]; then
    CMD+=(--models "NBEATS,NHITS,TFT,DeepAR")
fi
if [[ -n "" ]]; then
    CMD+=(--model-kwargs-file "")
fi
"${CMD[@]}"

echo "Done: $(date -Iseconds)"
