#!/usr/bin/env bash
#SBATCH --job-name=v731_t3_ce
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=378G
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:volta:1
#SBATCH --output=/work/projects/eint/logs/phase7_v731/v731_t3_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_v731/v731_t3_ce_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e

# Direct env activation (no micromamba required)
export CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /mnt/aiongpfs/projects/eint/repo_root

INSIDER_PY="${CONDA_PREFIX}/bin/python3"
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing or non-executable: ${INSIDER_PY}"
  exit 2
fi

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(which python3)"
echo "PythonV: $(python3 -V)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA:   $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>/dev/null || echo 'N/A')"
echo "============================================================"
python3 - <<'PY'
import sys, torch
print("sys.executable:", sys.executable)
if "insider" not in sys.executable:
    raise SystemExit("FATAL: runtime python is not insider")
if sys.version_info < (3, 11):
    raise SystemExit(f"FATAL: python >={3}.{11} required, got {sys.version_info}")
if not torch.cuda.is_available():
    raise SystemExit("FATAL: GPU required but torch.cuda.is_available()=False")
PY
${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"

echo "Task: task3_risk_adjust | Category: autofit | Ablation: core_edgar"
echo "Models: AutoFitV731"
echo "Preset: full | Seed: 42"
echo "Output: runs/benchmarks/block3_20260203_225620_phase7/task3_risk_adjust/autofit/core_edgar"
echo "============================================================"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task3_risk_adjust \
    --category autofit \
    --ablation core_edgar \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task3_risk_adjust/autofit/core_edgar \
    --seed 42 \
    --no-verify-first \
    --models AutoFitV731

echo "Done: $(date -Iseconds)"
