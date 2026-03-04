#!/usr/bin/env bash
#SBATCH --job-name=fc_t1_ct
#SBATCH --account=christian.fisch
#SBATCH --partition=l40s
#SBATCH --qos=iris-snt
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase7_fused_champion/fc_t1_ct_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_fused_champion/fc_t1_ct_%j.err
#SBATCH --export=ALL

set -e

export CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /mnt/aiongpfs/projects/eint/repo_root

INSIDER_PY="${CONDA_PREFIX}/bin/python3"
[[ -x "${INSIDER_PY}" ]] || { echo "FATAL: insider python missing"; exit 2; }

echo "============================================================"
echo "FusedChampion V7.3.2 | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Partition: ${SLURM_JOB_PARTITION} | QOS: iris-snt"
echo "Python: $(which python3) | $(python3 -V)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' 2>/dev/null || echo N/A)"
echo "============================================================"

python3 - <<'PY'
import sys, torch
if "insider" not in sys.executable:
    raise SystemExit("FATAL: not insider python")
if sys.version_info < (3, 11):
    raise SystemExit("FATAL: python >= 3.11 required")
if not torch.cuda.is_available():
    raise SystemExit("FATAL: GPU required")
PY

${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"

echo "Task: task1_outcome | Ablation: core_text | Model: FusedChampion"
echo "Output: runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_text"
echo "============================================================"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome \
    --category autofit \
    --ablation core_text \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_text \
    --seed 42 \
    --no-verify-first \
    --models FusedChampion

echo "Done: $(date -Iseconds)"
