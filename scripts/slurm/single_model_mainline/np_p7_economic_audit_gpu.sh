#!/bin/bash
#SBATCH --job-name=smm_p7_econ
#SBATCH --account=npin
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=150G
#SBATCH --time=0-06:00:00
#SBATCH --output=/work/projects/eint/logs/%x_%j.out
#SBATCH --error=/work/projects/eint/logs/%x_%j.err
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -euo pipefail
umask 002

mkdir -p /work/projects/eint/logs

_requeue_handler() {
  echo "[p7-economic-audit] caught USR1, requeueing ${SLURM_JOB_ID}" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap _requeue_handler USR1

REPO_ROOT=/work/projects/eint/repo_root
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3

export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export BLOCK3_CANONICAL_REPO_ROOT="$REPO_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"

# P7 经济学一致性审计:
# 对 mainline (三头联合模型) 和 DeepNPTS (单体专家) 分别 fit 三个 target，
# 在同一测试集上交叉检测：
# 1. Ghost Funding: investors=0 但 funding>0
# 2. Logical Inversion: P(funded)>=90% 但 funding 在底部 10%
#
# 覆盖: task1_outcome, core_only+core_edgar, h1/h7/h14/h30

"$INSIDER_PY" scripts/audit_economic_constraints.py \
  --benchmark DeepNPTS \
  --task task1_outcome \
  --ablations core_only core_edgar \
  --horizons 1 7 14 30 \
  --output-dir runs/benchmarks/single_model_mainline_localclear_20260420/p7_economic_audit
