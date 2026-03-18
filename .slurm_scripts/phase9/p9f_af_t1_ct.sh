#!/usr/bin/env bash
#SBATCH --job-name=p9f_af_t1_ct
#SBATCH --partition=bigmem
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=28
#SBATCH --output=/home/users/npin/repo_root/slurm_logs/phase9/p9f_af_t1_ct_%j.out
#SBATCH --error=/home/users/npin/repo_root/slurm_logs/phase9/p9f_af_t1_ct_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
INSIDER_PY="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3"
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"; exit 2
fi
echo "============================================================"
echo "Phase 9 FIXED AutoFit | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${INSIDER_PY} -V)"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

# V735 first (single model, fastest)
echo ">>> V735: task1_outcome | core_text | $(date -Iseconds)"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category autofit --ablation core_text \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/core_text --seed 42 \
    --no-verify-first --models AutoFitV735
echo ">>> V735 done: $(date -Iseconds)"

# V734 (3-model ensemble with fixed stack_k)
echo ">>> V734: task1_outcome | core_text | $(date -Iseconds)"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category autofit --ablation core_text \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/core_text --seed 42 \
    --no-verify-first --models AutoFitV734
echo ">>> V734 done: $(date -Iseconds)"

# V736 (3-model stacking ensemble)
echo ">>> V736: task1_outcome | core_text | $(date -Iseconds)"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task1_outcome --category autofit --ablation core_text \
    --preset full --output-dir runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/core_text --seed 42 \
    --no-verify-first --models AutoFitV736
echo ">>> V736 done: $(date -Iseconds)"

echo "ALL DONE: $(date -Iseconds)"
