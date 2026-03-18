#!/usr/bin/env bash
#SBATCH --job-name=cf_p9r3_fm_t3
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase9/cf_p9r3_fm_t3_%j.out
#SBATCH --error=/work/projects/eint/logs/phase9/cf_p9r3_fm_t3_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
# Direct env activation (no micromamba needed)
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"

cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"; exit 2
fi
echo "============================================================"
echo "Phase 9 Fair Benchmark — Chronos2 + TTM Fix | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${INSIDER_PY} -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

TASK=task3_risk_adjust
MODELS=Chronos2,TTM

# task3 has NO core_text ablation
for ABL in core_only core_edgar full; do
  echo ">>> ${TASK} | ${ABL} | Models: ${MODELS}"
  "${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
      --task "${TASK}" --category foundation --ablation "${ABL}" \
      --preset full --output-dir "runs/benchmarks/block3_phase9_fair/${TASK}/foundation/${ABL}" \
      --seed 42 --no-verify-first --models "${MODELS}" || true
  echo "<<< Done ${ABL}: $(date -Iseconds)"
done
echo "All done: $(date -Iseconds)"
