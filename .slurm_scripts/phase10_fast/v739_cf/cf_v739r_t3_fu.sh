#!/usr/bin/env bash
#SBATCH --job-name=cf_v739r_t3_fu
#SBATCH --account=christian.fisch
#SBATCH --partition=l40s
#SBATCH --qos=iris-snt
#SBATCH --time=2-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase10/cf_v739r_t3_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/cf_v739r_t3_fu_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/home/users/npin/.cache/huggingface"
cd /work/projects/eint/repo_root
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"; exit 2
fi
echo "============================================================"
echo "V739 cfisch RETRY (300G) | Job ${SLURM_JOB_ID} on $(hostname) | l40s/iris-snt"
echo "$(date -Iseconds) | Python: $(${INSIDER_PY} -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "============================================================"

echo "Task: task3_risk_adjust | Abl: full | Models: AutoFitV739"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task3_risk_adjust --category autofit --ablation full \
    --preset full --output-dir runs/benchmarks/block3_phase10/v739/task3_risk_adjust/autofit/full --seed 42 \
    --no-verify-first --models AutoFitV739
echo "Done: $(date -Iseconds)"
