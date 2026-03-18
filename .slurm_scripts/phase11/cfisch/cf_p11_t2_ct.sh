#!/usr/bin/env bash
#SBATCH --job-name=cf_p11_t2_ct
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase11/cf_p11_t2_ct_%j.out
#SBATCH --error=/work/projects/eint/logs/phase11/cf_p11_t2_ct_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
cd /work/projects/eint/repo_root

if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing: ${INSIDER_PY}"; exit 2
fi
echo "============================================================"
echo "Phase 11 (cfisch) TSLib New Models | Job ${SLURM_JOB_ID} on $(hostname)"
echo "$(date -Iseconds) | Python: $(${INSIDER_PY} -V)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Git: $(git rev-parse --short HEAD)"
echo "============================================================"

echo "Task: task2_forecast | Cat: tslib_sota | Abl: core_text | Models: CFPT,DeformableTST,ModernTCN,PathFormer,SEMPO,TimePerceiver,TimeBridge,TQNet,PIR,CARD,PDF,TimeRecipe,DUET,SRSNet"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast --category tslib_sota --ablation core_text \
    --preset full --output-dir runs/benchmarks/block3_phase11/task2_forecast/tslib_sota/core_text --seed 42 \
    --no-verify-first --models CFPT,DeformableTST,ModernTCN,PathFormer,SEMPO,TimePerceiver,TimeBridge,TQNet,PIR,CARD,PDF,TimeRecipe,DUET,SRSNet
echo "Done: $(date -Iseconds)"
