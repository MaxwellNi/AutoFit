#!/usr/bin/env bash
#SBATCH --job-name=p8_tsT1_t2_ce
#SBATCH --account=yves.letraon
#SBATCH --partition=gpu
#SBATCH --qos=iris-gpu-long
#SBATCH --time=3-00:00:00
#SBATCH --mem=189G
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:volta:1
#SBATCH --output=/work/projects/eint/logs/phase8/p8_tsT1_t2_ce_%j.out
#SBATCH --error=/work/projects/eint/logs/phase8/p8_tsT1_t2_ce_%j.err
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
  echo "FATAL: insider python missing: ${INSIDER_PY}"
  exit 2
fi

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Python: $(which python3) — $(python3 -V)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"
python3 - <<'PY'
import sys
if "insider" not in sys.executable:
    raise SystemExit("FATAL: runtime python is not insider")
if sys.version_info < (3, 11):
    raise SystemExit(f"FATAL: need >=3.11, got {sys.version}")
PY
${INSIDER_PY} scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"

echo "Task: task2_forecast | Category: tslib_sota | Ablation: core_edgar"
echo "Models: TimeFilter,WPMixer,MultiPatchFormer,MSGNet,PAttn,MambaSimple,Crossformer,Koopa"
echo "Preset: full | Seed: 42"
echo "Output: runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/tslib_sota/core_edgar"
echo "============================================================"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
    --task task2_forecast \
    --category tslib_sota \
    --ablation core_edgar \
    --preset full \
    --output-dir runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/tslib_sota/core_edgar \
    --seed 42 \
    --no-verify-first \
    --models TimeFilter,WPMixer,MultiPatchFormer,MSGNet,PAttn,MambaSimple,Crossformer,Koopa

echo "Done: $(date -Iseconds)"
