#!/usr/bin/env bash
#SBATCH --job-name=p7v72k_t3_ce_fru_h30
#SBATCH --account=yves.letraon
#SBATCH --partition=bigmem
#SBATCH --qos=iris-bigmem-long
#SBATCH --time=3-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64
#SBATCH --output=/work/projects/eint/logs/v72_completion_20260224_022433/p7v72k_t3_ce_fru_h30_%j.out
#SBATCH --error=/work/projects/eint/logs/v72_completion_20260224_022433/p7v72k_t3_ce_fru_h30_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -euo pipefail
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /mnt/aiongpfs/projects/eint/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing or non-executable: ${INSIDER_PY}"
  exit 2
fi
echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Task=task3_risk_adjust Ablation=core_edgar Target=funding_raised_usd Horizon=30"
echo "ResourceClass=XL Partition=bigmem QOS=iris-bigmem-long Mem=512G CPUs=64"
echo "Python: $(which python3)"
python3 -V
python3 - <<'PY'
import sys
print("sys.executable:", sys.executable)
if "insider" not in sys.executable:
    raise SystemExit("FATAL: runtime python is not insider")
if sys.version_info < (3, 11):
    raise SystemExit(
        f"FATAL: insider python must be >=3.11, got {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
PY
"${INSIDER_PY}" scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"
export B3_MEMORY_CLASS="XL"
export B3_RESOURCE_PROFILE_ID="XL:bigmem:iris-bigmem-long:512G:64"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \
  --task task3_risk_adjust \
  --category autofit \
  --ablation core_edgar \
  --models AutoFitV72 \
  --target-filter funding_raised_usd \
  --horizons-filter 30 \
  --preset full \
  --output-dir runs/benchmarks/block3_20260203_225620_phase7_v72_completion/task3_risk_adjust/autofit/core_edgar/funding_raised_usd/h30 \
  --seed 42 \
  --no-verify-first \
  --enable-global-dedup \
  --global-dedup-bench-glob block3_20260203_225620*
