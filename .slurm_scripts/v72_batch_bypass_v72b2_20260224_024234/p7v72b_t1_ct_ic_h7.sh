#!/usr/bin/env bash
#SBATCH --job-name=p7v72b_t1_ct_ic_h7
#SBATCH --account=yves.letraon
#SBATCH --partition=batch
#SBATCH --qos=iris-batch-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=112G
#SBATCH --cpus-per-task=28
#SBATCH --output=/work/projects/eint/logs/v72_batch_bypass_v72b2_20260224_024234/p7v72b_t1_ct_ic_h7_%j.out
#SBATCH --error=/work/projects/eint/logs/v72_batch_bypass_v72b2_20260224_024234/p7v72b_t1_ct_ic_h7_%j.err
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
# extra conservative cap for count lane
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "============================================================"
echo "Job ${SLURM_JOB_ID} on $(hostname) — $(date -Iseconds)"
echo "Task=task1_outcome Ablation=core_text Target=investors_count Horizon=7"
echo "ResourceClass=L-bypass-count Partition=batch QOS=iris-batch-long Mem=112G CPUs=28"
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
export B3_MEMORY_CLASS="L"
export B3_RESOURCE_PROFILE_ID="L-bypass-count:batch:iris-batch-long:112G:28"

"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py   --task task1_outcome   --category autofit   --ablation core_text   --models AutoFitV72   --target-filter investors_count   --horizons-filter 7   --preset full   --output-dir runs/benchmarks/block3_20260203_225620_phase7_v72_completion/task1_outcome/autofit/core_text/investors_count/h7   --seed 42   --no-verify-first   --enable-global-dedup   --global-dedup-bench-glob block3_20260203_225620*
