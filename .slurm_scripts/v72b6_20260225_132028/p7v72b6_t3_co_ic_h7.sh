#!/usr/bin/env bash
#SBATCH --job-name=p7v72b6_t3_co_ic_h7
#SBATCH --account=yves.letraon
#SBATCH --partition=batch
#SBATCH --qos=iris-batch-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=112G
#SBATCH --cpus-per-task=16
#SBATCH --output=/work/projects/eint/logs/v72b6_20260225_132028/p7v72b6_t3_co_ic_h7_%j.out
#SBATCH --error=/work/projects/eint/logs/v72b6_20260225_132028/p7v72b6_t3_co_ic_h7_%j.err
#SBATCH --export=ALL
set -euo pipefail
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
cd /home/users/npin/repo_root
INSIDER_PY="${CONDA_PREFIX}/bin/python3"
if [[ ! -x "${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing or non-executable: ${INSIDER_PY}"; exit 2
fi
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export B3_MEMORY_CLASS=L
export B3_RESOURCE_PROFILE_ID=L:batch:iris-batch-long:112G:16
"${INSIDER_PY}" scripts/assert_block3_execution_contract.py --entrypoint "slurm:${SLURM_JOB_NAME}"
"${INSIDER_PY}" scripts/run_block3_benchmark_shard.py   --task task3_risk_adjust   --category autofit   --ablation core_only   --models AutoFitV72   --target-filter investors_count   --horizons-filter 7   --preset full   --output-dir runs/benchmarks/block3_20260203_225620_phase7_v72_batch_bypass_v72b6_20260225_132028/task3_risk_adjust/autofit/core_only/investors_count/h7   --seed 42   --no-verify-first   --enable-global-dedup   --global-dedup-bench-glob block3_20260203_225620*
