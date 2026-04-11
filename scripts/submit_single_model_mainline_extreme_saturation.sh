#!/usr/bin/env bash
set -euo pipefail

cd /home/users/npin/repo_root

echo "[mainline-extreme] validating cfisch connectivity"
ssh -o BatchMode=yes -o ConnectTimeout=10 iris-cf 'echo [iris-cf] connected && hostname'

echo "[mainline-extreme] submit cfisch high-priority mainline jobs"
ssh iris-cf 'cd /work/projects/eint/repo_root && sbatch scripts/slurm/single_model_mainline/cf_mainline_shared112_investors_full_gpu.sh'
ssh iris-cf 'cd /work/projects/eint/repo_root && sbatch scripts/slurm/single_model_mainline/cf_mainline_shared112_binary_full_gpu.sh'
ssh iris-cf 'cd /work/projects/eint/repo_root && sbatch scripts/slurm/single_model_mainline/cf_mainline_shared112_funding_full_gpu.sh'
ssh iris-cf 'cd /work/projects/eint/repo_root && sbatch scripts/slurm/single_model_mainline/cf_mainline_task2_investors_audit_gpu.sh'

echo "[mainline-extreme] submit cfisch audit-overflow mainline jobs"
ssh iris-cf 'cd /work/projects/eint/repo_root && sbatch scripts/slurm/single_model_mainline/cf_mainline_task1_binary_audit_gpu.sh'
ssh iris-cf 'cd /work/projects/eint/repo_root && sbatch scripts/slurm/single_model_mainline/cf_mainline_task1_funding_audit_gpu.sh'