#!/usr/bin/env bash
set -euo pipefail

cd /home/users/npin/repo_root

echo "[mainline-submit] validating cfisch connectivity"
ssh -o BatchMode=yes -o ConnectTimeout=10 iris-cf 'echo [iris-cf] connected && hostname'

echo "[mainline-submit] submit npin native gate jobs"
sbatch scripts/slurm/single_model_mainline/np_mainline_shared112_investors_h1_native_gpu.sh
sbatch scripts/slurm/single_model_mainline/np_mainline_shared112_binary_guard_native_gpu.sh

echo "[mainline-submit] submit cfisch delegate gate jobs"
ssh iris-cf 'sbatch /dev/stdin' < scripts/slurm/single_model_mainline/cf_mainline_shared112_investors_h1_delegate_gpu.sh
ssh iris-cf 'sbatch /dev/stdin' < scripts/slurm/single_model_mainline/cf_mainline_shared112_binary_guard_delegate_gpu.sh