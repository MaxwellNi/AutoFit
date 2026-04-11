#!/bin/bash

set -euo pipefail

echo "[mainline-wave2] validating cfisch connectivity"
ssh iris-cf 'echo "[iris-cf] connected" && hostname'

echo "[mainline-wave2] submit cfisch investors h7/h14/h30 delegate jobs"
ssh iris-cf 'sbatch /dev/stdin' < scripts/slurm/single_model_mainline/cf_mainline_shared112_investors_h7_delegate_gpu.sh
ssh iris-cf 'sbatch /dev/stdin' < scripts/slurm/single_model_mainline/cf_mainline_shared112_investors_h14_delegate_gpu.sh
ssh iris-cf 'sbatch /dev/stdin' < scripts/slurm/single_model_mainline/cf_mainline_shared112_investors_h30_delegate_gpu.sh

echo "[mainline-wave2] submit cfisch full-funding h7/h14/h30 delegate jobs"
ssh iris-cf 'sbatch /dev/stdin' < scripts/slurm/single_model_mainline/cf_mainline_shared112_fullfund_h7_delegate_gpu.sh
ssh iris-cf 'sbatch /dev/stdin' < scripts/slurm/single_model_mainline/cf_mainline_shared112_fullfund_h14_delegate_gpu.sh
ssh iris-cf 'sbatch /dev/stdin' < scripts/slurm/single_model_mainline/cf_mainline_shared112_fullfund_h30_delegate_gpu.sh