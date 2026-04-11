#!/bin/bash
set -euo pipefail

cd /home/users/npin/repo_root

echo "[v743-submit] validating cfisch connectivity"
ssh iris-cf 'echo [iris-cf] connected'

echo "[v743-submit] submit npin hopper jobs"
sbatch .slurm_scripts/v740_local/v743_investors_audit_np_hopper.sh
sbatch .slurm_scripts/v740_local/v743_minibench_np_hopper.sh

echo "[v743-submit] submit cfisch gpu jobs"
ssh iris-cf 'cd /work/projects/eint/repo_root && sbatch .slurm_scripts/v740_local/v743_shared112_investors_loop_cf_gpu.sh'
ssh iris-cf 'cd /work/projects/eint/repo_root && sbatch .slurm_scripts/v740_local/v743_shared112_binary_loop_cf_gpu.sh'
ssh iris-cf 'cd /work/projects/eint/repo_root && sbatch .slurm_scripts/v740_local/v743_shared112_funding_loop_cf_gpu.sh'
ssh iris-cf 'cd /work/projects/eint/repo_root && sbatch .slurm_scripts/v740_local/v743_keyslice_sweep_cf_gpu.sh'