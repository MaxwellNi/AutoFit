#!/usr/bin/env bash
# Submit all Phase 11 TSLib new models jobs
set -e

mkdir -p /work/projects/eint/logs/phase11

sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/p11_tslib_new_t1_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/p11_tslib_new_t1_ct.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/p11_tslib_new_t1_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/p11_tslib_new_t1_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/p11_tslib_new_t2_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/p11_tslib_new_t2_ct.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/p11_tslib_new_t2_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/p11_tslib_new_t2_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/p11_tslib_new_t3_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/p11_tslib_new_t3_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/p11_tslib_new_t3_fu.sh

echo 'Submitted 11 Phase 11 jobs'
