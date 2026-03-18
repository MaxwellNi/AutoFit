#!/usr/bin/env bash
# Submit all cfisch jobs for Phase 11 + gap-fill
# MUST be run from cfisch account: ssh iris-cf
set -e

mkdir -p /work/projects/eint/logs/phase11

sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p11_t1_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p11_t1_ct.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p11_t1_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p11_t1_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p11_t2_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p11_t2_ct.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p11_t2_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p11_t2_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p11_t3_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p11_t3_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p11_t3_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p9g_t1_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p9g_t1_ct.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p9g_t1_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p9g_t1_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p9g_t2_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p9g_t2_ct.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p9g_t2_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p9g_t2_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p9g_t3_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p9g_t3_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_p9g_t3_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_nbglm_t1_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_nbglm_t1_ct.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_nbglm_t1_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_nbglm_t1_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_nbglm_t2_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_nbglm_t2_ct.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_nbglm_t2_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_nbglm_t2_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_nbglm_t3_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_nbglm_t3_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase11/cfisch/cf_nbglm_t3_fu.sh

echo 'Submitted 33 cfisch jobs'
