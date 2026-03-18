#!/usr/bin/env bash
mkdir -p /work/projects/eint/logs/phase10
echo 'Submitting 11 Phase 10 npin jobs...'
sbatch /home/users/npin/repo_root/.slurm_scripts/phase10/npin/p10_tsD_t1_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase10/npin/p10_tsD_t1_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase10/npin/p10_tsD_t1_ct.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase10/npin/p10_tsD_t1_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase10/npin/p10_tsD_t2_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase10/npin/p10_tsD_t2_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase10/npin/p10_tsD_t2_ct.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase10/npin/p10_tsD_t2_fu.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase10/npin/p10_tsD_t3_ce.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase10/npin/p10_tsD_t3_co.sh
sbatch /home/users/npin/repo_root/.slurm_scripts/phase10/npin/p10_tsD_t3_fu.sh
echo 'All npin jobs submitted.'
