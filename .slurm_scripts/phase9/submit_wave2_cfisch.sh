#!/usr/bin/env bash
# Submit all Phase 9 Wave 2 scripts for cfisch
cd /mnt/aiongpfs/projects/eint/repo_root/.slurm_scripts/phase9

sbatch p9n_sta_t1_ce.sh
sbatch p9n_sta_t1_co.sh
sbatch p9n_sta_t1_ct.sh
sbatch p9n_sta_t1_fu.sh
sbatch p9n_sta_t2_ce.sh
sbatch p9n_sta_t2_co.sh
sbatch p9n_sta_t2_ct.sh
sbatch p9n_sta_t2_fu.sh
sbatch p9n_sta_t3_ce.sh
sbatch p9n_sta_t3_co.sh
sbatch p9n_sta_t3_fu.sh
sbatch p9r_af9_t1_ce.sh
sbatch p9r_af9_t1_ct.sh
sbatch p9r_af9_t1_fu.sh
sbatch p9r_af9_t2_ce.sh
sbatch p9r_af9_t2_ct.sh
sbatch p9r_af9_t2_fu.sh
sbatch p9r_af9_t3_ce.sh
sbatch p9r_af9_t3_fu.sh
sbatch p9r_fmB_t1_ct.sh
sbatch p9r_fmB_t1_fu.sh
sbatch p9r_fmB_t2_ct.sh
sbatch p9r_fmB_t2_fu.sh
sbatch p9r_fmB_t3_fu.sh
sbatch p9r_irN_t1_ce.sh
sbatch p9r_irN_t1_ct.sh
sbatch p9r_irN_t1_fu.sh
sbatch p9r_irN_t2_ce.sh
sbatch p9r_irN_t2_ct.sh
sbatch p9r_irN_t2_fu.sh
sbatch p9r_irN_t3_ce.sh
sbatch p9r_irN_t3_fu.sh
sbatch p9r_mlT_t1_ce.sh
sbatch p9r_mlT_t1_co.sh
sbatch p9r_mlT_t1_ct.sh
sbatch p9r_mlT_t1_fu.sh
sbatch p9r_mlT_t2_ce.sh
sbatch p9r_mlT_t2_co.sh
sbatch p9r_mlT_t2_ct.sh
sbatch p9r_mlT_t2_fu.sh
sbatch p9r_mlT_t3_ce.sh
sbatch p9r_mlT_t3_fu.sh

echo "Submitted 42 jobs for cfisch"
