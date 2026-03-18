#!/usr/bin/env bash
# Submit all Phase 9 Wave 3 scripts for cfisch
cd /mnt/aiongpfs/projects/eint/repo_root/.slurm_scripts/phase9

sbatch w3r_af9_t1_ct.sh
sbatch w3r_af9_t1_fu.sh
sbatch w3r_af9_t2_ct.sh
sbatch w3r_af9_t2_fu.sh
sbatch w3r_af9_t3_fu.sh
sbatch w3r_mlT_t1_ct.sh
sbatch w3r_mlT_t1_fu.sh
sbatch w3r_mlT_t2_ct.sh
sbatch w3r_mlT_t2_fu.sh
sbatch w3r_mlT_t3_fu.sh
sbatch w3r_sta_t1_ce.sh
sbatch w3r_sta_t1_co.sh
sbatch w3r_sta_t1_ct.sh
sbatch w3r_sta_t1_fu.sh
sbatch w3r_sta_t2_ce.sh
sbatch w3r_sta_t2_co.sh
sbatch w3r_sta_t2_ct.sh
sbatch w3r_sta_t2_fu.sh
sbatch w3r_sta_t3_ce.sh
sbatch w3r_sta_t3_co.sh
sbatch w3r_sta_t3_fu.sh

echo "Submitted 21 jobs for cfisch"
