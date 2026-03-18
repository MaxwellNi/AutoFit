#!/usr/bin/env bash
# Submit all Phase 9 Wave 3 scripts for npin
cd /mnt/aiongpfs/projects/eint/repo_root/.slurm_scripts/phase9

sbatch w3l_fmB_t1_ct.sh
sbatch w3l_fmB_t1_fu.sh
sbatch w3l_fmB_t2_ct.sh
sbatch w3l_fmB_t2_fu.sh
sbatch w3l_fmB_t3_fu.sh
sbatch w3l_irN_t1_ce.sh
sbatch w3l_irN_t1_ct.sh
sbatch w3l_irN_t1_fu.sh
sbatch w3l_irN_t2_ce.sh
sbatch w3l_irN_t2_ct.sh
sbatch w3l_irN_t2_fu.sh
sbatch w3l_irN_t3_ce.sh
sbatch w3l_irN_t3_fu.sh

echo "Submitted 13 jobs for npin"
