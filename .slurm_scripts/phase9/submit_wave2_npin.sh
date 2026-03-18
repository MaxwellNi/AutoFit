#!/usr/bin/env bash
# Submit all Phase 9 Wave 2 scripts for npin
cd /mnt/aiongpfs/projects/eint/repo_root/.slurm_scripts/phase9

sbatch p9g_fmCT.sh
sbatch p9g_trCO.sh
sbatch p9g_trCT.sh
sbatch p9n_fmC_t1_ce.sh
sbatch p9n_fmC_t1_co.sh
sbatch p9n_fmC_t1_ct.sh
sbatch p9n_fmC_t1_fu.sh
sbatch p9n_fmC_t2_ce.sh
sbatch p9n_fmC_t2_co.sh
sbatch p9n_fmC_t2_ct.sh
sbatch p9n_fmC_t2_fu.sh
sbatch p9n_fmC_t3_ce.sh
sbatch p9n_fmC_t3_co.sh
sbatch p9n_fmC_t3_fu.sh
sbatch p9n_tpf_t1_ce.sh
sbatch p9n_tpf_t1_co.sh
sbatch p9n_tpf_t1_ct.sh
sbatch p9n_tpf_t1_fu.sh
sbatch p9n_tpf_t2_ce.sh
sbatch p9n_tpf_t2_co.sh
sbatch p9n_tpf_t2_ct.sh
sbatch p9n_tpf_t2_fu.sh
sbatch p9n_tpf_t3_ce.sh
sbatch p9n_tpf_t3_co.sh
sbatch p9n_tpf_t3_fu.sh
sbatch p9n_tsC_t1_ce.sh
sbatch p9n_tsC_t1_co.sh
sbatch p9n_tsC_t1_ct.sh
sbatch p9n_tsC_t1_fu.sh
sbatch p9n_tsC_t2_ce.sh
sbatch p9n_tsC_t2_co.sh
sbatch p9n_tsC_t2_ct.sh
sbatch p9n_tsC_t2_fu.sh
sbatch p9n_tsC_t3_ce.sh
sbatch p9n_tsC_t3_co.sh
sbatch p9n_tsC_t3_fu.sh

echo "Submitted 36 jobs for npin"
