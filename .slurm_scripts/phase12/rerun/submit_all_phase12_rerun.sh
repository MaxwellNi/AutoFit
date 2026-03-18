#!/bin/bash
# Phase 12: Submit ALL core_text/full re-run jobs
# Run AFTER text_embeddings.parquet is generated and verified
# Generated: 2026-03-12T02:20:27.971067

set -e
cd /work/projects/eint/repo_root

# Verify text embeddings exist
if [[ ! -f runs/text_embeddings/text_embeddings.parquet ]]; then
  echo "FATAL: text_embeddings.parquet not found!"; exit 1
fi

echo "Submitting 20 npin + 20 cfisch jobs..."

# npin jobs (direct sbatch)
sbatch .slurm_scripts/phase12/rerun/p12_deep_t1_ct.sh && echo "  ✓ p12_deep_t1_ct.sh"
sbatch .slurm_scripts/phase12/rerun/p12_foun_t1_ct.sh && echo "  ✓ p12_foun_t1_ct.sh"
sbatch .slurm_scripts/phase12/rerun/p12_irre_t1_ct.sh && echo "  ✓ p12_irre_t1_ct.sh"
sbatch .slurm_scripts/phase12/rerun/p12_stat_t1_ct.sh && echo "  ✓ p12_stat_t1_ct.sh"
sbatch .slurm_scripts/phase12/rerun/p12_deep_t1_fu.sh && echo "  ✓ p12_deep_t1_fu.sh"
sbatch .slurm_scripts/phase12/rerun/p12_foun_t1_fu.sh && echo "  ✓ p12_foun_t1_fu.sh"
sbatch .slurm_scripts/phase12/rerun/p12_irre_t1_fu.sh && echo "  ✓ p12_irre_t1_fu.sh"
sbatch .slurm_scripts/phase12/rerun/p12_stat_t1_fu.sh && echo "  ✓ p12_stat_t1_fu.sh"
sbatch .slurm_scripts/phase12/rerun/p12_deep_t2_ct.sh && echo "  ✓ p12_deep_t2_ct.sh"
sbatch .slurm_scripts/phase12/rerun/p12_foun_t2_ct.sh && echo "  ✓ p12_foun_t2_ct.sh"
sbatch .slurm_scripts/phase12/rerun/p12_irre_t2_ct.sh && echo "  ✓ p12_irre_t2_ct.sh"
sbatch .slurm_scripts/phase12/rerun/p12_stat_t2_ct.sh && echo "  ✓ p12_stat_t2_ct.sh"
sbatch .slurm_scripts/phase12/rerun/p12_deep_t2_fu.sh && echo "  ✓ p12_deep_t2_fu.sh"
sbatch .slurm_scripts/phase12/rerun/p12_foun_t2_fu.sh && echo "  ✓ p12_foun_t2_fu.sh"
sbatch .slurm_scripts/phase12/rerun/p12_irre_t2_fu.sh && echo "  ✓ p12_irre_t2_fu.sh"
sbatch .slurm_scripts/phase12/rerun/p12_stat_t2_fu.sh && echo "  ✓ p12_stat_t2_fu.sh"
sbatch .slurm_scripts/phase12/rerun/p12_deep_t3_fu.sh && echo "  ✓ p12_deep_t3_fu.sh"
sbatch .slurm_scripts/phase12/rerun/p12_foun_t3_fu.sh && echo "  ✓ p12_foun_t3_fu.sh"
sbatch .slurm_scripts/phase12/rerun/p12_irre_t3_fu.sh && echo "  ✓ p12_irre_t3_fu.sh"
sbatch .slurm_scripts/phase12/rerun/p12_stat_t3_fu.sh && echo "  ✓ p12_stat_t3_fu.sh"

# cfisch jobs (via ssh iris-cf)
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_tran_t1_ct.sh" && echo "  ✓ cf_p12_tran_t1_ct.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_tsli_t1_ct.sh" && echo "  ✓ cf_p12_tsli_t1_ct.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_ml_t_t1_ct.sh" && echo "  ✓ cf_p12_ml_t_t1_ct.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_af39_t1_ct.sh" && echo "  ✓ cf_p12_af39_t1_ct.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_tran_t1_fu.sh" && echo "  ✓ cf_p12_tran_t1_fu.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_tsli_t1_fu.sh" && echo "  ✓ cf_p12_tsli_t1_fu.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_ml_t_t1_fu.sh" && echo "  ✓ cf_p12_ml_t_t1_fu.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_af39_t1_fu.sh" && echo "  ✓ cf_p12_af39_t1_fu.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_tran_t2_ct.sh" && echo "  ✓ cf_p12_tran_t2_ct.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_tsli_t2_ct.sh" && echo "  ✓ cf_p12_tsli_t2_ct.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_ml_t_t2_ct.sh" && echo "  ✓ cf_p12_ml_t_t2_ct.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_af39_t2_ct.sh" && echo "  ✓ cf_p12_af39_t2_ct.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_tran_t2_fu.sh" && echo "  ✓ cf_p12_tran_t2_fu.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_tsli_t2_fu.sh" && echo "  ✓ cf_p12_tsli_t2_fu.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_ml_t_t2_fu.sh" && echo "  ✓ cf_p12_ml_t_t2_fu.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_af39_t2_fu.sh" && echo "  ✓ cf_p12_af39_t2_fu.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_tran_t3_fu.sh" && echo "  ✓ cf_p12_tran_t3_fu.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_tsli_t3_fu.sh" && echo "  ✓ cf_p12_tsli_t3_fu.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_ml_t_t3_fu.sh" && echo "  ✓ cf_p12_ml_t_t3_fu.sh"
ssh iris-cf "cd /work/projects/eint/repo_root && sbatch .slurm_scripts/phase12/rerun/cf_p12_af39_t3_fu.sh" && echo "  ✓ cf_p12_af39_t3_fu.sh"

echo "All 40 jobs submitted!"
