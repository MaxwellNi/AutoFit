#!/usr/bin/env bash
# Phase 15: Submit all new TSLib model benchmark jobs
# NPIN jobs (core_only + core_edgar) — submit from npin login
# CFISCH jobs (core_text + full) — submit via ssh iris-cf

set -e
echo "=== Phase 15: TSLib New Model Benchmark ==="
echo "23 models × 3 tasks × 4 ablations = 276 combos"
echo ""

# NPIN jobs (gpu, 256G)
echo "--- Submitting NPIN jobs (core_only + core_edgar) ---"
sbatch .slurm_scripts/phase15/p15_new_t1_co.sh && echo "  Submitted p15_new_t1_co"
sbatch .slurm_scripts/phase15/p15_new_t2_co.sh && echo "  Submitted p15_new_t2_co"
sbatch .slurm_scripts/phase15/p15_new_t3_co.sh && echo "  Submitted p15_new_t3_co"
sbatch .slurm_scripts/phase15/p15_new_t1_ce.sh && echo "  Submitted p15_new_t1_ce"
sbatch .slurm_scripts/phase15/p15_new_t2_ce.sh && echo "  Submitted p15_new_t2_ce"
sbatch .slurm_scripts/phase15/p15_new_t3_ce.sh && echo "  Submitted p15_new_t3_ce"

echo ""
echo "--- CFISCH jobs (core_text + full) ---"
echo "Run from cfisch login (ssh iris-cf):"
echo "  sbatch /work/projects/eint/repo_root/.slurm_scripts/phase15/cfisch/cf_p15_new_t1_ct.sh"
echo "  sbatch /work/projects/eint/repo_root/.slurm_scripts/phase15/cfisch/cf_p15_new_t2_ct.sh"
echo "  sbatch /work/projects/eint/repo_root/.slurm_scripts/phase15/cfisch/cf_p15_new_t3_ct.sh"
echo "  sbatch /work/projects/eint/repo_root/.slurm_scripts/phase15/cfisch/cf_p15_new_t1_fu.sh"
echo "  sbatch /work/projects/eint/repo_root/.slurm_scripts/phase15/cfisch/cf_p15_new_t2_fu.sh"
echo "  sbatch /work/projects/eint/repo_root/.slurm_scripts/phase15/cfisch/cf_p15_new_t3_fu.sh"

echo ""
echo "=== Submission complete ==="
squeue -u npin -o "%.8i %.20j %.4t %.10M %.20R"
