#!/usr/bin/env bash
#SBATCH --job-name=oracle_refresh
#SBATCH --account=christian.fisch
#SBATCH --partition=batch
#SBATCH --qos=iris-batch-long
#SBATCH --time=0-00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/work/projects/eint/logs/phase7_v735_cefu/oracle_refresh_%j.out
#SBATCH --error=/work/projects/eint/logs/phase7_v735_cefu/oracle_refresh_%j.err
#SBATCH --export=ALL

set -e

export CONDA_PREFIX="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
cd /mnt/aiongpfs/projects/eint/repo_root

INSIDER_PY="${CONDA_PREFIX}/bin/python3"

echo "============================================================"
echo "Oracle Refresh for V735 | Job ${SLURM_JOB_ID}"
echo "$(date -Iseconds) | $(hostname)"
echo "============================================================"

# Pull latest code
git pull --ff-only 2>/dev/null || echo "WARN: git pull failed, using local code"

# Run oracle refresh with --apply
"${INSIDER_PY}" scripts/refresh_v735_oracle.py --apply

# Commit & push if changes were made
if git diff --quiet src/narrative/block3/models/nf_adaptive_champion.py; then
    echo "No oracle changes needed"
else
    git add src/narrative/block3/models/nf_adaptive_champion.py
    git commit -m "V735: Auto-refresh oracle from baseline re-run results (job ${SLURM_JOB_ID})"
    git push
    echo "Oracle updated and pushed"
fi

echo "Done: $(date -Iseconds)"
