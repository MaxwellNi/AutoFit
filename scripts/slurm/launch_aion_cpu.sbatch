#!/bin/bash
#SBATCH --job-name=b11_v2_aion_cpu
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/work/projects/eint/logs/b11_v2_aion_%A.out

set -euo pipefail

cd /home/users/npin/repo_root
source /work/projects/eint/envs/.micromamba/etc/profile.d/micromamba.sh
micromamba activate insider

STAMP="$(cat runs/backups/current_audit_stamp.txt)"
ORCH_DIR="runs/orchestrator/${STAMP}"

python scripts/collect_results.py --runs_dir runs/benchmarks --output_dir "${ORCH_DIR}/collected"
python scripts/make_paper_tables.py --bench_root runs/benchmarks --output_dir "${ORCH_DIR}/paper_tables"
