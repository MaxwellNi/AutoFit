#!/usr/bin/env bash
# Wrapper to launch benchmark inside tmux with proper conda setup
set -euo pipefail

CONDA_BASE="/home/pni/miniforge3"
REPO="/home/pni/projects/repo_root"

# Initialize conda for this shell
eval "$(${CONDA_BASE}/bin/conda shell.bash hook)"
conda activate insider

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"
echo "Starting benchmark..."

cd "${REPO}"
exec bash scripts/run_block3_full_4090.sh
