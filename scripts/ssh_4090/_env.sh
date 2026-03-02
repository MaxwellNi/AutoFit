#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/pni/projects/repo_root}"
cd "${REPO_ROOT}"

if [ -x "$HOME/.local/bin/micromamba" ]; then
  eval "$("$HOME/.local/bin/micromamba" shell hook -s bash)"
  micromamba activate insider
else
  source ~/.bashrc
  conda activate insider
fi

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
