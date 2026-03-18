#!/usr/bin/env bash
#SBATCH --job-name=v739r2_t2_fu
#SBATCH --account=npin
#SBATCH --partition=hopper
#SBATCH --qos=besteffort
#SBATCH --time=2-00:00:00
#SBATCH --mem=500G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/projects/eint/logs/phase10/v739r2_t2_fu_%j.out
#SBATCH --error=/work/projects/eint/logs/phase10/v739r2_t2_fu_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -e
umask 002
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root

echo "V739 npin RETRY2 (500G) | Job ${SLURM_JOB_ID} on $(hostname) | hopper/besteffort"
echo "Task: task2_forecast | Abl: full | Models: AutoFitV739"
python3 scripts/run_block3_benchmark_shard.py \
  --task task2_forecast --category autofit --ablation full \
  --models AutoFitV739 --preset full \
  --output-dir runs/benchmarks/block3_phase10/v739
