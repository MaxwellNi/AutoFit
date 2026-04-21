#!/bin/bash
#SBATCH --job-name=v740_inv_t2occ
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --qos=normal
#SBATCH --output=/work/projects/eint/logs/phase15/v740_inv_t2occ_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/v740_inv_t2occ_%j.err
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -euo pipefail

INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
REPO_ROOT=/home/users/npin/repo_root
RUN_ROOT="$REPO_ROOT/runs/benchmarks/v740_localclear_20260404"
JOB_ROOT="$RUN_ROOT/v740_investors_task2_h1_occurrence_probe_20260404"

mkdir -p /work/projects/eint/logs/phase15
mkdir -p "$JOB_ROOT"
cd "$REPO_ROOT"

run_variant() {
	local variant="$1"
	shift
	local out_dir="$JOB_ROOT/$variant"
	local summary_md="$JOB_ROOT/${variant}.md"
	local surface_json="$JOB_ROOT/${variant}_surface.json"

	echo "[v740-inv-t2occ] variant=$variant start $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
	"$INSIDER_PY" scripts/run_v740_shared112_champion_loop.py \
		--profile quick \
		--task task2_forecast \
		--target investors_count \
		--horizon 1 \
		--models v740_alpha,incumbent \
		--output-dir "$out_dir" \
		--summary-md "$summary_md" \
		--surface-json "$surface_json" \
		--skip-existing \
		"$@"
	echo "[v740-inv-t2occ] variant=$variant done $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
}

run_variant baseline_task2_occ
run_variant no_sparsity_gate --disable-count-sparsity-gate
run_variant route_off_nojump --disable-target-routing --disable-count-jump

echo "[v740-inv-t2occ] all variants complete $(date -u +'%Y-%m-%dT%H:%M:%SZ')"