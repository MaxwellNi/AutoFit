#!/bin/bash
#SBATCH --job-name=v740_inv_gate_cf
#SBATCH --account=christian.fisch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --qos=besteffort
#SBATCH --output=/work/projects/eint/logs/phase15/v740_inv_gate_cf_%j.out
#SBATCH --error=/work/projects/eint/logs/phase15/v740_inv_gate_cf_%j.err
#SBATCH --signal=USR1@120
#SBATCH --requeue

set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib:${LD_LIBRARY_PATH:-}"
INSIDER_PY=/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3
REPO_ROOT=/home/users/npin/repo_root
RUN_ROOT="$REPO_ROOT/runs/benchmarks/v740_localclear_cfisch_20260404"
JOB_ROOT="$RUN_ROOT/v740_investors_h1_count_gate_cfisch_20260404"

mkdir -p /work/projects/eint/logs/phase15
mkdir -p "$JOB_ROOT"
cd "$REPO_ROOT"

run_variant() {
	local variant="$1"
	shift
	local out_dir="$JOB_ROOT/$variant"
	local summary_md="$JOB_ROOT/${variant}.md"
	local surface_json="$JOB_ROOT/${variant}_surface.json"

	echo "[v740-inv-gate-cf] variant=$variant start $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
	"$INSIDER_PY" scripts/run_v740_shared112_champion_loop.py \
		--profile quick \
		--target investors_count \
		--horizon 1 \
		--models v740_alpha,incumbent \
		--output-dir "$out_dir" \
		--summary-md "$summary_md" \
		--surface-json "$surface_json" \
		--skip-existing \
		"$@"
	echo "[v740-inv-gate-cf] variant=$variant done $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
}

run_variant baseline
run_variant nojump --disable-count-jump
run_variant noanchor --disable-count-anchor
run_variant anchor_high_nojump --disable-count-jump --count-anchor-strength 1.20
run_variant route_off_nojump --disable-target-routing --disable-count-jump

echo "[v740-inv-gate-cf] all variants complete $(date -u +'%Y-%m-%dT%H:%M:%SZ')"