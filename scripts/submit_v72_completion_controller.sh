#!/usr/bin/env bash
# ============================================================================
# Submit AutoFitV72 completion jobs from missing-key manifest.
#
# Strategy:
#   - Read docs/benchmarks/block3_truth_pack/missing_key_manifest.csv
#   - Collapse to unique (task, ablation) shards
#   - Submit only missing shards in priority order
#
# Usage:
#   python scripts/v72_coverage_guard.py
#   bash scripts/submit_v72_completion_controller.sh --dry-run
#   bash scripts/submit_v72_completion_controller.sh --submit
# ============================================================================

set -euo pipefail

MODE="dry-run"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LIMIT=0
MANIFEST="docs/benchmarks/block3_truth_pack/missing_key_manifest.csv"
STAMP="20260203_225620"
REPO="/home/users/npin/repo_root"
SLURM_DIR="${REPO}/.slurm_scripts/v72_completion_${RUN_TAG}"
LOG_DIR="/work/projects/eint/logs/v72_completion_${RUN_TAG}"
OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7_v72_completion_${RUN_TAG}"
ACCOUNT="${SLURM_ACCOUNT:-yves.letraon}"

for arg in "$@"; do
  case "$arg" in
    --dry-run) MODE="dry-run" ;;
    --submit) MODE="submit" ;;
    --run-tag=*) RUN_TAG="${arg#*=}" ;;
    --limit=*) LIMIT="${arg#*=}" ;;
    --manifest=*) MANIFEST="${arg#*=}" ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

SLURM_DIR="${REPO}/.slurm_scripts/v72_completion_${RUN_TAG}"
LOG_DIR="/work/projects/eint/logs/v72_completion_${RUN_TAG}"
OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7_v72_completion_${RUN_TAG}"
mkdir -p "${SLURM_DIR}" "${LOG_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "FATAL: manifest not found: ${MANIFEST}"
  echo "Run: python scripts/v72_coverage_guard.py"
  exit 2
fi

map_task() {
  case "$1" in
    task1_outcome) echo "t1" ;;
    task2_forecast) echo "t2" ;;
    task3_risk_adjust) echo "t3" ;;
    *) echo "tx" ;;
  esac
}

map_abl() {
  case "$1" in
    core_only) echo "co" ;;
    core_text) echo "ct" ;;
    core_edgar) echo "ce" ;;
    full) echo "fu" ;;
    *) echo "xx" ;;
  esac
}

# Build unique shard list in priority order.
mapfile -t SHARDS < <(
  awk -F',' '
    NR==1 {next}
    {
      pr=$5; task=$1; abl=$2;
      key=task":"abl;
      if (!(key in minp) || pr < minp[key]) minp[key]=pr;
    }
    END {
      for (k in minp) {
        print minp[k] "," k;
      }
    }
  ' "${MANIFEST}" | sort -t',' -k1,1n -k2,2
)

if [[ "${#SHARDS[@]}" -eq 0 ]]; then
  echo "No missing shards in manifest."
  exit 0
fi

submitted=0
for row in "${SHARDS[@]}"; do
  pr="${row%%,*}"
  key="${row#*,}"
  task="${key%%:*}"
  abl="${key#*:}"
  tshort="$(map_task "${task}")"
  ashort="$(map_abl "${abl}")"
  job="p7v72c_${tshort}_${ashort}"
  script="${SLURM_DIR}/${job}.sh"

  if squeue -u "${USER}" -h -o '%j' | grep -Fxq "${job}"; then
    echo "[SKIP] ${job} already queued/running"
    continue
  fi

  cat > "${script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=batch
#SBATCH --qos=iris-batch-long
#SBATCH --time=2-00:00:00
#SBATCH --mem=112G
#SBATCH --cpus-per-task=28
#SBATCH --output=${LOG_DIR}/${job}_%j.out
#SBATCH --error=${LOG_DIR}/${job}_%j.err
#SBATCH --export=ALL

set -euo pipefail
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "\$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
cd ${REPO}

python3 scripts/run_block3_benchmark_shard.py \\
  --task ${task} \\
  --category autofit \\
  --ablation ${abl} \\
  --models AutoFitV72 \\
  --preset full \\
  --output-dir ${OUTPUT_BASE}/${task}/autofit/${abl} \\
  --seed 42 \\
  --no-verify-first
EOF
  chmod +x "${script}"

  if [[ "${MODE}" == "dry-run" ]]; then
    echo "[DRY] priority=${pr} ${job} => ${task}/${abl}"
  else
    jid="$(sbatch "${script}" | awk '{print $4}')"
    echo "[SUB] priority=${pr} ${job} => ${task}/${abl} (JobID ${jid})"
  fi

  submitted=$((submitted + 1))
  if [[ "${LIMIT}" -gt 0 && "${submitted}" -ge "${LIMIT}" ]]; then
    echo "Reached --limit=${LIMIT}"
    break
  fi
done

echo "Mode: ${MODE}"
echo "Manifest: ${MANIFEST}"
echo "Run tag: ${RUN_TAG}"
echo "Submitted shard count: ${submitted}"
