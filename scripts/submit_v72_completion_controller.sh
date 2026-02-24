#!/usr/bin/env bash
# ============================================================================
# Submit AutoFitV72 completion jobs at strict-key granularity.
#
# Key design:
#   - Input missing keys from missing_key_manifest.csv
#   - Build per-key job manifest (task, ablation, target, horizon)
#   - Apply memory-aware resource class (L vs XL) before submit
#   - Keep canonical output root stable for maximal resume reuse
#
# Usage:
#   bash scripts/submit_v72_completion_controller.sh --dry-run
#   bash scripts/submit_v72_completion_controller.sh --submit
# ============================================================================

set -euo pipefail

MODE="dry-run"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LIMIT=0
STAMP="20260203_225620"
REPO="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACCOUNT="${SLURM_ACCOUNT:-yves.letraon}"
SEED="${SEED:-42}"
PRESET="full"
MANIFEST_PATH="docs/benchmarks/block3_truth_pack/missing_key_manifest.csv"
KEY_JOB_MANIFEST="docs/benchmarks/block3_truth_pack/v72_key_job_manifest.csv"
MEM_PLAN_JSON="docs/benchmarks/block3_truth_pack/v72_memory_plan.json"
MEM_PLAN_CSV="docs/benchmarks/block3_truth_pack/v72_memory_plan.csv"
OUTPUT_BASE="${V72_COMPLETION_OUTPUT_BASE:-runs/benchmarks/block3_${STAMP}_phase7_v72_completion}"
SLURM_DIR="${REPO}/.slurm_scripts/v72_completion_${RUN_TAG}"
LOG_DIR="/work/projects/eint/logs/v72_completion_${RUN_TAG}"

for arg in "$@"; do
  case "$arg" in
    --dry-run) MODE="dry-run" ;;
    --submit) MODE="submit" ;;
    --run-tag=*) RUN_TAG="${arg#*=}" ;;
    --limit=*) LIMIT="${arg#*=}" ;;
    --manifest=*) MANIFEST_PATH="${arg#*=}" ;;
    --output-base=*) OUTPUT_BASE="${arg#*=}" ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

activate_insider_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "insider" ]]; then
        return 0
    fi
    if command -v micromamba >/dev/null 2>&1; then
        local roots=()
        if [[ -n "${MAMBA_ROOT_PREFIX:-}" ]]; then
            roots+=("${MAMBA_ROOT_PREFIX}")
        fi
        roots+=(
            "/mnt/aiongpfs/projects/eint/envs/.micromamba"
            "${HOME}/.local/share/micromamba"
            "${HOME}/micromamba"
        )
        local r
        for r in "${roots[@]}"; do
            [[ -d "${r}" ]] || continue
            export MAMBA_ROOT_PREFIX="${r}"
            eval "$(micromamba shell hook -s bash)"
            if micromamba activate insider; then
                return 0
            fi
        done
    fi
    if command -v conda >/dev/null 2>&1; then
        local conda_base
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1090
            source "${conda_base}/etc/profile.d/conda.sh"
            if conda activate insider; then
                return 0
            fi
        fi
    fi
    echo "FATAL: failed to activate insider environment."
    return 1
}

activate_insider_env
PY_BIN="$(command -v python3 || true)"
if [[ -z "${PY_BIN}" || "${PY_BIN}" != *"insider"* ]]; then
    echo "FATAL: python3 is not from insider env: ${PY_BIN:-<missing>}"
    exit 2
fi
python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(
        f"FATAL: insider python must be >=3.11, got {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
PY
python3 scripts/assert_block3_execution_contract.py \
  --entrypoint "scripts/submit_v72_completion_controller.sh"
bash scripts/install_block3_deps_in_insider.sh

cd "${REPO}"
mkdir -p "${SLURM_DIR}" "${LOG_DIR}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Missing ${MANIFEST_PATH}; generating from v72_coverage_guard..."
  python3 scripts/v72_coverage_guard.py
fi
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "FATAL: missing manifest not found: ${MANIFEST_PATH}"
  exit 2
fi

echo "[1/3] Estimating memory plan..."
python3 scripts/estimate_block3_memory_requirements.py \
  --missing-manifest "${MANIFEST_PATH}" \
  --output-json "${MEM_PLAN_JSON}" \
  --output-csv "${MEM_PLAN_CSV}"

echo "[2/3] Building key-level V7.2 job manifest..."
python3 scripts/build_v72_missing_key_jobs.py \
  --missing-manifest "${MANIFEST_PATH}" \
  --memory-plan-json "${MEM_PLAN_JSON}" \
  --output "${KEY_JOB_MANIFEST}" \
  --output-root "${OUTPUT_BASE}" \
  --seed "${SEED}"

if [[ ! -f "${KEY_JOB_MANIFEST}" ]]; then
  echo "FATAL: key job manifest not generated: ${KEY_JOB_MANIFEST}"
  exit 2
fi

echo "[3/3] Submitting key-level jobs (${MODE})..."
submitted=0
while IFS=$'\t' read -r job_name task ablation target horizon priority_rank priority_group resource_class partition qos mem cpus time_limit seed output_dir reason; do
  [[ -n "${job_name}" ]] || continue

  if squeue -u "${USER}" -h -o '%j' | grep -Fxq "${job_name}"; then
    echo "[SKIP] ${job_name} already queued/running"
    continue
  fi

  script="${SLURM_DIR}/${job_name}.sh"
  cat > "${script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${partition}
#SBATCH --qos=${qos}
#SBATCH --time=${time_limit}
#SBATCH --mem=${mem}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -euo pipefail
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "\$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
cd ${REPO}
INSIDER_PY="\${CONDA_PREFIX}/bin/python3"
if [[ ! -x "\${INSIDER_PY}" ]]; then
  echo "FATAL: insider python missing or non-executable: \${INSIDER_PY}"
  exit 2
fi
echo "============================================================"
echo "Job \${SLURM_JOB_ID} on \$(hostname) — \$(date -Iseconds)"
echo "Task=${task} Ablation=${ablation} Target=${target} Horizon=${horizon}"
echo "ResourceClass=${resource_class} Partition=${partition} QOS=${qos} Mem=${mem} CPUs=${cpus}"
echo "Python: \$(which python3)"
python3 -V
python3 - <<'PY'
import sys
print("sys.executable:", sys.executable)
if "insider" not in sys.executable:
    raise SystemExit("FATAL: runtime python is not insider")
if sys.version_info < (3, 11):
    raise SystemExit(
        f"FATAL: insider python must be >=3.11, got {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
PY
"\${INSIDER_PY}" scripts/assert_block3_execution_contract.py --entrypoint "slurm:\${SLURM_JOB_NAME}"
export B3_MEMORY_CLASS="${resource_class}"
export B3_RESOURCE_PROFILE_ID="${resource_class}:${partition}:${qos}:${mem}:${cpus}"

"\${INSIDER_PY}" scripts/run_block3_benchmark_shard.py \\
  --task ${task} \\
  --category autofit \\
  --ablation ${ablation} \\
  --models AutoFitV72 \\
  --target-filter ${target} \\
  --horizons-filter ${horizon} \\
  --preset ${PRESET} \\
  --output-dir ${output_dir} \\
  --seed ${seed} \\
  --no-verify-first \\
  --enable-global-dedup \\
  --global-dedup-bench-glob block3_20260203_225620*
EOF
  chmod +x "${script}"

  if [[ "${MODE}" == "dry-run" ]]; then
    echo "[DRY] pr=${priority_rank} ${job_name} => ${task}/${ablation}/${target}/h${horizon} (${resource_class}, ${partition}/${qos}, ${mem}, ${cpus}c)"
  else
    jid="$(sbatch "${script}" | awk '{print $4}')"
    echo "[SUB] pr=${priority_rank} ${job_name} => JobID ${jid} (${task}/${ablation}/${target}/h${horizon})"
  fi

  submitted=$((submitted + 1))
  if [[ "${LIMIT}" -gt 0 && "${submitted}" -ge "${LIMIT}" ]]; then
    echo "Reached --limit=${LIMIT}"
    break
  fi
done < <(
  python3 - "${KEY_JOB_MANIFEST}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
rows.sort(
    key=lambda r: (
        int(float(r.get("priority_rank", 99))),
        r.get("task", ""),
        r.get("ablation", ""),
        r.get("target", ""),
        int(float(r.get("horizon", 0))),
    )
)
for r in rows:
    vals = [
        r.get("job_name", ""),
        r.get("task", ""),
        r.get("ablation", ""),
        r.get("target", ""),
        str(int(float(r.get("horizon", 0)))),
        str(int(float(r.get("priority_rank", 99)))),
        r.get("priority_group", ""),
        r.get("resource_class", ""),
        r.get("partition", ""),
        r.get("qos", ""),
        r.get("mem", ""),
        str(int(float(r.get("cpus", 1)))),
        r.get("time_limit", ""),
        str(int(float(r.get("seed", 42)))),
        r.get("output_dir", ""),
        r.get("reason", ""),
    ]
    print("\t".join(vals))
PY
)

echo "Mode: ${MODE}"
echo "Run tag: ${RUN_TAG}"
echo "Output base: ${OUTPUT_BASE}"
echo "Missing manifest: ${MANIFEST_PATH}"
echo "Key job manifest: ${KEY_JOB_MANIFEST}"
echo "Memory plan: ${MEM_PLAN_JSON}"
echo "Submitted key count: ${submitted}"
