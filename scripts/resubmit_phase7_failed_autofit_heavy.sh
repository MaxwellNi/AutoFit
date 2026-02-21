#!/usr/bin/env bash
# ============================================================================
# Resubmit failed canonical Phase-7 AutoFit FULL shards with schedulable
# resources and extended walltime.
# ============================================================================
#
# Scope (fixed):
#   - p7_af1_t1_fu  (AutoFitV1,V2,V2E,V3,V3E)
#   - p7_af1_t2_fu  (AutoFitV1,V2,V2E,V3,V3E)
#   - p7_af1_t3_fu  (AutoFitV1,V2,V2E,V3,V3E)
#   - p7_af2_t1_fu  (AutoFitV3Max,V4,V5,V6,V7)
#
# Usage:
#   bash scripts/resubmit_phase7_failed_autofit_heavy.sh --dry-run
#   bash scripts/resubmit_phase7_failed_autofit_heavy.sh --submit
#   bash scripts/resubmit_phase7_failed_autofit_heavy.sh --submit --run-tag=mytag
#   bash scripts/resubmit_phase7_failed_autofit_heavy.sh --submit --scope=ce
#   bash scripts/resubmit_phase7_failed_autofit_heavy.sh --submit --scope=fu
# ============================================================================

set -euo pipefail

MODE="dry-run"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SCOPE="all" # all | ce | fu
for arg in "$@"; do
    case "$arg" in
        --dry-run) MODE="dry-run" ;;
        --submit) MODE="submit" ;;
        --run-tag=*) RUN_TAG="${arg#*=}" ;;
        --scope=*) SCOPE="${arg#*=}" ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

if [[ "${SCOPE}" != "all" && "${SCOPE}" != "ce" && "${SCOPE}" != "fu" ]]; then
    echo "Invalid --scope=${SCOPE} (expected all|ce|fu)"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="/work/projects/eint/logs/phase7_autofit_resubmit_${RUN_TAG}"
SLURM_DIR="${REPO_ROOT}/.slurm_scripts/phase7_resubmit_autofit_${RUN_TAG}"
mkdir -p "${SLURM_DIR}" "${LOG_DIR}"

ACCOUNT="${SLURM_ACCOUNT:-yves.letraon}"
PARTITION="batch"
QOS="iris-batch-long"
TIME_LIMIT="${P7_AF_RESUBMIT_TIME:-4-00:00:00}"
MEMORY="${P7_AF_RESUBMIT_MEM:-112G}"
CPUS="${P7_AF_RESUBMIT_CPUS:-28}"
STAMP="20260203_225620"
OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7"
SEED=42

parse_mem_to_mb() {
    local mem="$1"
    if [[ "${mem}" =~ ^([0-9]+)[Gg]$ ]]; then
        echo "$(( ${BASH_REMATCH[1]} * 1024 ))"
        return 0
    fi
    if [[ "${mem}" =~ ^([0-9]+)[Mm]$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    echo "Unsupported memory format: ${mem}" >&2
    return 1
}

batch_mem_guard() {
    if ! command -v sinfo >/dev/null 2>&1; then
        return 0
    fi
    local min_mem_mb req_mem_mb
    min_mem_mb="$(sinfo -p "${PARTITION}" -h -o "%m" | awk 'NR==1{m=$1} $1<m{m=$1} END{print m+0}')"
    req_mem_mb="$(parse_mem_to_mb "${MEMORY}")"
    if [[ -n "${min_mem_mb}" ]] && (( req_mem_mb > min_mem_mb )); then
        echo "FATAL: Requested mem=${MEMORY} exceeds ${PARTITION} node memory cap (${min_mem_mb}MB)." >&2
        echo "Hint: export P7_AF_RESUBMIT_MEM=112G (or <= node cap)." >&2
        exit 2
    fi
}

batch_mem_guard

read -r -d '' ENV_BLOCK <<'ENVBLOCK' || true
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root
ENVBLOCK

submit_job() {
    local job_name="$1"
    local task="$2"
    local ablation="$3"
    local models="$4"
    local out_dir="${OUTPUT_BASE}/${task}/autofit/${ablation}"
    local script_path="${SLURM_DIR}/${job_name}.sh"

    cat > "${script_path}" <<SLURM
#!/usr/bin/env bash
#SBATCH --job-name=${job_name}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -euo pipefail
${ENV_BLOCK}

echo "============================================================"
echo "Job \${SLURM_JOB_ID} on \$(hostname) — \$(date -Iseconds)"
echo "Task: ${task} | Category: autofit | Ablation: ${ablation}"
echo "Models: ${models}"
echo "Output: ${out_dir}"
echo "============================================================"

python3 scripts/run_block3_benchmark_shard.py \\
    --task ${task} \\
    --category autofit \\
    --ablation ${ablation} \\
    --preset full \\
    --output-dir ${out_dir} \\
    --seed ${SEED} \\
    --no-verify-first \\
    --models ${models}

echo "Done: \$(date -Iseconds)"
SLURM

    chmod +x "${script_path}"

    # Avoid duplicate in-flight jobs with the same name.
    if squeue -u "${USER:-$(id -un)}" -h -o '%j' | grep -Fxq "${job_name}"; then
        echo "[SKIP] ${job_name} already active in queue"
        return 0
    fi

    if [[ "${MODE}" == "dry-run" ]]; then
        echo "[DRY] ${job_name} => ${PARTITION}/${QOS} ${TIME_LIMIT} ${MEMORY} ${CPUS}c"
        echo "      task=${task} ablation=${ablation}"
        echo "      script: ${script_path}"
    else
        local jid
        jid="$(sbatch "${script_path}" | awk '{print $4}')"
        echo "[SUB] ${job_name} => JobID ${jid}"
    fi
}

AF1_MODELS="AutoFitV1,AutoFitV2,AutoFitV2E,AutoFitV3,AutoFitV3E"
AF2_MODELS="AutoFitV3Max,AutoFitV4,AutoFitV5,AutoFitV6,AutoFitV7"

if [[ "${SCOPE}" == "all" || "${SCOPE}" == "ce" ]]; then
    submit_job "p7r_af1_t1_ce" "task1_outcome" "core_edgar" "${AF1_MODELS}"
    submit_job "p7r_af2_t1_ce" "task1_outcome" "core_edgar" "${AF2_MODELS}"
fi

if [[ "${SCOPE}" == "all" || "${SCOPE}" == "fu" ]]; then
    submit_job "p7r_af1_t1_fu" "task1_outcome" "full" "${AF1_MODELS}"
    submit_job "p7r_af1_t2_fu" "task2_forecast" "full" "${AF1_MODELS}"
    submit_job "p7r_af1_t3_fu" "task3_risk_adjust" "full" "${AF1_MODELS}"
    submit_job "p7r_af2_t1_fu" "task1_outcome" "full" "${AF2_MODELS}"
fi

echo "Mode: ${MODE}"
echo "Run tag: ${RUN_TAG}"
echo "Scope: ${SCOPE}"
echo "Slurm scripts: ${SLURM_DIR}"
