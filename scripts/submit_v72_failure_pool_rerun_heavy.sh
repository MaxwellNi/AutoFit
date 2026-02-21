#!/usr/bin/env bash
# ============================================================================
# Submit V7.2 failure-pool heavy reruns (4 fixed spike keys).
# ============================================================================
#
# Fixed scope:
#   task1_outcome / core_edgar / investors_count / horizon in {1,7,14,30}
#
# Resource profile (Heavy+schedulable on Iris batch):
#   partition=batch, qos=iris-batch-long, mem=112G, cpus=28
#
# Usage:
#   bash scripts/submit_v72_failure_pool_rerun_heavy.sh --dry-run
#   bash scripts/submit_v72_failure_pool_rerun_heavy.sh --submit
#
# Notes:
#   - Does not cancel existing queue.
#   - Writes to independent rerun directory with *_rerun_heavy tag.
# ============================================================================

set -euo pipefail

MODE="dry-run"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
STAMP="20260203_225620"
REPO="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

for arg in "$@"; do
    case "$arg" in
        --dry-run) MODE="dry-run" ;;
        --submit) MODE="submit" ;;
        --run-tag=*) RUN_TAG="${arg#*=}" ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

ACCOUNT="${SLURM_ACCOUNT:-yves.letraon}"
PARTITION="batch"
QOS="iris-batch-long"
TIME_LIMIT="4-00:00:00"
MEMORY="${V72_FAILURE_POOL_MEM:-112G}"
CPUS="${V72_FAILURE_POOL_CPUS:-28}"
SEED="42"
PRESET="full"
MODELS="AutoFitV71,AutoFitV72"

OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_phase7_v72_failure_pool_rerun_heavy_${RUN_TAG}"
SLURM_DIR="${REPO}/.slurm_scripts/v72_failure_pool_rerun_heavy_${RUN_TAG}"
LOG_DIR="/work/projects/eint/logs/v72_failure_pool_rerun_heavy_${RUN_TAG}"
mkdir -p "${SLURM_DIR}" "${LOG_DIR}"

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
        echo "Hint: export V72_FAILURE_POOL_MEM=112G (or <= node cap)." >&2
        exit 2
    fi
}

batch_mem_guard

ENV_BLOCK='
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root
'

submit_one() {
    local h="$1"
    local job="p7r_v72_ic_ce_h${h}_heavy"
    local outdir="${OUTPUT_BASE}/task1_outcome/autofit/core_edgar/h${h}_rerun_heavy"
    local script="${SLURM_DIR}/${job}.sh"

    cat > "${script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${LOG_DIR}/${job}_%j.out
#SBATCH --error=${LOG_DIR}/${job}_%j.err
#SBATCH --export=ALL
#SBATCH --signal=USR1@120

set -euo pipefail
${ENV_BLOCK}

echo "============================================================"
echo "Job \${SLURM_JOB_ID} on \$(hostname) — \$(date -Iseconds)"
echo "Failure pool rerun heavy | h=${h}"
echo "Output: ${outdir}"
echo "============================================================"

python3 scripts/run_block3_benchmark_shard.py \\
    --task task1_outcome \\
    --category autofit \\
    --ablation core_edgar \\
    --preset ${PRESET} \\
    --output-dir ${outdir} \\
    --seed ${SEED} \\
    --no-verify-first \\
    --models ${MODELS} \\
    --target-filter investors_count \\
    --horizons-filter ${h}
EOF

    chmod +x "${script}"
    if [[ "${MODE}" == "dry-run" ]]; then
        echo "[DRY] ${job} -> ${PARTITION}/${QOS} ${TIME_LIMIT} ${MEMORY} ${CPUS}c"
        echo "      script: ${script}"
    else
        local jid
        jid="$(sbatch "${script}" | awk '{print $4}')"
        echo "[SUB] ${job} -> JobID ${jid}"
    fi
}

echo "Mode: ${MODE}"
echo "Run tag: ${RUN_TAG}"
echo "Output base: ${OUTPUT_BASE}"

for h in 1 7 14 30; do
    submit_one "${h}"
done

echo "Done."
