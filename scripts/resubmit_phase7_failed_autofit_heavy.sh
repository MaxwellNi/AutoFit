#!/usr/bin/env bash
# Resubmit failed canonical Phase-7 AutoFit core_edgar jobs with larger resources.
# Keeps the exact same data/task/ablation/model roster for fair comparison.
#
# Usage:
#   bash scripts/resubmit_phase7_failed_autofit_heavy.sh --dry-run
#   bash scripts/resubmit_phase7_failed_autofit_heavy.sh --submit

set -euo pipefail

MODE="dry-run"
for arg in "$@"; do
    case "$arg" in
        --dry-run) MODE="dry-run" ;;
        --submit) MODE="submit" ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="/work/projects/eint/logs/phase7"
SLURM_DIR="${REPO_ROOT}/.slurm_scripts/phase7_resubmit_heavy_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SLURM_DIR" "$LOG_DIR"

ACCOUNT="yves.letraon"
PARTITION="batch"
QOS="iris-batch-long"
TIME_LIMIT="4-00:00:00"
MEMORY="160G"
CPUS="28"
STAMP="20260203_225620"
OUT_BASE="runs/benchmarks/block3_${STAMP}_phase7/task1_outcome/autofit/core_edgar"

read -r -d '' ENV_BLOCK <<'ENVBLOCK' || true
export MAMBA_ROOT_PREFIX=/mnt/aiongpfs/projects/eint/envs/.micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate insider
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /home/users/npin/repo_root
ENVBLOCK

submit_job() {
    local job_name="$1"
    local models="$2"
    local script_path="${SLURM_DIR}/${job_name}.sh"

    cat > "$script_path" <<SLURM
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
#SBATCH --signal=B:USR1@120

set -e
${ENV_BLOCK}

echo "============================================================"
echo "Job \${SLURM_JOB_ID} on \$(hostname) — \$(date -Iseconds)"
echo "Task: task1_outcome | Category: autofit | Ablation: core_edgar"
echo "Models: ${models}"
echo "Output: ${OUT_BASE}"
echo "============================================================"

python3 scripts/run_block3_benchmark_shard.py \\
    --task task1_outcome \\
    --category autofit \\
    --ablation core_edgar \\
    --preset full \\
    --output-dir ${OUT_BASE} \\
    --seed 42 \\
    --no-verify-first \\
    --models ${models}

echo "Done: \$(date -Iseconds)"
SLURM

    chmod +x "$script_path"

    if [[ "$MODE" == "dry-run" ]]; then
        echo "[DRY] ${job_name} => ${PARTITION}/${QOS} ${TIME_LIMIT} ${MEMORY} ${CPUS}c"
        echo "      script: ${script_path}"
    else
        local jid
        jid=$(sbatch "$script_path" | awk '{print $4}')
        echo "[SUB] ${job_name} => JobID ${jid}"
    fi
}

submit_job "p7r_af1_t1_ce" "AutoFitV1,AutoFitV2,AutoFitV2E,AutoFitV3,AutoFitV3E"
submit_job "p7r_af2_t1_ce" "AutoFitV3Max,AutoFitV4,AutoFitV5,AutoFitV6,AutoFitV7,AutoFitV71"

echo "Mode: ${MODE}"
echo "Slurm scripts: ${SLURM_DIR}"
