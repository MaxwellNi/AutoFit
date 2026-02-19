#!/usr/bin/env bash
# ============================================================================
# Block3 AutoFit V7/V71/V72 one-shot runner for 4090 host (tmux friendly)
# Network assumption: this host can SSH to Iris; Iris cannot SSH back.
# ============================================================================
set -euo pipefail

REPO="${REPO_ROOT:-/home/pni/projects/repo_root}"
IRIS_HOST="${IRIS_HOST:-iris}"
IRIS_REPO="${IRIS_REPO:-/home/users/npin/repo_root}"
STAMP="${STAMP:-20260203_225620}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTBASE_REL="runs/benchmarks/block3_${STAMP}_phase7_v72_4090_${RUN_TAG}"
OUTBASE_ABS="${REPO}/${OUTBASE_REL}"
MODELS="${MODELS:-AutoFitV7,AutoFitV71,AutoFitV72}"
V71_VARIANT="${V71_VARIANT:-g02}"
SKIP_SANITY_CHECKS="${SKIP_SANITY_CHECKS:-0}"
SKIP_TABPFN_UNINSTALL="${SKIP_TABPFN_UNINSTALL:-0}"
MEM_GUARD_GB="${MEM_GUARD_GB:-70}"
GPU_GUARD_MB="${GPU_GUARD_MB:-8000}"

cd "${REPO}"

activate_insider() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "insider" ]]; then
        return 0
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "FATAL: conda not found"
        return 2
    fi
    # shellcheck disable=SC1090
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate insider
}

activate_insider

PY_BIN="$(command -v python3 || true)"
if [[ -z "${PY_BIN}" ]]; then
    echo "FATAL: python3 not found"
    exit 2
fi

python3 - <<'PY'
import sys
print("python:", sys.executable)
if "insider" not in sys.executable:
    raise SystemExit("FATAL: not running insider python")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi not found"
    exit 2
fi

GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
if [[ -z "${GPU_COUNT}" || "${GPU_COUNT}" -lt 2 ]]; then
    echo "FATAL: expected at least 2 GPUs, found ${GPU_COUNT:-0}"
    exit 2
fi

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export HF_HUB_DISABLE_TELEMETRY=1

if [[ "${SKIP_TABPFN_UNINSTALL}" != "1" ]]; then
    python3 -m pip uninstall -y tabpfn tabpfn-common-utils >/dev/null 2>&1 || true
fi

mkdir -p "${OUTBASE_ABS}/logs"
{
    echo "RUN_TAG=${RUN_TAG}"
    echo "OUTBASE_REL=${OUTBASE_REL}"
    echo "OUTBASE_ABS=${OUTBASE_ABS}"
    echo "MODELS=${MODELS}"
    date -Iseconds
} | tee "${OUTBASE_ABS}/RUN_INFO.txt"

if [[ "${SKIP_SANITY_CHECKS}" != "1" ]]; then
    python3 scripts/block3_verify_freeze.py >/dev/null
    bash scripts/preflight_block3_v71_gate.sh --v71-variant="${V71_VARIANT}" --skip-smoke >/dev/null
fi

mem_guard_gb() {
    local need_gb="$1"
    while true; do
        local avail_gb
        avail_gb="$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
        if [[ -n "${avail_gb}" && "${avail_gb}" -ge "${need_gb}" ]]; then
            break
        fi
        echo "[$(date -Iseconds)] waiting mem: avail=${avail_gb}GB need=${need_gb}GB"
        sleep 20
    done
}

gpu_guard_mb() {
    local gpu="$1"
    local need_mb="$2"
    while true; do
        local line total used free
        line="$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "${gpu}" | head -n1 || true)"
        if [[ -z "${line}" ]]; then
            echo "[$(date -Iseconds)] waiting nvidia-smi gpu=${gpu}"
            sleep 20
            continue
        fi
        total="$(echo "${line}" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
        used="$(echo "${line}" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
        free=$(( total - used ))
        if [[ "${free}" -ge "${need_mb}" ]]; then
            break
        fi
        echo "[$(date -Iseconds)] waiting gpu${gpu}: free=${free}MB need=${need_mb}MB"
        sleep 20
    done
}

is_done() {
    local outdir="$1"
    [[ -f "${outdir}/metrics.json" && -f "${outdir}/MANIFEST.json" ]] || return 1
    python3 - <<PY >/dev/null 2>&1
import json
import pathlib

m = pathlib.Path("${outdir}/MANIFEST.json")
d = json.loads(m.read_text())
if str(d.get("status", "")).lower() != "completed":
    raise SystemExit(1)
PY
}

run_shard() {
    local worker="$1"
    local gpu="$2"
    local task="$3"
    local ablation="$4"

    local outdir="${OUTBASE_ABS}/${task}/autofit/${ablation}"
    local logf="${OUTBASE_ABS}/logs/${worker}_${task}_${ablation}.log"
    mkdir -p "${outdir}" "$(dirname "${logf}")"

    if is_done "${outdir}"; then
        echo "[$(date -Iseconds)] SKIP completed ${worker} ${task}/${ablation}" | tee -a "${logf}"
        return 0
    fi

    mem_guard_gb "${MEM_GUARD_GB}"
    gpu_guard_mb "${gpu}" "${GPU_GUARD_MB}"

    local attempt=1
    local max_attempt=2
    local threads=20
    while (( attempt <= max_attempt )); do
        echo "[$(date -Iseconds)] START ${worker} ${task}/${ablation} gpu=${gpu} attempt=${attempt}" | tee -a "${logf}"
        if OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}" OPENBLAS_NUM_THREADS="${threads}" \
            CUDA_VISIBLE_DEVICES="${gpu}" \
            python3 scripts/run_block3_benchmark_shard.py \
                --task "${task}" \
                --category autofit \
                --models "${MODELS}" \
                --ablation "${ablation}" \
                --preset full \
                --output-dir "${outdir}" \
                --seed 42 \
                --no-verify-first \
                >> "${logf}" 2>&1; then
            echo "[$(date -Iseconds)] DONE ${worker} ${task}/${ablation}" | tee -a "${logf}"
            return 0
        fi
        echo "[$(date -Iseconds)] FAIL ${worker} ${task}/${ablation} attempt=${attempt}" | tee -a "${logf}"
        attempt=$((attempt + 1))
        threads=12
        sleep 20
    done

    echo "[$(date -Iseconds)] FATAL ${worker} ${task}/${ablation} exhausted retries" | tee -a "${logf}"
    return 1
}

worker_a() {
    run_shard A 0 task1_outcome core_only
    run_shard A 0 task1_outcome core_text
    run_shard A 0 task1_outcome core_edgar
    run_shard A 0 task1_outcome full
    run_shard A 0 task2_forecast core_only
    run_shard A 0 task2_forecast core_text
}

worker_b() {
    run_shard B 1 task2_forecast core_edgar
    run_shard B 1 task2_forecast full
    run_shard B 1 task3_risk_adjust core_only
    run_shard B 1 task3_risk_adjust core_edgar
    run_shard B 1 task3_risk_adjust full
}

(worker_a) > "${OUTBASE_ABS}/logs/worker_a_master.log" 2>&1 & PA=$!
(worker_b) > "${OUTBASE_ABS}/logs/worker_b_master.log" 2>&1 & PB=$!

echo "PIDS: A=${PA} B=${PB}" | tee -a "${OUTBASE_ABS}/RUN_INFO.txt"

RCA=0
RCB=0
if ! wait "${PA}"; then RCA=$?; fi
if ! wait "${PB}"; then RCB=$?; fi

echo "worker_a_rc=${RCA}" | tee -a "${OUTBASE_ABS}/RUN_INFO.txt"
echo "worker_b_rc=${RCB}" | tee -a "${OUTBASE_ABS}/RUN_INFO.txt"

N_METRICS="$(find "${OUTBASE_ABS}" -name metrics.json | wc -l | tr -d ' ')"
echo "metrics_json_count=${N_METRICS}" | tee -a "${OUTBASE_ABS}/RUN_INFO.txt"
if [[ "${N_METRICS}" -lt 11 ]]; then
    echo "WARNING: expected >=11 metrics.json, got ${N_METRICS}" | tee -a "${OUTBASE_ABS}/RUN_INFO.txt"
fi

echo "[$(date -Iseconds)] syncing back to iris..."
ssh "${IRIS_HOST}" "mkdir -p '${IRIS_REPO}/${OUTBASE_REL}'"
rsync -aH --partial --append-verify --inplace --info=progress2,stats2 --human-readable \
    "${OUTBASE_ABS}/" "${IRIS_HOST}:${IRIS_REPO}/${OUTBASE_REL}/"
echo "SYNC_BACK_DONE: ${IRIS_REPO}/${OUTBASE_REL}" | tee -a "${OUTBASE_ABS}/RUN_INFO.txt"

FINAL_RC=0
if [[ "${RCA}" -ne 0 || "${RCB}" -ne 0 ]]; then
    FINAL_RC=1
fi

echo "ALL_DONE final_rc=${FINAL_RC}" | tee -a "${OUTBASE_ABS}/RUN_INFO.txt"
exit "${FINAL_RC}"

