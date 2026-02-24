#!/usr/bin/env bash
# =============================================================================
# Block 3 Phase 7 Local Runner on Dual 3090 (Safety-First Saturation)
#
# Goals:
#   1) Enforce audit/smoke preflight before large local runs
#   2) Saturate dual GPUs and CPU safely with memory headroom
#   3) Avoid OOM/restart events via host/GPU watermark guards
#
# Environment policy:
#   - Must run in existing insider env (conda/micromamba)
#   - Do NOT create a new env and do NOT use base
#
# Usage:
#   conda activate insider
#   bash scripts/run_phase7_dual3090_safe.sh --full --v71-variant=g02
#   bash scripts/run_phase7_dual3090_safe.sh --pilot
#   ALLOW_UNSAFE_SKIP_PREFLIGHT=1 bash scripts/run_phase7_dual3090_safe.sh --full --skip-preflight
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO="${REPO_ROOT:-${DEFAULT_REPO}}"

# Known host paths:
#   3090: /home/pni/project/repo_root
#   4090: /home/pni/projects/repo_root
if [[ ! -f "${REPO}/scripts/run_block3_benchmark_shard.py" ]]; then
    for cand in \
        "${DEFAULT_REPO}" \
        "/home/pni/project/repo_root" \
        "/home/pni/projects/repo_root" \
        "/home/users/npin/repo_root"
    do
        if [[ -f "${cand}/scripts/run_block3_benchmark_shard.py" ]]; then
            REPO="${cand}"
            break
        fi
    done
fi

if [[ ! -f "${REPO}/scripts/run_block3_benchmark_shard.py" ]]; then
    echo "FATAL: cannot locate repo root. Set REPO_ROOT and retry."
    exit 2
fi
STAMP="20260203_225620"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
MODE="full"
V71_VARIANT="g02"
SKIP_PREFLIGHT=false
DRY_RUN=false

# Safety watermarks (tunable)
HOST_MEM_FLOOR_GB=36
GPU_MIN_FREE_MB=18000
GPU_THREADS=8
CPU_THREADS=24

for arg in "$@"; do
    case "$arg" in
        --pilot)
            MODE="pilot"
            ;;
        --full)
            MODE="full"
            ;;
        --run-tag=*)
            RUN_TAG="${arg#*=}"
            ;;
        --v71-variant=*)
            V71_VARIANT="${arg#*=}"
            ;;
        --host-mem-floor-gb=*)
            HOST_MEM_FLOOR_GB="${arg#*=}"
            ;;
        --gpu-min-free-mb=*)
            GPU_MIN_FREE_MB="${arg#*=}"
            ;;
        --skip-preflight)
            SKIP_PREFLIGHT=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

if $SKIP_PREFLIGHT && [[ "${ALLOW_UNSAFE_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    echo "Refusing --skip-preflight without explicit override."
    echo "Use only for emergency reruns:"
    echo "  ALLOW_UNSAFE_SKIP_PREFLIGHT=1 bash scripts/run_phase7_dual3090_safe.sh --${MODE} --skip-preflight"
    exit 2
fi

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
if [[ -z "$PY_BIN" || "$PY_BIN" != *"insider"* ]]; then
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
python3 "${REPO}/scripts/assert_block3_execution_contract.py" \
  --entrypoint "scripts/run_phase7_dual3090_safe.sh"
INSIDER_PY="${CONDA_PREFIX}/bin/python3"
if [[ ! -x "${INSIDER_PY}" ]]; then
    echo "FATAL: insider python missing or non-executable: ${INSIDER_PY}"
    exit 2
fi

GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
if [[ -z "$GPU_COUNT" || "$GPU_COUNT" -lt 2 ]]; then
    echo "FATAL: dual GPU host expected, found GPU count=${GPU_COUNT:-0}."
    exit 2
fi

cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

if ! $DRY_RUN; then
    echo "=== Auto-fix insider dependencies ==="
    bash "${REPO}/scripts/install_block3_deps_in_insider.sh"
fi

OUTPUT_BASE="runs/benchmarks/block3_${STAMP}_dual3090_phase7_${RUN_TAG}"
LOG_BASE="${OUTPUT_BASE}/logs"
mkdir -p "$LOG_BASE"

declare -A V71_VARIANTS
V71_VARIANTS["g01"]='{"AutoFitV71":{"top_k":8,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":true,"enable_regime_retrieval":true}}'
V71_VARIANTS["g02"]='{"AutoFitV71":{"top_k":10,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":true,"enable_regime_retrieval":true}}'
V71_VARIANTS["g03"]='{"AutoFitV71":{"top_k":6,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":true,"enable_regime_retrieval":true}}'
V71_VARIANTS["g04"]='{"AutoFitV71":{"top_k":8,"min_ensemble_size_heavy_tail":3,"dynamic_weighting":true,"enable_regime_retrieval":false}}'
V71_VARIANTS["g05"]='{"AutoFitV71":{"top_k":8,"min_ensemble_size_heavy_tail":2,"dynamic_weighting":false,"enable_regime_retrieval":true}}'

if [[ -z "${V71_VARIANTS[$V71_VARIANT]:-}" ]]; then
    echo "Unknown --v71-variant=${V71_VARIANT}. Allowed: g01,g02,g03,g04,g05"
    exit 1
fi

if ! $DRY_RUN && ! $SKIP_PREFLIGHT; then
    echo "=== Mandatory preflight gate before dual-3090 run ==="
    bash "${REPO}/scripts/preflight_block3_v71_gate.sh" --v71-variant="${V71_VARIANT}"
fi

TS_GPU0="PatchTST,iTransformer,TimesNet,TSMixer,Informer,Autoformer,FEDformer,VanillaTransformer,TiDE,NBEATSx"
TS_GPU1="BiTCN,KAN,RMoK,SOFTS,StemGNN"
FD_GPU0="Chronos,ChronosBolt,Chronos2,Moirai,MoiraiLarge"
FD_GPU1="Moirai2,Timer,TimeMoE,MOMENT,LagLlama,TimesFM"
AF_CPU1="AutoFitV1,AutoFitV2,AutoFitV2E,AutoFitV3,AutoFitV3E,AutoFitV3Max"
AF_CPU2="AutoFitV4,AutoFitV5,AutoFitV6,AutoFitV7,AutoFitV71"
AF_CPU2_KW="${V71_VARIANTS[$V71_VARIANT]}"

TASKS=("task1_outcome" "task2_forecast" "task3_risk_adjust")

task_ablations() {
    local task="$1"
    if [[ "$MODE" == "pilot" ]]; then
        echo "core_only core_edgar"
        return
    fi
    case "$task" in
        task1_outcome|task2_forecast)
            echo "core_only core_text core_edgar full"
            ;;
        task3_risk_adjust)
            echo "core_only core_edgar full"
            ;;
        *)
            echo ""
            ;;
    esac
}

# job format:
# task|category|ablation|models|out_subdir|gpu_id|threads|min_mem_gb|model_kwargs_json
GPU0_QUEUE=()
GPU1_QUEUE=()
CPUA_QUEUE=()
CPUB_QUEUE=()

for task in "${TASKS[@]}"; do
    read -r -a abls <<< "$(task_ablations "$task")"
    for abl in "${abls[@]}"; do
        GPU0_QUEUE+=("${task}|deep_classical|${abl}||deep_classical|0|${GPU_THREADS}|56|")
        GPU0_QUEUE+=("${task}|transformer_sota|${abl}|${TS_GPU0}|transformer_sota_A|0|${GPU_THREADS}|64|")
        GPU0_QUEUE+=("${task}|foundation|${abl}|${FD_GPU0}|foundation_A|0|${GPU_THREADS}|64|")

        GPU1_QUEUE+=("${task}|transformer_sota|${abl}|${TS_GPU1}|transformer_sota_B|1|${GPU_THREADS}|64|")
        GPU1_QUEUE+=("${task}|foundation|${abl}|${FD_GPU1}|foundation_B|1|${GPU_THREADS}|64|")
        GPU1_QUEUE+=("${task}|irregular|${abl}||irregular|1|${GPU_THREADS}|56|")

        CPUA_QUEUE+=("${task}|ml_tabular|${abl}||ml_tabular|-1|${CPU_THREADS}|120|")
        CPUA_QUEUE+=("${task}|autofit|${abl}|${AF_CPU1}|autofit_af1|-1|${CPU_THREADS}|120|")

        CPUB_QUEUE+=("${task}|statistical|${abl}||statistical|-1|16|130|")
        CPUB_QUEUE+=("${task}|autofit|${abl}|${AF_CPU2}|autofit_af2|-1|16|130|${AF_CPU2_KW}")
    done
done

wait_for_host_memory() {
    local min_gb="$1"
    local min_kb=$(( min_gb * 1024 * 1024 ))
    while true; do
        local avail_kb
        avail_kb="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
        if [[ -n "$avail_kb" && "$avail_kb" -ge "$min_kb" ]]; then
            break
        fi
        echo "[$(date -Iseconds)] waiting host memory: available_kb=${avail_kb:-0}, need_kb=${min_kb}" >&2
        sleep 20
    done
}

wait_for_gpu_ready() {
    local gpu_id="$1"
    local min_free_mb="$2"
    while true; do
        local line
        line="$(nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" | head -n1)"
        if [[ -z "$line" ]]; then
            echo "[$(date -Iseconds)] waiting GPU${gpu_id} stats..." >&2
            sleep 20
            continue
        fi
        local total used util
        IFS=',' read -r total used util <<< "$line"
        total="${total// /}"
        used="${used// /}"
        util="${util// /}"
        local free=$(( total - used ))
        if [[ "$free" -ge "$min_free_mb" ]]; then
            break
        fi
        echo "[$(date -Iseconds)] waiting GPU${gpu_id}: free=${free}MB need=${min_free_mb}MB util=${util}%" >&2
        sleep 20
    done
}

run_job() {
    local worker="$1"
    local spec="$2"

    IFS='|' read -r task category ablation models out_subdir gpu_id threads min_mem_gb kwargs_json <<< "$spec"

    local outdir="${OUTPUT_BASE}/${task}/${out_subdir}/${ablation}"
    local logdir="${LOG_BASE}/${worker}"
    local logfile="${logdir}/${task}_${out_subdir}_${ablation}.log"
    mkdir -p "$outdir" "$logdir"

    if $DRY_RUN; then
        echo "[DRY] ${worker}: task=${task} category=${category} ablation=${ablation} models=${models:-ALL} gpu=${gpu_id} mem>=${min_mem_gb}G"
        return 0
    fi

    local effective_min_mem_gb="$min_mem_gb"
    if [[ "$HOST_MEM_FLOOR_GB" -gt "$effective_min_mem_gb" ]]; then
        effective_min_mem_gb="$HOST_MEM_FLOOR_GB"
    fi
    wait_for_host_memory "$effective_min_mem_gb"

    if [[ "$gpu_id" -ge 0 ]]; then
        wait_for_gpu_ready "$gpu_id" "$GPU_MIN_FREE_MB"
        export CUDA_VISIBLE_DEVICES="$gpu_id"
    else
        unset CUDA_VISIBLE_DEVICES || true
    fi

    echo "[$(date -Iseconds)] START ${worker}: ${task}/${category}/${ablation} (models=${models:-ALL}, gpu=${gpu_id})"

    cmd=(
        "${INSIDER_PY}" scripts/run_block3_benchmark_shard.py
        --task "$task"
        --category "$category"
        --ablation "$ablation"
        --preset full
        --output-dir "$outdir"
        --seed 42
        --no-verify-first
    )
    if [[ -n "$models" ]]; then
        cmd+=(--models "$models")
    fi
    if [[ -n "$kwargs_json" ]]; then
        cmd+=(--model-kwargs-json "$kwargs_json")
    fi

    set +e
    OMP_NUM_THREADS="$threads" \
    MKL_NUM_THREADS="$threads" \
    OPENBLAS_NUM_THREADS="$threads" \
    PYTHONUNBUFFERED=1 \
    "${cmd[@]}" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ "$rc" -eq 0 ]]; then
        echo "[$(date -Iseconds)] DONE  ${worker}: ${task}/${category}/${ablation}"
    else
        echo "[$(date -Iseconds)] FAIL  ${worker}: ${task}/${category}/${ablation} rc=${rc}" >&2
    fi
    return "$rc"
}

run_worker() {
    local worker="$1"
    local queue_name="$2"
    local -n queue_ref="$queue_name"

    local failed=0
    for spec in "${queue_ref[@]}"; do
        if ! run_job "$worker" "$spec"; then
            failed=$((failed + 1))
        fi
    done
    echo "[$(date -Iseconds)] ${worker} finished: total=${#queue_ref[@]}, failed=${failed}"
    return 0
}

echo "================================================================"
echo "Dual-3090 Phase7 runner"
echo "  mode=${MODE} run_tag=${RUN_TAG} v71_variant=${V71_VARIANT}"
echo "  repo=${REPO}"
echo "  output=${OUTPUT_BASE}"
echo "  host_mem_floor=${HOST_MEM_FLOOR_GB}GB gpu_min_free=${GPU_MIN_FREE_MB}MB"
echo "  queues: gpu0=${#GPU0_QUEUE[@]} gpu1=${#GPU1_QUEUE[@]} cpua=${#CPUA_QUEUE[@]} cpub=${#CPUB_QUEUE[@]}"
echo "================================================================"

action_prefix=""
if $DRY_RUN; then
    action_prefix="[DRY] "
fi

echo "${action_prefix}Launching workers..."
run_worker "gpu0" GPU0_QUEUE > "${LOG_BASE}/gpu0_master.log" 2>&1 &
PID_GPU0=$!
run_worker "gpu1" GPU1_QUEUE > "${LOG_BASE}/gpu1_master.log" 2>&1 &
PID_GPU1=$!
run_worker "cpua" CPUA_QUEUE > "${LOG_BASE}/cpua_master.log" 2>&1 &
PID_CPUA=$!
run_worker "cpub" CPUB_QUEUE > "${LOG_BASE}/cpub_master.log" 2>&1 &
PID_CPUB=$!

echo "PIDs: gpu0=${PID_GPU0} gpu1=${PID_GPU1} cpua=${PID_CPUA} cpub=${PID_CPUB}"
echo "Monitor: tail -f ${LOG_BASE}/*_master.log"

wait "$PID_GPU0"; RC_GPU0=$?
wait "$PID_GPU1"; RC_GPU1=$?
wait "$PID_CPUA"; RC_CPUA=$?
wait "$PID_CPUB"; RC_CPUB=$?

echo "================================================================"
echo "Dual-3090 run complete"
echo "  rc_gpu0=${RC_GPU0} rc_gpu1=${RC_GPU1} rc_cpua=${RC_CPUA} rc_cpub=${RC_CPUB}"
echo "  output=${OUTPUT_BASE}"
echo "================================================================"

exit 0
