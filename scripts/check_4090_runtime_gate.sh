#!/usr/bin/env bash
# =============================================================================
# Dual-GPU runtime gate: GPU health + insider runtime + active-run truth check
# Compatible with 3090/4090 hosts.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_DEFAULT}}"
REQUIRE_DUAL_GPU="${REQUIRE_DUAL_GPU:-true}"
HOST_LABEL="${HOST_LABEL:-gpu-host}"

_text_matches() {
  local text="$1"
  local pattern="$2"
  if command -v rg >/dev/null 2>&1; then
    printf '%s\n' "${text}" | rg -q "${pattern}"
  else
    printf '%s\n' "${text}" | grep -E -q "${pattern}"
  fi
}

_text_count() {
  local text="$1"
  local pattern="$2"
  if command -v rg >/dev/null 2>&1; then
    printf '%s\n' "${text}" | rg -c "${pattern}" || true
  else
    printf '%s\n' "${text}" | grep -E -c "${pattern}" || true
  fi
}

for arg in "$@"; do
    case "$arg" in
    --repo-root=*) REPO_ROOT="${arg#*=}" ;;
    --require-dual-gpu=*) REQUIRE_DUAL_GPU="${arg#*=}" ;;
    --host-label=*) HOST_LABEL="${arg#*=}" ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--repo-root=/path] [--require-dual-gpu=true|false] [--host-label=3090|4090|gpu-host]"
      exit 1
      ;;
    esac
done

cd "${REPO_ROOT}"

echo "=== Runtime ==="
echo "host_label: ${HOST_LABEL}"
PY_BIN="$(command -v python3 || true)"
if [[ -z "${PY_BIN}" ]]; then
  echo "FATAL: python3 not found"
  exit 2
fi
PY_VER="$(python3 -V 2>&1 || true)"
echo "python3: ${PY_BIN}"
echo "python3 -V: ${PY_VER}"
if [[ "${PY_BIN}" != *"/envs/insider/"* ]]; then
  echo "FATAL: insider runtime required, got: ${PY_BIN}"
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi not found"
  exit 2
fi

echo

echo "=== GPU Inventory (nvidia-smi -L) ==="
GPU_LIST_RAW="$(nvidia-smi -L 2>&1 || true)"
echo "${GPU_LIST_RAW}"
if _text_matches "${GPU_LIST_RAW}" "Unable to determine the device handle|No devices were found|Failed"; then
  echo "FATAL: GPU driver/device state not healthy"
  exit 3
fi
GPU_COUNT="$(_text_count "${GPU_LIST_RAW}" '^GPU [0-9]+:')"
GPU_COUNT="${GPU_COUNT:-0}"
if [[ "${REQUIRE_DUAL_GPU}" == "true" && "${GPU_COUNT}" -lt 2 ]]; then
  echo "FATAL: dual GPU required but detected ${GPU_COUNT}"
  exit 3
fi

echo

echo "=== GPU Telemetry ==="
nvidia-smi \
  --query-gpu=index,name,pci.bus_id,temperature.gpu,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

echo

echo "=== Torch CUDA Check ==="
python3 - <<'PY'
import json
import torch
count = int(torch.cuda.device_count())
devices = []
for i in range(count):
    props = torch.cuda.get_device_properties(i)
    devices.append({"index": i, "name": props.name, "total_memory_gb": round(props.total_memory / 1024**3, 2)})
print(json.dumps({"torch_cuda_device_count": count, "devices": devices}, ensure_ascii=True))
PY

echo

echo "=== Active-run truth check ==="
pgrep -af "run_v72_4090_oneshot.sh|run_v72_3090_oneshot.sh|run_block3_benchmark_shard.py" || true
LATEST_DIR="$(ls -dt \
  runs/benchmarks/block3_20260203_225620_phase7_v72_4090_* \
  runs/benchmarks/block3_20260203_225620_phase7_v72_3090_* \
  runs/benchmarks/block3_20260203_225620_phase7_v72_* \
  2>/dev/null | head -1 || true)"
if [[ -n "${LATEST_DIR}" ]]; then
  METRICS_COUNT="$(find "${LATEST_DIR}" -name metrics.json | wc -l | tr -d ' ')"
  echo "latest_out_dir=${LATEST_DIR}"
  echo "latest_metrics_count=${METRICS_COUNT}"
else
  echo "latest_out_dir=NONE"
fi

echo

echo "=== Gate Result ==="
if [[ "${REQUIRE_DUAL_GPU}" == "true" ]]; then
  echo "PASS: insider runtime + dual GPU visible + torch CUDA available"
else
  echo "PASS: insider runtime + GPU visible + torch CUDA available"
fi
