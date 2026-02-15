#!/usr/bin/env bash
# =============================================================================
# Pull Block3 freeze-required runs assets from Iris to local repo (3090/4090)
#
# This script syncs only the assets referenced by docs/audits/FULL_SCALE_POINTER.yaml
# so that preflight verification can pass on local servers.
#
# Typical usage on 3090:
#   conda activate insider
#   bash scripts/pull_block3_freeze_from_iris.sh --iris-login npin@iris
#
# Optional benchmark outputs:
#   bash scripts/pull_block3_freeze_from_iris.sh --iris-login npin@iris --with-benchmarks
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LOCAL_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_REPO="${REPO_ROOT:-${DEFAULT_LOCAL_REPO}}"

IRIS_LOGIN="${IRIS_LOGIN:-iris}"
IRIS_REPO="${IRIS_REPO:-/home/users/npin/repo_root}"
POINTER_REL="docs/audits/FULL_SCALE_POINTER.yaml"
WITH_BENCHMARKS=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --iris-login=*)
            IRIS_LOGIN="${arg#*=}"
            ;;
        --iris-repo=*)
            IRIS_REPO="${arg#*=}"
            ;;
        --local-repo=*)
            LOCAL_REPO="${arg#*=}"
            ;;
        --with-benchmarks)
            WITH_BENCHMARKS=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--iris-login=user@host] [--iris-repo=/path] [--local-repo=/path] [--with-benchmarks] [--dry-run]"
            exit 1
            ;;
    esac
done

if [[ ! -d "${LOCAL_REPO}" ]]; then
    echo "FATAL: local repo not found: ${LOCAL_REPO}"
    exit 2
fi
if [[ ! -f "${LOCAL_REPO}/pyproject.toml" ]]; then
    echo "FATAL: local repo does not look valid (missing pyproject.toml): ${LOCAL_REPO}"
    exit 2
fi

cd "${LOCAL_REPO}"

RSYNC_BASE=(rsync -aH --partial --append-verify --info=progress2 --human-readable)
if $DRY_RUN; then
    RSYNC_BASE+=(--dry-run)
fi

POINTER_LOCAL="${LOCAL_REPO}/${POINTER_REL}"
mkdir -p "$(dirname "${POINTER_LOCAL}")"

echo "=== Pull pointer from Iris ==="
"${RSYNC_BASE[@]}" "${IRIS_LOGIN}:${IRIS_REPO}/${POINTER_REL}" "${POINTER_LOCAL}"

if [[ ! -f "${POINTER_LOCAL}" ]]; then
    echo "FATAL: pointer not available after sync: ${POINTER_LOCAL}"
    exit 2
fi

mapfile -t REQUIRED_REL < <(python3 - "${POINTER_LOCAL}" <<'PY'
import sys
from pathlib import Path
import yaml

p = Path(sys.argv[1])
d = yaml.safe_load(p.read_text(encoding="utf-8"))

keys = [
    ("offers_core_daily", "dir"),
    ("offers_core_snapshot", "dir"),
    ("offers_text", "dir"),
    ("edgar_store_full_daily", "dir"),
    ("multiscale_full", "dir"),
    ("snapshots_index", "offer_day"),
    ("snapshots_index", "cik_day"),
    ("analysis", "dir"),
]

vals = []
for a, b in keys:
    v = d.get(a, {}).get(b, "")
    if isinstance(v, str) and v.startswith("runs/"):
        vals.append(v.rstrip("/"))

# Deduplicate and drop children if parent already present
uniq = sorted(set(vals), key=lambda x: (x.count("/"), x))
kept = []
for cur in uniq:
    if any(cur.startswith(parent + "/") for parent in kept):
        continue
    kept.append(cur)

for x in kept:
    print(x)
PY
)

if [[ "${#REQUIRED_REL[@]}" -eq 0 ]]; then
    echo "FATAL: no required runs paths found in pointer."
    exit 2
fi

echo "=== Sync freeze-required runs assets ==="
for rel in "${REQUIRED_REL[@]}"; do
    local_parent="${LOCAL_REPO}/$(dirname "${rel}")"
    mkdir -p "${local_parent}"
    echo "-> ${rel}"
    "${RSYNC_BASE[@]}" "${IRIS_LOGIN}:${IRIS_REPO}/${rel}" "${local_parent}/"
done

if $WITH_BENCHMARKS; then
    echo "=== Sync benchmark outputs (optional) ==="
    BENCH_ROOT="runs/benchmarks"
    mkdir -p "${LOCAL_REPO}/${BENCH_ROOT}"

    # Canonical Phase7 and V7.1 runs
    for pat in \
        "block3_20260203_225620_phase7" \
        "block3_20260203_225620_phase7_v71extreme_*" \
        "block3_preflight_v71_*"
    do
        echo "-> ${BENCH_ROOT}/${pat}"
        "${RSYNC_BASE[@]}" --prune-empty-dirs --include='*/' --include='*.json' --include='*.csv' --include='*.md' --exclude='*' \
            "${IRIS_LOGIN}:${IRIS_REPO}/${BENCH_ROOT}/${pat}" "${LOCAL_REPO}/${BENCH_ROOT}/" || true
    done
fi

echo "=== Done ==="
echo "Local repo: ${LOCAL_REPO}"
echo "Iris source: ${IRIS_LOGIN}:${IRIS_REPO}"

echo "You can now run:"
echo "  bash scripts/preflight_block3_v71_gate.sh --v71-variant=g02 --skip-smoke"
