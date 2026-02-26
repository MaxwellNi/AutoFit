#!/usr/bin/env bash
# =============================================================================
# Pull V7.3 handoff package from Iris to 4090/3090 (one-way: local host -> iris)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_REPO="${REPO_ROOT:-${DEFAULT_REPO}}"

IRIS_LOGIN="${IRIS_LOGIN:-iris}"
IRIS_REPO="${IRIS_REPO:-/home/users/npin/repo_root}"
WITH_BENCHMARKS=true
WITH_PREFLIGHT=false
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
        --with-benchmarks=*)
            WITH_BENCHMARKS="${arg#*=}"
            ;;
        --with-preflight=*)
            WITH_PREFLIGHT="${arg#*=}"
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--iris-login=user@host] [--iris-repo=/path] [--local-repo=/path] [--with-benchmarks=true|false] [--with-preflight=true|false] [--dry-run]"
            exit 1
            ;;
    esac
done

if [[ ! -d "${LOCAL_REPO}" ]]; then
    echo "FATAL: local repo does not exist: ${LOCAL_REPO}"
    exit 2
fi
if [[ ! -f "${LOCAL_REPO}/pyproject.toml" ]]; then
    echo "FATAL: local repo missing pyproject.toml: ${LOCAL_REPO}"
    exit 2
fi

cd "${LOCAL_REPO}"

RSYNC_RSH=(
    ssh -T
    -o Compression=no
    -c aes128-gcm@openssh.com
    -o ControlMaster=auto
    -o ControlPersist=10m
    -o ControlPath="${HOME}/.ssh/cm-%r@%h:%p"
)

RSYNC_BASE=(
    rsync
    -rlt
    --partial
    --whole-file
    --omit-dir-times
    --no-perms
    --no-owner
    --no-group
    --info=progress2,stats2
    --human-readable
    -e "${RSYNC_RSH[*]}"
)

if ${DRY_RUN}; then
    RSYNC_BASE+=(--dry-run)
fi

sync_dir_mirror() {
    local src="$1"
    local dst="$2"
    mkdir -p "${dst}"
    "${RSYNC_BASE[@]}" --delete --delete-delay "${src%/}/" "${dst%/}/"
}

sync_dir_append() {
    local src="$1"
    local dst="$2"
    mkdir -p "${dst}"
    "${RSYNC_BASE[@]}" "${src%/}/" "${dst%/}/"
}

sync_file() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "${dst}")"
    "${RSYNC_BASE[@]}" "${src}" "${dst}"
}

echo "=== Pull code and docs delta (non-GitHub handoff) ==="
for rel in docs scripts configs src; do
    echo "-> ${rel}/"
    sync_dir_append "${IRIS_LOGIN}:${IRIS_REPO}/${rel}" "${LOCAL_REPO}/${rel}"
done

echo "=== Pull freeze pointer and required assets ==="
sync_file "${IRIS_LOGIN}:${IRIS_REPO}/docs/audits/FULL_SCALE_POINTER.yaml" "${LOCAL_REPO}/docs/audits/FULL_SCALE_POINTER.yaml"

FREEZE_PATHS=(
    "runs/offers_core_full_daily_wide_20260203_225620"
    "runs/offers_core_full_snapshot_wide_20260203_225620"
    "runs/offers_text_v1_20260129_073037_full"
    "runs/edgar_feature_store_full_daily_wide_20260203_225620"
    "runs/multiscale_full_wide_20260203_225620"
    "runs/orchestrator/20260129_073037/analysis/wide_20260203_225620"
)
for rel in "${FREEZE_PATHS[@]}"; do
    echo "-> ${rel}"
    sync_dir_mirror "${IRIS_LOGIN}:${IRIS_REPO}/${rel}" "${LOCAL_REPO}/${rel}"
done

echo "=== Pull truth-pack latest artifacts ==="
sync_dir_append \
    "${IRIS_LOGIN}:${IRIS_REPO}/docs/benchmarks/block3_truth_pack" \
    "${LOCAL_REPO}/docs/benchmarks/block3_truth_pack"

echo "=== Pull benchmark results delta (strict-comparable reuse scope) ==="
if [[ "${WITH_BENCHMARKS}" == "true" ]]; then
    mkdir -p "${LOCAL_REPO}/runs/benchmarks"
    "${RSYNC_BASE[@]}" \
        --prune-empty-dirs \
        --include='block3_20260203_225620*/' \
        --include='block3_20260203_225620*/**' \
        --exclude='block3_preflight_*' \
        --exclude='*' \
        "${IRIS_LOGIN}:${IRIS_REPO}/runs/benchmarks/" \
        "${LOCAL_REPO}/runs/benchmarks/"

    if [[ "${WITH_PREFLIGHT}" == "true" ]]; then
        "${RSYNC_BASE[@]}" \
            --prune-empty-dirs \
            --include='block3_preflight_v71_*/' \
            --include='block3_preflight_v71_*/**' \
            --exclude='*' \
            "${IRIS_LOGIN}:${IRIS_REPO}/runs/benchmarks/" \
            "${LOCAL_REPO}/runs/benchmarks/"
    fi
fi

echo "=== Done ==="
echo "Local repo: ${LOCAL_REPO}"
echo "Iris source: ${IRIS_LOGIN}:${IRIS_REPO}"
echo
echo "Recommended verification:"
echo "  python3 scripts/block3_verify_freeze.py"
echo "  bash scripts/preflight_block3_v71_gate.sh --v71-variant=g02 --skip-smoke --skip-audits"
echo "  python3 scripts/build_block3_truth_pack.py --include-freeze-history --capture-slurm --slurm-since 2026-02-12 --update-master-doc"

