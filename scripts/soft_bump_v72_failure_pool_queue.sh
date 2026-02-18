#!/usr/bin/env bash
# ============================================================================
# Soft-bump scheduler helper for V7.2 failure-pool jobs.
# ============================================================================
#
# Goal:
#   Accelerate the four fixed V7.2 failure-pool reruns without changing
#   benchmark fairness semantics (data/split/metrics/model definition).
#
# Strategy:
#   Stage t0:   push p7r_v72_ic_ce_h* jobs to top of user's queue.
#   Stage t2h:  if still not running, hold a small batch of non-core pending
#               jobs (p7x/p7xF) so the p7r jobs can start when slots free up.
#   Release:    release previously held non-core jobs.
#
# Usage:
#   bash scripts/soft_bump_v72_failure_pool_queue.sh --stage=t0 --dry-run
#   bash scripts/soft_bump_v72_failure_pool_queue.sh --stage=t0 --apply
#   bash scripts/soft_bump_v72_failure_pool_queue.sh --stage=t2h --apply
#   bash scripts/soft_bump_v72_failure_pool_queue.sh --release-held --apply
# ============================================================================

set -euo pipefail

STAGE="t0"
MODE="dry-run"
RELEASE_HELD=false
HOLD_COUNT=8
USER_NAME="${USER:-$(id -un)}"

P7R_REGEX='^p7r_v72_ic_ce_h(1|7|14|30)_heavy$'
NON_CORE_REGEX='^(p7x|p7xF)'

for arg in "$@"; do
    case "$arg" in
        --stage=*)
            STAGE="${arg#*=}"
            ;;
        --hold-count=*)
            HOLD_COUNT="${arg#*=}"
            ;;
        --release-held)
            RELEASE_HELD=true
            ;;
        --apply)
            MODE="apply"
            ;;
        --dry-run)
            MODE="dry-run"
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

if [[ "${STAGE}" != "t0" && "${STAGE}" != "t2h" ]]; then
    echo "Invalid --stage=${STAGE} (expected t0 or t2h)"
    exit 1
fi

if ! command -v squeue >/dev/null 2>&1; then
    echo "FATAL: squeue not found."
    exit 2
fi
if ! command -v scontrol >/dev/null 2>&1; then
    echo "FATAL: scontrol not found."
    exit 2
fi

echo "Mode: ${MODE}"
echo "User: ${USER_NAME}"
echo "Stage: ${STAGE}"
echo "Release held: ${RELEASE_HELD}"
echo "Hold count: ${HOLD_COUNT}"

mapfile -t QUEUE_LINES < <(squeue -u "${USER_NAME}" -h -o '%A|%j|%T|%r')

get_p7r_ids_by_state() {
    local wanted_state="$1"
    local line id name state
    for line in "${QUEUE_LINES[@]}"; do
        IFS='|' read -r id name state _reason <<< "${line}"
        if [[ "${state}" == "${wanted_state}" && "${name}" =~ ${P7R_REGEX} ]]; then
            echo "${id}"
        fi
    done
}

if ${RELEASE_HELD}; then
    mapfile -t HELD_IDS < <(
        for line in "${QUEUE_LINES[@]}"; do
            IFS='|' read -r id name state reason <<< "${line}"
            if [[ "${state}" == "PENDING" && "${reason}" == "JobHeldUser" && "${name}" =~ ${NON_CORE_REGEX} ]]; then
                echo "${id}"
            fi
        done
    )
    if [[ "${#HELD_IDS[@]}" -eq 0 ]]; then
        echo "No held non-core jobs to release."
        exit 0
    fi
    echo "Release candidates (${#HELD_IDS[@]}): ${HELD_IDS[*]}"
    if [[ "${MODE}" == "apply" ]]; then
        scontrol release "${HELD_IDS[@]}"
        echo "Released ${#HELD_IDS[@]} jobs."
    else
        echo "[DRY] scontrol release ${HELD_IDS[*]}"
    fi
    exit 0
fi

mapfile -t P7R_PENDING_IDS < <(get_p7r_ids_by_state "PENDING")
mapfile -t P7R_RUNNING_IDS < <(get_p7r_ids_by_state "RUNNING")

echo "p7r pending: ${#P7R_PENDING_IDS[@]} ${P7R_PENDING_IDS[*]:-}"
echo "p7r running: ${#P7R_RUNNING_IDS[@]} ${P7R_RUNNING_IDS[*]:-}"

if [[ "${#P7R_PENDING_IDS[@]}" -eq 0 && "${#P7R_RUNNING_IDS[@]}" -eq 0 ]]; then
    echo "No p7r_v72 failure-pool jobs found in queue."
    exit 0
fi

if [[ "${STAGE}" == "t0" ]]; then
    if [[ "${#P7R_PENDING_IDS[@]}" -eq 0 ]]; then
        echo "No pending p7r jobs to bump."
        exit 0
    fi
    if [[ "${MODE}" == "apply" ]]; then
        top_fail=0
        for jid in "${P7R_PENDING_IDS[@]}"; do
            if ! scontrol top "${jid}" >/dev/null 2>&1; then
                top_fail=$((top_fail + 1))
            fi
        done
        if (( top_fail == 0 )); then
            echo "Applied queue top bump to p7r jobs."
        else
            echo "WARNING: scontrol top denied for ${top_fail} job(s)."
            echo "Use stage=t2h to apply soft hold fallback:"
            echo "  bash scripts/soft_bump_v72_failure_pool_queue.sh --stage=t2h --apply"
        fi
    else
        echo "[DRY] scontrol top ${P7R_PENDING_IDS[*]}"
    fi
    exit 0
fi

# Stage t2h: conditional hold of non-core pending jobs if p7r still blocked.
if [[ "${#P7R_RUNNING_IDS[@]}" -gt 0 ]]; then
    echo "At least one p7r job is already running; skip hold step."
    exit 0
fi
if [[ "${#P7R_PENDING_IDS[@]}" -eq 0 ]]; then
    echo "No pending p7r jobs; nothing to accelerate."
    exit 0
fi

min_p7r_id="${P7R_PENDING_IDS[0]}"
for jid in "${P7R_PENDING_IDS[@]}"; do
    if (( jid < min_p7r_id )); then
        min_p7r_id="${jid}"
    fi
done

mapfile -t HOLD_CANDIDATES < <(
    for line in "${QUEUE_LINES[@]}"; do
        IFS='|' read -r id name state reason <<< "${line}"
        if [[ "${state}" != "PENDING" ]]; then
            continue
        fi
        if [[ "${reason}" == "JobHeldUser" ]]; then
            continue
        fi
        if [[ ! "${name}" =~ ${NON_CORE_REGEX} ]]; then
            continue
        fi
        if (( id >= min_p7r_id )); then
            continue
        fi
        echo "${id}|${name}"
    done | sort -t'|' -k1,1n | head -n "${HOLD_COUNT}" | cut -d'|' -f1
)

if [[ "${#HOLD_CANDIDATES[@]}" -eq 0 ]]; then
    echo "No eligible non-core pending jobs to hold."
    exit 0
fi

echo "Hold candidates (${#HOLD_CANDIDATES[@]}): ${HOLD_CANDIDATES[*]}"
if [[ "${MODE}" == "apply" ]]; then
    scontrol hold "${HOLD_CANDIDATES[@]}"
    echo "Applied hold to ${#HOLD_CANDIDATES[@]} non-core jobs."
    echo "After at least one p7r starts, release with:"
    echo "  bash scripts/soft_bump_v72_failure_pool_queue.sh --release-held --apply"
else
    echo "[DRY] scontrol hold ${HOLD_CANDIDATES[*]}"
    echo "[DRY] release command:"
    echo "[DRY] bash scripts/soft_bump_v72_failure_pool_queue.sh --release-held --apply"
fi
