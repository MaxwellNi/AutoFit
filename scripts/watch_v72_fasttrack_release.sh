#!/usr/bin/env bash
# ============================================================================
# Auto-release watcher for temporary holds used in V7.2 fast-track.
# ============================================================================
#
# Behavior:
#   - Polls queue state.
#   - Once any p7r_v72_ic_ce_h*_heavy job starts RUNNING, it releases:
#       1) Held non-core jobs (p7x/p7xF) via soft_bump helper
#       2) Temporarily held core jobs that were put ahead of v72
#
# Usage:
#   bash scripts/watch_v72_fasttrack_release.sh
#   bash scripts/watch_v72_fasttrack_release.sh --interval=120 --max-wait-min=2880
#
# Notes:
#   - No cancellations are performed.
#   - Safe to run multiple times; release commands are idempotent.
# ============================================================================

set -euo pipefail

USER_NAME="${USER:-$(id -un)}"
INTERVAL_SEC=60
MAX_WAIT_MIN=1440

CORE_FASTTRACK_NAMES=(
  "p7_sta_t1_ct"
  "p7r_af1_t1_ce"
  "p7r_af2_t1_ce"
)

for arg in "$@"; do
  case "$arg" in
    --interval=*)
      INTERVAL_SEC="${arg#*=}"
      ;;
    --max-wait-min=*)
      MAX_WAIT_MIN="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

if ! command -v squeue >/dev/null 2>&1; then
  echo "FATAL: squeue not found."
  exit 2
fi
if ! command -v scontrol >/dev/null 2>&1; then
  echo "FATAL: scontrol not found."
  exit 2
fi

echo "Watching for V7.2 start..."
echo "User: ${USER_NAME}"
echo "Interval: ${INTERVAL_SEC}s"
echo "Max wait: ${MAX_WAIT_MIN} min"

start_ts="$(date +%s)"
timeout_sec="$(( MAX_WAIT_MIN * 60 ))"
loop_count=0

while true; do
  loop_count=$(( loop_count + 1 ))
  now_ts="$(date +%s)"
  elapsed="$(( now_ts - start_ts ))"
  if (( elapsed > timeout_sec )); then
    echo "Timeout reached without V7.2 RUNNING. Exiting."
    exit 3
  fi

  mapfile -t v72_running < <(
    squeue -u "${USER_NAME}" -h -o '%j|%T' \
      | awk -F'|' '$1 ~ /^p7r_v72_ic_ce_h(1|7|14|30)_heavy$/ && $2=="RUNNING" {print $1}'
  )
  mapfile -t v72_badconstraints < <(
    squeue -u "${USER_NAME}" -h -o '%A|%j|%T|%R' \
      | awk -F'|' '$2 ~ /^p7r_v72_ic_ce_h(1|7|14|30)_heavy$/ && $3=="PENDING" && $4=="(BadConstraints)" {print $1 "|" $2}'
  )

  if (( ${#v72_badconstraints[@]} > 0 )); then
    echo "Detected p7r jobs blocked by BadConstraints:"
    printf '  %s\n' "${v72_badconstraints[@]}"
    echo "Releasing temporary holds to avoid deadlock..."
    bash scripts/soft_bump_v72_failure_pool_queue.sh --release-held --apply || true
    echo "Fix submit resources, then resubmit p7r jobs."
    exit 4
  fi

  if (( loop_count % 10 == 0 )); then
    mapfile -t v72_pending < <(
      squeue -u "${USER_NAME}" -h -o '%j|%T|%R' \
        | awk -F'|' '$1 ~ /^p7r_v72_ic_ce_h(1|7|14|30)_heavy$/ && $2=="PENDING" {print $1" "$3}'
    )
    echo "[heartbeat $(date -Iseconds)] pending=${#v72_pending[@]} running=${#v72_running[@]}"
  fi

  if (( ${#v72_running[@]} > 0 )); then
    echo "Detected V7.2 RUNNING jobs: ${v72_running[*]}"
    echo "Releasing temporary holds..."

    # Release p7x/p7xF holds.
    bash scripts/soft_bump_v72_failure_pool_queue.sh --release-held --apply || true

    # Release explicitly held core jobs by name.
    mapfile -t core_hold_ids < <(
      squeue -u "${USER_NAME}" -h -o '%A|%j|%T|%R' \
        | awk -F'|' '$3=="PENDING" && $4=="(JobHeldUser)" {print $1 "|" $2}'
    )
    ids_to_release=()
    for row in "${core_hold_ids[@]}"; do
      jid="${row%%|*}"
      jname="${row#*|}"
      for cname in "${CORE_FASTTRACK_NAMES[@]}"; do
        if [[ "${jname}" == "${cname}" ]]; then
          ids_to_release+=("${jid}")
        fi
      done
    done
    if (( ${#ids_to_release[@]} > 0 )); then
      scontrol release "${ids_to_release[@]}" || true
      echo "Released core fast-track holds: ${ids_to_release[*]}"
    else
      echo "No core fast-track holds found."
    fi

    echo "Done."
    exit 0
  fi

  sleep "${INTERVAL_SEC}"
done
