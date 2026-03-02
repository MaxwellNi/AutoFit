#!/usr/bin/env bash
# ============================================================================
# Cancel duplicate Phase-7 jobs that are already covered by strict-comparable
# results and should not consume queue slots in the mainline comparison.
#
# Default target patterns:
#   - p7x_g03_*
#   - p7x_g04_*
#   - p7x_g05_*
#   - p7xF_afb_*
#
# Usage:
#   bash scripts/clean_phase7_duplicate_jobs.sh --dry-run
#   bash scripts/clean_phase7_duplicate_jobs.sh --apply
#   bash scripts/clean_phase7_duplicate_jobs.sh --apply --pending-only
# ============================================================================

set -euo pipefail

MODE="dry-run"
PENDING_ONLY=false

for arg in "$@"; do
  case "$arg" in
    --dry-run) MODE="dry-run" ;;
    --apply) MODE="apply" ;;
    --pending-only) PENDING_ONLY=true ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

QUEUE="$(squeue -u "${USER}" -h -o '%i|%T|%j' || true)"
if [[ -z "${QUEUE}" ]]; then
  echo "No jobs found for user ${USER}."
  exit 0
fi

mapfile -t CANDIDATES < <(
  printf '%s\n' "${QUEUE}" \
  | awk -F'|' '
      {
        id=$1; st=$2; name=$3;
        if (name ~ /^p7x_g0(3|4|5)_/ || name ~ /^p7xF_afb_/) {
          print id "|" st "|" name;
        }
      }
    '
)

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
  echo "No duplicate pattern jobs matched."
  exit 0
fi

IDS=()
echo "Matched duplicate jobs:"
for row in "${CANDIDATES[@]}"; do
  jid="${row%%|*}"
  rest="${row#*|}"
  st="${rest%%|*}"
  name="${row##*|}"
  if ${PENDING_ONLY} && [[ "${st}" != "PENDING" ]]; then
    continue
  fi
  IDS+=("${jid}")
  echo "  ${jid} | ${st} | ${name}"
done

if [[ "${#IDS[@]}" -eq 0 ]]; then
  echo "No jobs selected after filters."
  exit 0
fi

echo "Selected ${#IDS[@]} jobs."
if [[ "${MODE}" == "dry-run" ]]; then
  echo "[DRY-RUN] No jobs cancelled."
  exit 0
fi

scancel "${IDS[@]}"
echo "[APPLY] Cancelled ${#IDS[@]} jobs."
