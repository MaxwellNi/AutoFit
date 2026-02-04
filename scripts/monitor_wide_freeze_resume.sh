#!/usr/bin/env bash
set -euo pipefail

WIDE_STAMP="${WIDE_STAMP:-20260203_225620}"
INTERVAL_SEC="${INTERVAL_SEC:-900}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-scripts/slurm/run_wide_freeze_aion_from_daily.sbatch}"
FULL_SRUN_SCRIPT="${FULL_SRUN_SCRIPT:-scripts/slurm/run_wide_freeze_aion.sbatch}"
JOB_NAME="${JOB_NAME:-wide_freeze_aion_resume}"
LOG_DIR="${LOG_DIR:-runs/slurm}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/monitor_wide_freeze_${WIDE_STAMP}.log}"
STATE_FILE="${STATE_FILE:-${LOG_DIR}/monitor_wide_freeze_${WIDE_STAMP}.state}"
WIDE_ANALYSIS="${WIDE_ANALYSIS:-runs/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}}"

SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-16}"
MAX_RESUBMITS="${MAX_RESUBMITS:-5}"

mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG_FILE}"
}

load_state() {
  if [ -f "${STATE_FILE}" ]; then
    # shellcheck source=/dev/null
    source "${STATE_FILE}"
  fi
  LAST_JOBID="${LAST_JOBID:-}"
  RESUBMITS="${RESUBMITS:-0}"
  LAST_REASON="${LAST_REASON:-}"
}

save_state() {
  cat > "${STATE_FILE}" <<EOF
LAST_JOBID=${LAST_JOBID:-}
RESUBMITS=${RESUBMITS:-0}
LAST_REASON=${LAST_REASON:-}
SBATCH_CPUS_PER_TASK=${SBATCH_CPUS_PER_TASK:-16}
EOF
}

calc_mem_limit_gb() {
  local mem_mb
  mem_mb="$(sinfo -N -p batch -o "%m" | sed -n '2p' || true)"
  if [ -n "${mem_mb}" ]; then
    echo $(( (mem_mb / 1024) * 60 / 100 ))
  else
    echo 64
  fi
}

get_current_job() {
  local max_id="" max_state="" id state name
  while IFS='|' read -r id state name; do
    [ -z "${id}" ] && continue
    if [[ "${name}" == "${JOB_NAME}"* ]]; then
      if [ -z "${max_id}" ] || [ "${id}" -gt "${max_id}" ]; then
        max_id="${id}"
        max_state="${state}"
      fi
    fi
  done < <(squeue -u "${USER}" -h -o "%i|%T|%j")

  if [ -n "${max_id}" ]; then
    echo "${max_id}|${max_state}"
  fi
}

check_outputs() {
  local step4=0 step5=0 step6=0 step7=0 step8=0 step9=0 step10=0 pointer_ok=0

  if [ -f "runs/offers_core_full_daily_wide_${WIDE_STAMP}/offers_core_daily.parquet" ] && \
     [ -f "runs/offers_core_full_daily_wide_${WIDE_STAMP}/MANIFEST.json" ]; then
    step4=1
  fi
  if [ -d "runs/offers_core_full_daily_wide_${WIDE_STAMP}/snapshots_index" ]; then
    step5=1
  fi
  if [ -d "runs/edgar_feature_store_full_daily_wide_${WIDE_STAMP}/edgar_features" ]; then
    step6=1
  fi
  if [ -f "runs/multiscale_full_wide_${WIDE_STAMP}/MANIFEST.json" ]; then
    step7=1
  fi
  if [ -f "${WIDE_ANALYSIS}/column_manifest.json" ] && [ -f "${WIDE_ANALYSIS}/column_manifest.md" ]; then
    step8=1
  fi
  if [ -f "${WIDE_ANALYSIS}/raw_cardinality_coverage_wide_${WIDE_STAMP}.json" ] && \
     [ -f "${WIDE_ANALYSIS}/raw_cardinality_coverage_wide_${WIDE_STAMP}.md" ]; then
    step9=1
  fi
  if [ -f "${WIDE_ANALYSIS}/freeze_candidates.json" ] && [ -f "${WIDE_ANALYSIS}/freeze_candidates.md" ]; then
    step10=1
  fi
  if [ -f "docs/audits/FULL_SCALE_POINTER.yaml" ] && \
     grep -q "${WIDE_STAMP}" "docs/audits/FULL_SCALE_POINTER.yaml"; then
    pointer_ok=1
  fi

  log "Steps: 4=${step4} 5=${step5} 6=${step6} 7=${step7} 8=${step8} 9=${step9} 10=${step10} pointer=${pointer_ok}"

  if [ "${step4}" -eq 1 ] && [ "${step5}" -eq 1 ] && [ "${step6}" -eq 1 ] && \
     [ "${step7}" -eq 1 ] && [ "${step8}" -eq 1 ] && [ "${step9}" -eq 1 ] && \
     [ "${step10}" -eq 1 ] && [ "${pointer_ok}" -eq 1 ]; then
    return 0
  fi
  return 1
}

classify_failure() {
  local jobid="$1"
  local err="runs/slurm/${JOB_NAME}_${jobid}.err"
  if [ -f "${err}" ]; then
    if grep -Eiq "Out Of Memory|oom_kill|OOM" "${err}"; then
      echo "OOM"; return
    fi
    if grep -Eiq "bash: No such file or directory" "${err}"; then
      echo "BASH"; return
    fi
    if grep -Eiq "ModuleNotFoundError: No module named 'duckdb'|ModuleNotFoundError: No module named 'deltalake'" "${err}"; then
      echo "DEPS"; return
    fi
    if grep -Eiq "snapshot not found" "${err}"; then
      echo "SNAPSHOT_MISSING"; return
    fi
  fi
  echo "GENERIC"
}

submit_job() {
  local export_vars="ALL,WIDE_STAMP=${WIDE_STAMP}"
  if [ -n "${DUCKDB_THREADS_OVERRIDE:-}" ]; then
    export_vars="${export_vars},DUCKDB_THREADS=${DUCKDB_THREADS_OVERRIDE}"
  fi
  if [ -n "${DUCKDB_MEMORY_LIMIT_GB_OVERRIDE:-}" ]; then
    export_vars="${export_vars},DUCKDB_MEMORY_LIMIT_GB=${DUCKDB_MEMORY_LIMIT_GB_OVERRIDE}"
  fi

  local args=()
  if [ -n "${SBATCH_CPUS_PER_TASK}" ]; then
    args+=("-c" "${SBATCH_CPUS_PER_TASK}")
  fi

  local output=""
  if ! output=$(sbatch "${args[@]}" --export="${export_vars}" "${SBATCH_SCRIPT}" 2>&1); then
    log "sbatch failed: ${output}"
    if [[ "${output}" == *"CPU count per node can not be satisfied"* ]] || \
       [[ "${output}" == *"Requested node configuration is not available"* ]]; then
      if [ "${SBATCH_CPUS_PER_TASK}" -gt 8 ]; then
        SBATCH_CPUS_PER_TASK=8
        log "Retrying sbatch with -c ${SBATCH_CPUS_PER_TASK}"
        submit_job
        return
      fi
    fi
    return 1
  fi

  local jobid="${output##* }"
  LAST_JOBID="${jobid}"
  log "Submitted batch job ${jobid} (export=${export_vars}, -c ${SBATCH_CPUS_PER_TASK})"
  save_state
  return 0
}

auto_fix() {
  local reason="$1"
  RESUBMITS=$((RESUBMITS + 1))
  LAST_REASON="${reason}"
  if [ "${RESUBMITS}" -gt "${MAX_RESUBMITS}" ]; then
    log "Max resubmits (${MAX_RESUBMITS}) reached; stopping auto-fix."
    save_state
    exit 2
  fi

  DUCKDB_THREADS_OVERRIDE=""
  DUCKDB_MEMORY_LIMIT_GB_OVERRIDE=""
  SBATCH_SCRIPT="${SBATCH_SCRIPT:-scripts/slurm/run_wide_freeze_aion_from_daily.sbatch}"

  case "${reason}" in
    OOM)
      DUCKDB_THREADS_OVERRIDE=4
      DUCKDB_MEMORY_LIMIT_GB_OVERRIDE="$(calc_mem_limit_gb)"
      ;;
    CPU)
      SBATCH_CPUS_PER_TASK=8
      ;;
    BASH)
      ;;
    DEPS)
      ;;
    SNAPSHOT_MISSING)
      SBATCH_SCRIPT="${FULL_SRUN_SCRIPT}"
      ;;
    *)
      ;;
  esac

  log "Auto-fix reason=${reason} resubmit #${RESUBMITS}"
  save_state
  submit_job || true
}

load_state
log "Monitoring start: WIDE_STAMP=${WIDE_STAMP}, interval=${INTERVAL_SEC}s, job=${JOB_NAME}"

while true; do
  current="$(get_current_job || true)"
  if [ -n "${current}" ]; then
    IFS='|' read -r jobid state <<< "${current}"
    LAST_JOBID="${jobid}"
    save_state
    log "Job ${jobid} state=${state}"
    if [ "${state}" = "R" ]; then
      check_outputs && { log "All steps complete. Exiting monitor."; exit 0; }
    fi
  else
    if [ -n "${LAST_JOBID}" ]; then
      sacct_state=""
      while IFS='|' read -r id state exit_code; do
        if [ "${id}" = "${LAST_JOBID}" ]; then
          sacct_state="${state}"
          break
        fi
      done < <(sacct -j "${LAST_JOBID}" -P -n -o JobIDRaw,State,ExitCode 2>/dev/null || true)

      if [ -n "${sacct_state}" ]; then
        log "Last job ${LAST_JOBID} sacct_state=${sacct_state}"
        case "${sacct_state}" in
          COMPLETED)
            check_outputs && { log "All steps complete. Exiting monitor."; exit 0; }
            ;;
          OUT_OF_MEMORY|FAILED|CANCELLED|TIMEOUT)
            reason="$(classify_failure "${LAST_JOBID}")"
            auto_fix "${reason}"
            ;;
          *)
            ;;
        esac
      else
        log "No active job and no sacct info; submitting job."
        submit_job || true
      fi
    else
      log "No active job found; submitting job."
      submit_job || true
    fi
  fi

  sleep "${INTERVAL_SEC}"
done
