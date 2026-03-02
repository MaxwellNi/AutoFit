#!/usr/bin/env bash
# =============================================================================
# Phase 7 Benchmark — Live Monitor
#
# Shows real-time progress of Phase 7 SLURM jobs.
#
# Usage:
#   bash scripts/phase7_monitor.sh           # one-shot
#   watch -n 60 bash scripts/phase7_monitor.sh  # auto-refresh
# =============================================================================
set -euo pipefail

OUTROOT="/work/projects/eint/repo_root/runs/benchmarks/block3_20260203_225620_iris_phase7"

echo "============================================================"
echo " Phase 7 Benchmark Monitor — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ── SLURM Queue Status ──────────────────────────────────────────────────────
echo "━━━ SLURM Queue ━━━"
running=$(squeue -u "$USER" -t R -n "p7_%" --noheader 2>/dev/null | wc -l)
pending=$(squeue -u "$USER" -t PD -n "p7_%" --noheader 2>/dev/null | wc -l)
echo "  Running: ${running}"
echo "  Pending: ${pending}"
echo ""

# ── Results Progress ─────────────────────────────────────────────────────────
if [[ -d "$OUTROOT" ]]; then
    echo "━━━ Results Progress ━━━"

    total_metrics=0
    total_models=0
    completed_shards=0
    partial_shards=0
    failed_shards=0

    while IFS= read -r mf; do
        dir="$(dirname "$mf")"
        manifest="${dir}/MANIFEST.json"

        # Count metrics
        n=$(python3 -c "import json; print(len(json.load(open('$mf'))))" 2>/dev/null || echo 0)
        total_metrics=$((total_metrics + n))

        # Check manifest status
        if [[ -f "$manifest" ]]; then
            status=$(python3 -c "import json; print(json.load(open('$manifest')).get('status','?'))" 2>/dev/null || echo "?")
            case "$status" in
                completed) completed_shards=$((completed_shards + 1)) ;;
                partial*)  partial_shards=$((partial_shards + 1)) ;;
                failed)    failed_shards=$((failed_shards + 1)) ;;
            esac
        fi
    done < <(find "$OUTROOT" -name metrics.json 2>/dev/null)

    # Count unique models
    if [[ $total_metrics -gt 0 ]]; then
        total_models=$(find "$OUTROOT" -name metrics.json -exec python3 -c "
import json, sys
models = set()
for f in sys.argv[1:]:
    try:
        for r in json.load(open(f)):
            models.add(r.get('model_name',''))
    except: pass
print(len(models))
" {} + 2>/dev/null || echo 0)
    fi

    echo "  Completed shards: ${completed_shards}"
    echo "  Partial shards:   ${partial_shards}"
    echo "  Failed shards:    ${failed_shards}"
    echo "  Total metrics:    ${total_metrics}"
    echo "  Unique models:    ${total_models} / 67"
    echo ""

    # ── Per-Category Breakdown ───────────────────────────────────────────
    echo "━━━ Per-Category Breakdown ━━━"
    for cat in ml_tabular statistical deep_classical transformer_sota foundation irregular autofit; do
        cat_metrics=$(find "$OUTROOT" -path "*${cat}*" -name metrics.json -exec python3 -c "
import json, sys
n=0
for f in sys.argv[1:]:
    try: n += len(json.load(open(f)))
    except: pass
print(n)
" {} + 2>/dev/null || echo 0)
        cat_status="—"
        if [[ $cat_metrics -gt 0 ]]; then
            cat_status="${cat_metrics} records"
        fi
        printf "  %-20s %s\n" "$cat" "$cat_status"
    done
    echo ""

    # ── Recent Failures ──────────────────────────────────────────────────
    fail_count=$(find "$OUTROOT" -name MANIFEST.json -exec grep -l '"failed"' {} \; 2>/dev/null | wc -l)
    if [[ $fail_count -gt 0 ]]; then
        echo "━━━ Failed Shards (${fail_count}) ━━━"
        find "$OUTROOT" -name MANIFEST.json -exec grep -l '"failed"' {} \; 2>/dev/null | while read mf; do
            err=$(python3 -c "import json; d=json.load(open('$mf')); print(f\"  {d.get('category','?'):20s} {d.get('ablation','?'):12s} — {d.get('error','?')[:80]}\")" 2>/dev/null)
            echo "$err"
        done
        echo ""
    fi
else
    echo "  No results directory yet: ${OUTROOT}"
    echo ""
fi

# ── Recent SLURM Errors ─────────────────────────────────────────────────────
echo "━━━ Recent SLURM Errors (last 5) ━━━"
errfiles=$(ls -t /work/projects/eint/logs/phase7/p7_*.err 2>/dev/null | head -5)
if [[ -n "$errfiles" ]]; then
    for ef in $errfiles; do
        sz=$(wc -c < "$ef")
        if [[ $sz -gt 0 ]]; then
            echo "  $(basename "$ef"):"
            tail -3 "$ef" | sed 's/^/    /'
        fi
    done
else
    echo "  No error logs yet."
fi
echo ""
echo "============================================================"
