# Minimal Current Handoff

> Last verified: 2026-03-15
> Purpose: minimal clean brief for continuing work on the current Block 3 line without pulling historical noise back into execution.

## Read First

1. Root `AGENTS.md`
2. `.local_mandatory_preexec.md`
3. `docs/CURRENT_SOURCE_OF_TRUTH.md`
4. `docs/PHASE9_V739_FACT_ALIGNMENT.md`
5. `docs/BLOCK3_MODEL_STATUS.md`
6. `docs/BLOCK3_RESULTS.md`
7. `docs/benchmarks/phase9_current_snapshot.md`
8. `docs/V739_CURRENT_RUN_MONITOR.md`
9. `docs/PHASE12_TEXT_RERUN_EXECUTION.md`
10. `docs/PHASE9_V739_LESSONS_LEARNED.md`

## Current Truth in One Page

1. Canonical benchmark root: `runs/benchmarks/block3_phase9_fair/`
2. Canonical AutoFit baseline: `AutoFitV739`
3. Freeze pointer: `docs/audits/FULL_SCALE_POINTER.yaml`
4. Raw Phase 9 materialization:
   - `91` metrics files
   - `10213` raw records (Phase 12 text reruns actively landing)
   - `91` raw models
   - `80` raw complete models (≥104)
   - `11` raw partial models
5. V739 state: 112/112 COMPLETE, **#13/80** by mean rank (top 16%)
6. Text embedding artifacts exist and are fully functional (5.77M rows, 64 PCA dims, float32).
7. Phase 12 text reruns actively landing — 44/91 models now have core_text, 43/91 have full results.
8. V739 landed: `112/112` ALL COMPLETE.
9. Gap-fill: 12 TSLib jobs RUNNING. Chronos2/TTM now COMPLETE (≥104).

## What Is Historical Only

1. `docs/_legacy_repo/`
2. `docs/benchmarks/LEGACY__block3_truth_pack__v72_v73/`
3. V734-V738 empirical results or design narratives
4. Phase 7 / Phase 8 benchmark outputs
5. The archived large result table in `docs/_legacy_repo/BLOCK3_RESULTS_table_20260314.md`

## Immediate Work That Still Matters

1. ~~land the first valid V739 results into `runs/benchmarks/block3_phase9_fair/`~~ ✅ DONE (112/112)
2. ~~execute the real Phase 12 text-enabled reruns~~ ✅ 40 jobs submitted, actively landing
3. close the remaining partial-model gaps (11 partial, NegBinGLM excluded)
4. refresh snapshot + aggregation after Phase 12 completes

## Immediate Work That Does Not Matter Yet

1. V740+
2. any new AutoFit generation derived from V72 / early V73 docs
3. paper claims based on non-canonical outputs

## Operational Commands

### Refresh the current fact snapshot

```bash
python3 scripts/build_phase9_current_snapshot.py
```

### Refresh the filtered leaderboard

```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/aggregate_block3_results.py
```

### Inspect live queue

```bash
squeue -u npin,cfisch
```

### Check canonical autofit landings

```bash
find runs/benchmarks/block3_phase9_fair -path '*/autofit/*/metrics.json' | sort
```

## Non-Negotiable Rules

1. Never write current-status claims from historical docs.
2. Never treat `block3_phase10/v739` outputs as canonical fair-line evidence.
3. Never cite V739 as empirically landed until canonical fair-line metrics exist.
4. Never mix seed-replication placeholder results with real text-enabled reruns.
