# Claude Minimal Handoff

> Last verified: 2026-03-15
> Purpose: fastest safe continuation brief for Claude or any incoming coding agent.

## Start Here

Treat this file as the compressed entrypoint for the current Block 3 line.
If any older document conflicts with this file, prefer this file and the supporting current-truth docs listed below.

## Supporting Files You May Open Next

Only open these after reading this file:
1. `docs/CURRENT_SOURCE_OF_TRUTH.md`
2. `docs/PHASE9_V739_FACT_ALIGNMENT.md`
3. `docs/V739_CURRENT_RUN_MONITOR.md`
4. `docs/PHASE12_TEXT_RERUN_EXECUTION.md`

## Current Facts You Must Use

1. Canonical benchmark root: `runs/benchmarks/block3_phase9_fair/`
2. Canonical AutoFit baseline: `AutoFitV739`
3. Freeze pointer: `docs/audits/FULL_SCALE_POINTER.yaml`
4. Current benchmark materialization:
   - raw: `91` metrics files, `10213` records, `91` models, `80` complete, `11` partial
5. Current V739 state:
   - landed fair-line conditions: `112/112` (ALL COMPLETE)
   - ablations: core_only=28, core_edgar=28, core_text=28, full=28
   - quality: 0 NaN/Inf, 0 fallback, 100% fairness pass
   - benchmark ranking: **#13/80** by mean rank (top 16%), computed across 56 universal conditions
6. Text embedding artifacts exist in `runs/text_embeddings/`
7. Phase 12 text reruns actively landing — 44/91 models now have core_text, 43/91 have full results
8. Gap-fill in progress: 12 TSLib jobs RUNNING
9. NegativeBinomialGLM: structural failure (20/104), cannot complete

## Claims You Must Not Make

1. Do not treat `runs/benchmarks/block3_phase10/v739/` as canonical evidence.
2. Do not say `104/104` is invalid; it remains the official completion unit.
3. Do not say text embeddings are missing.
4. Do not say embedding collapse root cause is proven.
5. Do not start V740+ work yet.

## Current Work Order

1. ~~Land the first clean V739 results into `runs/benchmarks/block3_phase9_fair/`~~ ✅ DONE (112/112)
2. Finish the 12 remaining partial models (NegBinGLM excluded = structural failure)
3. ~~Run and land the real Phase 12 text-enabled reruns~~ ✅ Phase 12 submitted (40 jobs) — actively landing
4. Refresh snapshot and aggregation after Phase 12 completes
5. Only then consider V740+

## Verification Commands

Refresh current facts:
```bash
python3 scripts/build_phase9_current_snapshot.py
```

Refresh current leaderboard:
```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/aggregate_block3_results.py
```

Check live V739 queue:
```bash
squeue -u npin,cfisch | egrep 'JOBID|v739'
for jid in $(squeue -u npin,cfisch -h -o '%i %j' | awk '$2 ~ /v739/ {print $1}'); do
  scontrol show job "$jid" | egrep 'JobId=|JobName=|Command=|WorkDir=|StdOut=|StdErr='
done
```

Check canonical AutoFit landings:
```bash
find runs/benchmarks/block3_phase9_fair -path '*/autofit/*/metrics.json' | sort
```

## If You Need More Context

1. `AGENTS.md` defines the repository-wide mission and hard constraints.
2. `.local_mandatory_preexec.md` is the clean local current-truth checklist.
3. `docs/PHASE9_V739_LESSONS_LEARNED.md` preserves useful lessons without reviving legacy benchmark claims.
4. `docs/_legacy_repo/` is archive only.
