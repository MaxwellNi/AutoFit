# Claude Startup Prompt

Use the following prompt when handing the repo to Claude and you want strict separation between current Phase 9 / AutoFitV739 truth and all legacy material.

```text
Please switch into strict fact-audit mode before making any claims.

You are working on a repo where old V72 / early V73 materials still exist for archival traceability, but they are not current operational truth.

Current canonical line only:
- benchmark root: `runs/benchmarks/block3_phase9_fair/`
- AutoFit baseline: `AutoFitV739`
- freeze pointer: `docs/audits/FULL_SCALE_POINTER.yaml`

Read only these files first, in this exact order:
1. `.local_mandatory_preexec.md`
2. `docs/CLAUDE_MINIMAL_HANDOFF.md`
3. `docs/CURRENT_SOURCE_OF_TRUTH.md`
4. `docs/PHASE9_V739_FACT_ALIGNMENT.md`
5. `docs/BLOCK3_MODEL_STATUS.md`
6. `docs/benchmarks/phase9_current_snapshot.md`
7. `docs/V739_CURRENT_RUN_MONITOR.md`
8. `docs/PHASE12_TEXT_RERUN_EXECUTION.md`
9. `docs/LEGACY_SCRIPT_SURFACES.md`

Hard interpretation rules:
1. Any directory prefixed with `LEGACY__` is archive-only, not current truth.
2. `docs/_legacy_repo/` is archive-only, not current truth.
3. `docs/references/` is background knowledge, not status truth.
4. Tombstone paths such as `docs/benchmarks/block3_truth_pack/README.md` and `docs/benchmarks/block3_v72_local_4090/README.md` are redirects to legacy archives, not live benchmark surfaces.
5. `runs/benchmarks/block3_phase10/v739/` is legacy exploratory evidence only, never current benchmark truth.

Do not use any of the following as current evidence unless the task explicitly asks for historical comparison:
- `docs/benchmarks/LEGACY__block3_truth_pack__v72_v73/`
- `docs/benchmarks/LEGACY__block3_v72_local_4090/`
- `docs/audits/LEGACY__freeze_build_profile_and_old_benchmark_audits/`
- `docs/_legacy_repo/`
- `runs/benchmarks/block3_phase10/v739/`
- V734-V738 narratives, outputs, or routing logic

Every claim you make must be classified as one of:
- `current fact`
- `legacy exploratory evidence`
- `hypothesis`

If a statement cannot be supported by a current artifact, current code path, or current live queue inspection, do not state it as fact.

The key current facts you must accept before doing anything else:
- canonical fair-line landed V739 coverage is currently `0/104`
- observed live V739 jobs still target legacy `runs/benchmarks/block3_phase10/v739/`
- text embeddings exist, but real text-enabled reruns have not yet landed into the canonical fair line
- current priorities are:
  1. land the first clean V739 results
  2. finish remaining partial models
  3. land real Phase 12 text/full reruns
  4. consolidate the clean benchmark surface
  5. only then consider V740+

For your first reply, output exactly four sections:
1. `Current facts I accept`
2. `What I will ignore as current truth`
3. `Immediate next execution priorities`
4. `Verification commands I will rely on`

Do not propose V740. Do not summarize legacy V72/V73 benchmark findings as current reality. Do not collapse `104/104` official benchmark units into a different benchmark definition.
```
