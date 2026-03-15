# Block 3 Operational SOP (Current)

> Last updated: 2026-03-14
> Scope: current Phase 9 / V739 benchmark operations only

## 1. Start-of-Session Checklist

1. Read root `AGENTS.md`.
2. Read `.local_mandatory_preexec.md`.
3. Read `docs/CURRENT_SOURCE_OF_TRUTH.md`.
4. Read `docs/PHASE9_V739_FACT_ALIGNMENT.md`.
5. Rebuild the current fact snapshot:

```bash
python3 scripts/build_phase9_current_snapshot.py
```

6. Confirm the live queue directly:

```bash
squeue -u npin,cfisch
```

7. For current execution work, also read:
   - `docs/V739_CURRENT_RUN_MONITOR.md`
   - `docs/PHASE12_TEXT_RERUN_EXECUTION.md`

## 2. Canonical Directories

- Benchmark: `runs/benchmarks/block3_phase9_fair/`
- Freeze pointer: `docs/audits/FULL_SCALE_POINTER.yaml`
- Text embeddings: `runs/text_embeddings/`

Do not treat any other benchmark root as current.

## 3. Data and Validity Rules

1. Freeze assets are read-only.
2. V734-V738 are retired and invalid.
3. V739 is the only valid current AutoFit baseline.
4. Never use test metrics for model selection.
5. If a claim has not been verified by direct file scan or live queue check, do not write it as fact.
6. When old/legacy claims conflict with current artifacts, use `docs/PHASE9_V739_FACT_ALIGNMENT.md`.

## 4. Current Benchmark Interpretation Rules

1. Current landed Phase 9 results still reflect the seed-replication reinterpretation of `core_text` and `full`.
2. Real text/full reruns have not landed yet.
3. Text embeddings now exist, so this is no longer an embedding-generation blocker.
4. The observed live V739 queue still targets legacy `block3_phase10/v739`, so it does not yet change fair-line empirical status.

## 5. Result Refresh Commands

```bash
python3 scripts/build_phase9_current_snapshot.py
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/aggregate_block3_results.py
```

## 6. Current Priority Order

1. Land V739 results.
2. Finish the 13 partial raw models.
3. Execute the real text-enabled reruns.
4. Consolidate the clean benchmark surface.
5. Only then start V740+.

## 7. Current Execution Surfaces

1. V739 live status and landing rules:
   - `docs/V739_CURRENT_RUN_MONITOR.md`
2. Real `core_text` / `full` rerun execution:
   - `docs/PHASE12_TEXT_RERUN_EXECUTION.md`
