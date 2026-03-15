# Block 3 Delivery Plan (Current)

## Plan 1 - Finish the Current Fair Benchmark Line

1. Keep `runs/benchmarks/block3_phase9_fair/` as the only canonical benchmark directory.
2. Do not mix Phase 7/8 or V72/V73 outputs back into the current line.
3. Track progress through `docs/benchmarks/phase9_current_snapshot.md` and live `squeue` only.
4. Use `docs/PHASE9_V739_FACT_ALIGNMENT.md` to reject common legacy misstatements before they spread into current docs.

## Plan 2 - Land the First Clean V739 Results

1. V739 is the only active and valid AutoFit baseline.
2. Current state is `0/104` landed conditions.
3. No V739 performance claim is allowed until metrics actually land.
4. The observed live V739 queue still points at legacy `runs/benchmarks/block3_phase10/v739/`, so it does not count toward the canonical fair benchmark line.
5. Use `docs/V739_CURRENT_RUN_MONITOR.md` as the current operational guide.

## Plan 3 - Close the Remaining Coverage Gaps

1. Finish the 13 partial raw models.
2. Keep raw-materialization counts and filtered-leaderboard counts separate.
3. Update docs only after re-scanning `metrics.json` and `all_results.csv`.

## Plan 4 - Execute the Real Text-Enabled Reruns

1. Text embedding artifacts now exist in `runs/text_embeddings/`.
2. The project is no longer blocked on generating embeddings.
3. The next benchmark correction is to run the real `core_text` / `full` reruns and replace the temporary seed-replication interpretation.
4. Use `docs/PHASE12_TEXT_RERUN_EXECUTION.md` as the current operational guide.

## Plan 5 - Defer V740+ Until the Current Line Is Clean

1. Do not start a new AutoFit generation from archived V72/V73 design docs.
2. Do not start V740+ before:
   - V739 has landed
   - partial models are closed or explicitly excluded
   - real text/full reruns have landed
3. When a new version starts, it must build from V739 and the lessons in `docs/PHASE9_V739_LESSONS_LEARNED.md`.
