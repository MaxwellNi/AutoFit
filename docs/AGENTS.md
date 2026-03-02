# Block 3 Project Context (Documentation Entry)

This file records the current handoff baseline for Block 3 benchmarking and model iteration.

## Scope

1. Data freeze: `TRAIN_WIDE_FINAL` with pointer `docs/audits/FULL_SCALE_POINTER.yaml`.
2. Benchmark scope: strict-comparable condition lattice (`104/104` completed).
3. Current iteration scope: V7.3 design and validation against the fixed strict baseline.

## End-to-End Chronology

1. The WIDE2 freeze was sealed and verified. All five freeze gates passed, and Block 3 now reads only through `FreezePointer`.
2. The full strict benchmark was completed and consolidated into a single canonical ledger (`104/104` conditions).
3. The full SOTA table was rebuilt as the canonical reference:
   - `docs/BLOCK3_FULL_SOTA_BENCHMARK.md`
   - `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv`
   - `docs/benchmarks/block3_truth_pack/full_sota_104_summary.json`
4. AutoFit V7.2 was completed across the full strict lattice (`104/104` materialized).
5. V7.2 rank tables, truth-pack summaries, and the V7.3 handoff pack were refreshed against the latest materialized metrics.

## Current Status

1. Full SOTA strict benchmark is complete (`104/104`).
2. AutoFit V7.2 strict coverage is complete (`104/104`, missing keys `0`).
3. The latest execution snapshot shows no active V7.2 work:
   - `running_total = 0`
   - `pending_total = 8`
   - all remaining pending jobs are queue-limited reference jobs, not V7.2 completion blockers
   - evidence: `docs/benchmarks/block3_truth_pack/execution_status_latest.json`
4. V7.2 is fully ranked but not competitive on the fixed benchmark:
   - `rank1 wins = 0`
   - `median rank = 22`
   - `median gap vs best = 170.018388%`
   - `win rate vs V7 = 42.307692%`
   - evidence: `docs/benchmarks/block3_truth_pack/v72_rank_104_summary_latest.json`
5. Target-level V7.2 standing:
   - `funding_raised_usd`: materially close to the frontier (`median rank 16`, `median gap 5.7378915%`)
   - `investors_count`: dominant failure lane (`median rank 26`, `median gap 512.6292905%`)
   - `is_funded`: still weak (`median rank 22`, `median gap 170.018388%`)

## Canonical References

1. Full benchmark view:
   - `docs/BLOCK3_FULL_SOTA_BENCHMARK.md`
   - `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv`
2. Current results summaries:
   - `docs/BLOCK3_RESULTS.md`
   - `docs/BLOCK3_MODEL_STATUS.md`
   - `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`
3. Current status and rank evidence:
   - `docs/benchmarks/block3_truth_pack/execution_status_latest.json`
   - `docs/benchmarks/block3_truth_pack/truth_pack_summary.json`
   - `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json`
   - `docs/benchmarks/block3_truth_pack/v72_rank_104_table_latest.csv`
   - `docs/benchmarks/block3_truth_pack/v72_rank_104_summary_latest.json`
4. V7.3 handoff package:
   - `docs/BLOCK3_V73_RESEARCH_EXECUTION_SPEC_20260225.md`
   - `docs/benchmarks/block3_truth_pack/v73_reuse_manifest.csv`
   - `docs/benchmarks/block3_truth_pack/v73_champion_component_map.csv`
   - `docs/benchmarks/block3_truth_pack/v73_rl_policy_spec.json`

## Constraints

1. Freeze assets under `runs/*_20260203_225620/` are read-only.
2. No benchmark conclusion may use non-strict rows.
3. Evaluation, routing, and policy updates must remain train/validation/OOF only.
4. No rerun is required to establish the current SOTA baseline or the current V7.2 standing.

## Immediate Next Work

1. Keep the strict benchmark fixed and reuse-first.
2. Treat V7.2 as a completed baseline, not an active completion project.
3. Prioritize V7.3 redesign on the count lane first (`investors_count`), then binary calibration/routing.
4. Run V7.3 in the sequence: smoke, targeted pilot, full 104-key closure.
