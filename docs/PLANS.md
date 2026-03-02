# Block 3 Delivery Plan (Current)

## Plan A: Baseline Preservation

1. Keep the strict-comparable benchmark frozen at `104/104`.
2. Regenerate the canonical benchmark table after any truth-pack refresh:
   - `python scripts/build_block3_full_sota_benchmark.py`
3. Keep the summary docs synchronized:
   - `docs/BLOCK3_RESULTS.md`
   - `docs/BLOCK3_MODEL_STATUS.md`
   - `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`

## Plan B: V7.2 Closure and Postmortem

1. Treat V7.2 completion as closed (`104/104`, missing keys `0`).
2. Preserve the latest V7.2 evidence package:
   - `docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json`
   - `docs/benchmarks/block3_truth_pack/v72_rank_104_table_latest.csv`
   - `docs/benchmarks/block3_truth_pack/v72_rank_104_summary_latest.json`
3. Use V7.2 only as a completed baseline and failure analysis input:
   - no rank-1 wins
   - median rank `22`
   - count-lane underperformance is the primary redesign driver

## Plan C: V7.3 Development

1. Reuse the strict benchmark evidence as the fixed comparison baseline.
2. Reuse already materialized strict conditions by default:
   - current reuse manifest requires `0` reruns
3. Development order:
   - count lane redesign (`investors_count`)
   - binary calibration and routing redesign (`is_funded`)
   - heavy-tail robustness verification (`funding_raised_usd`)
4. Progress sequence:
   - smoke
   - targeted pilot
   - full `104/104`

## Plan D: Queue and Execution Hygiene

1. Ignore non-blocking reference queue residue when planning V7.3:
   - the latest snapshot has `0` active V7.2 jobs
   - the remaining `8` pending jobs are queue-limited reference jobs
2. Do not submit duplicate work for already materialized strict keys.
3. Continue to assert the execution contract before any run:
   - `python3 scripts/assert_block3_execution_contract.py --entrypoint <script>`
