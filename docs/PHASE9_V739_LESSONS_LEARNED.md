# Phase 9 / V739 Lessons Learned

This document preserves the useful knowledge from earlier failed or superseded phases without carrying their outdated conclusions forward.

## 1. Never Use Test-Set Metrics for Model Selection

1. `metrics.json` is test-set output.
2. Oracle tables built from `metrics.json` are test-set leakage.
3. V734-V738 failed exactly because they turned test metrics into routing logic.
4. The only valid model-selection signal for current AutoFit is training/validation evidence, especially harness-provided `val_raw`.

## 2. Keep Experiment Semantics Separate from Historical Narrative

1. Phase 7/8 documents were useful for debugging, but their benchmark results are not reusable.
2. Historical root-cause writeups should be archived once a new clean benchmark line exists.
3. Current-facing docs must report only what is true now, not what used to be true during earlier exploration.

## 3. Distinguish Raw Materialization from Filtered Leaderboard Truth

1. Raw materialization answers: “what has physically landed?”
2. Filtered leaderboard answers: “what remains fair and comparable after exclusions?”
3. Both views are necessary, but they must never be mixed in one headline number without labeling.
4. Current examples:
   - raw: 90 models, 77 complete
   - filtered: 69 models, 59 complete

## 4. Validate Artifact Availability Before Writing Ablation Claims

1. The project previously treated `core_text` and `full` as if text embeddings existed.
2. In reality, the benchmark had to be reinterpreted as a 2-seed replication setup.
3. Artifact existence must be checked directly in `runs/` before any doc claims an ablation is active.
4. As of 2026-03-13, text embedding artifacts now exist; documentation must reflect that changed fact.

## 5. Freeze Means Freeze

1. The WIDE2 freeze is sealed and read-only.
2. Model and execution iteration must happen on top of the frozen pointer path, not by backfilling or mutating freeze assets.
3. Mid-benchmark hyperparameter drift invalidates fairness; NF production configs remain frozen unless explicitly re-benchmarked.

## 6. Queue State Must Be Verified Live, Not Copied Forward

1. Queue summaries age very quickly.
2. V739 status changed from running-on-l40s documentation to all-pending reality.
3. Every status doc should be written from a direct `squeue` check, not from yesterday’s narrative.

## 7. The Safe AutoFit Baseline Is Simple and Auditable

1. V739 is not a “magic” meta-system.
2. It is valuable precisely because it is easier to audit:
   - explicit candidate pool
   - explicit temporal validation split
   - no oracle tables
   - direct `val_raw` use from the harness
3. Future versions should preserve this auditability instead of reintroducing opaque shortcuts.

## 8. Current Practical Next Step

1. Do not start V740+ from archived V72/V73 materials.
2. First finish the current fair benchmark line:
   - land V739 results
   - close partial models
   - run real text-enabled reruns
3. After that, use the retained reference knowledge in `docs/references/`, the corrected-claim guardrails in `docs/PHASE9_V739_FACT_ALIGNMENT.md`, and the failure guardrails in `.local_mandatory_preexec.md` to design the next version cleanly.
