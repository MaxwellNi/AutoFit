# Block 3 Model Benchmark Status

> Last updated: 2026-03-26 15:04 CET
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: direct scan of `runs/benchmarks/block3_phase9_fair/`, `all_results.csv`, live `squeue`, `sacct`, and benchmark aggregation scripts.

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw records | **16077** | direct scan 2026-03-26 |
| raw models (all) | 137 | 116 non-retired + 21 retired AutoFit legacy lines |
| raw complete @160 | 75 | direct unique-condition scan |
| active leaderboard models | **92** | 116 non-retired raw models - 24 audit-excluded |
| active complete @160 | **62** | 75 raw complete - 13 excluded complete models |
| incomplete active models | **30** | 92 - 62 |
| post-filter records in `all_results.csv` | **12230** | fairness_only=True, min_coverage=0.98 |
| post-filter distinct models | 107 | includes 21 retired AutoFit legacy lines |
| post-filter non-retired models | 86 | `all_results.csv` minus retired AutoFit legacy lines |
| conditions per full model | 160 | t1(72) + t2(48) + t3(40) |
| live jobs | **59** | 27R + 32PD (gpu 23R+2PD, l40s 1R+16PD, hopper 3R+14PD) |
| text embeddings | available | 5,774,931 rows, 64 PCA dims |

## V739 Status

| Fact | Value | Evidence |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | Root `AGENTS.md` |
| landed conditions | `132/160` | co=28, ce=28, ct=28, fu=28, s2/e2 gap-filling |
| missing total | `28` | direct `all_results.csv` diff against expected 160 cells |
| missing breakdown | `t1_e2=9`, `t1_s2=8`, `t2_e2=4`, `t2_s2=4`, `t3_e2=3` | direct query |
| live V739 jobs | `5` | 5 RUNNING |
| quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| mean rank (last computed stable slice) | **#13** | current valid AutoFit baseline only |
| V734-V738 | retired | oracle test-set leakage |

### Current V739 Queue Surface

| Job | State | Purpose |
| --- | --- | --- |
| `af739_t1_e2` | RUNNING | task1 `core_edgar_seed2` gap-fill |
| `af739_t1_s2` | RUNNING | task1 `core_only_seed2` gap-fill |
| `af739_t2_s2` | RUNNING | task2 `core_only_seed2` gap-fill |
| `af739_t2_e2` | RUNNING | task2 `core_edgar_seed2` gap-fill |
| `af739_t3_e2` | RUNNING | task3 `core_edgar_seed2` gap-fill |

## Incomplete Active Models

| Tier | Models | Current state |
| --- | --- | --- |
| structural OOM | `XGBoost@159`, `XGBoostPoisson@157` | known unfixable gaps |
| AutoFit gap-fill | `AutoFitV739@132` | fully covered by 5 live af739 jobs |
| old TSLib gap-fill | `ETSformer`, `LightTS`, `Pyraformer`, `Reformer`, `Crossformer`, `MSGNet`, `MambaSimple`, `PAttn` | covered by accel_v2 queue |
| Phase 15 valid models | `CARD`, `DUET`, `FiLM`, `FilterTS`, `FreTS`, `Fredformer`, `ModernTCN`, `NonstationaryTransformer`, `PDF`, `PIR`, `SCINet`, `SRSNet`, `SegRNN`, `TimeRecipe`, `xPatch` | all at `78/160`, covered by accel_v2 queue |

**Operational conclusion:** aside from the two structural OOM tabular models, there is currently **no known mandatory non-structural gap that is completely unqueued**.

## Live Queue Reality

Detailed per-job progress/ETA snapshot: `docs/RUN_QUEUE_PROGRESS_CURRENT.md`

| Slice | Value | Notes |
| --- | ---: | --- |
| gpu RUNNING | 23 | 17 `g2_ac_*` + 5 `af739_*` + 1 `gpu_cos2_t2` |
| gpu PENDING | 2 | local-only `v740_mb_case1_v739` and `v740_lh_fu_f90` research jobs |
| l40s RUNNING | 1 | `l2_ac_t2_s2` |
| l40s PENDING | 16 | overflow / resume-safe accel_v2 backlog |
| hopper RUNNING | 3 | `h2_ac_t1_e2`, `h2_ac_t1_fu`, `h2_ac_t1_s2` on iris-197 |
| hopper PENDING | 14 | priority-limited overflow backlog |
| **total** | **59** | **27 RUNNING + 32 PENDING** |

### Current Throughput Interpretation

- `gpu` remains the main critical path and is fully utilized.
- one additional `gpu` pending slot now belongs to a local-only resumable V740 compare job (`v740_mb_case1_v739`), which stays outside the canonical benchmark harness.
- `l40s` remains the best overflow partition when memory needs fit within the 15G/CPU rule and a resume-safe script exists.
- `hopper` is not memory-constrained in the way older docs implied; the real limiter is priority/preemption. It should be treated as opportunistic overflow, not as the sole critical path.
- `ModernTCN` remains the dominant throughput bottleneck for non-`e2` accel jobs.

## Queue Repairs Executed on 2026-03-25

1. **Patched `scripts/build_phase9_current_snapshot.py`** so live V739 jobs are counted from both `v739_*` and `af739_*` names. The old logic falsely reported `v739_jobs_live=0`.
2. **Cancelled and resubmitted `gpu_cos2_t2`** onto the trimmed 23-model working list. The old live copy still carried excluded/broken models and was wasting GPU time on invalid candidates.
3. **Resubmitted `af739_t3_e2`** after confirming it had dropped out of queue (`5273997` TIMEOUT, `5273998` duplicate cancelled).
4. **Rebuilt** `docs/benchmarks/phase9_current_snapshot.{json,md}` and `docs/BLOCK3_RESULTS.md` using the insider Python environment.

## Resource Policy (Current)

| Partition | When to use | Current guidance |
| --- | --- | --- |
| `gpu` | mainline critical path | primary partition for af739 and accel_v2 |
| `l40s` | overflow with correct CPU/memory sizing | preferred secondary partition when jobs are resume-safe |
| `hopper` | opportunistic overflow only | use only with checkpoint/requeue-safe scripts; do not rely on it as sole critical path |

## Immediate Next Actions

1. Let the repaired `gpu_cos2_t2` and `af739_t3_e2` continue to drain while monitoring the new local-only `v740_mb_case1_v739` compare job separately from canonical benchmark jobs.
2. Continue letting accel_v2 drain on `gpu + l40s + hopper`; do not touch the benchmark harness while the queue is active.
3. Rebuild `docs/benchmarks/phase9_current_snapshot.*` and `docs/BLOCK3_RESULTS.md` again after the next meaningful landing increment.
4. Only after V739 and the remaining incomplete active models stabilize should any benchmark-facing V740 evaluation begin.
