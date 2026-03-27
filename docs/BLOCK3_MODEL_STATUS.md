# Block 3 Model Benchmark Status

> Last updated: 2026-03-27 17:10 CET
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: direct scan of `runs/benchmarks/block3_phase9_fair/`, `all_results.csv`, live `squeue`, `sacct`, and benchmark aggregation scripts.

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw records | **16122** | direct scan 2026-03-27 |
| raw models (all) | 137 | 116 non-retired + 21 retired AutoFit legacy lines |
| raw complete @160 | 75 | direct unique-condition scan |
| active leaderboard models | **92** | 116 non-retired raw models - 24 audit-excluded |
| active complete @160 | **62** | 75 raw complete - 13 excluded complete models |
| incomplete active models | **30** | 92 - 62 |
| post-filter records in `all_results.csv` | **12272** | fairness_only=True, min_coverage=0.98 |
| post-filter distinct models | 107 | includes 21 retired AutoFit legacy lines |
| post-filter non-retired models | 86 | `all_results.csv` minus retired AutoFit legacy lines |
| conditions per full model | 160 | t1(72) + t2(48) + t3(40) |
| live jobs | **40** | 5R + 35PD (gpu 0R+6PD, l40s 3R+14PD, hopper 2R+15PD) |
| clean full comparable frontier | **55** | post-filter non-retired models at shared 160/160 |
| text embeddings | available | 5,774,931 rows, 64 PCA dims |

## V739 Status

| Fact | Value | Evidence |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | Root `AGENTS.md` |
| landed conditions | `132/160` | co=28, ce=28, ct=28, fu=28, s2/e2 gap-filling |
| missing total | `28` | direct `all_results.csv` diff against expected 160 cells |
| missing breakdown | `t1_e2=9`, `t1_s2=8`, `t2_e2=4`, `t2_s2=4`, `t3_e2=3` | direct query |
| live V739 jobs | `5` | 0 RUNNING + 5 PENDING |
| quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| mean rank (last computed stable slice) | **#13** | current valid AutoFit baseline only |
| V734-V738 | retired | oracle test-set leakage |

### Current V739 Queue Surface

| Job | State | Purpose |
| --- | --- | --- |
| `af739_t1_e2` | PENDING | task1 `core_edgar_seed2` gap-fill, resubmitted as `5290110` |
| `af739_t1_s2` | PENDING | task1 `core_only_seed2` gap-fill, OOM-fixed to `189G` as `5290111` |
| `af739_t2_s2` | PENDING | task2 `core_only_seed2` gap-fill, OOM-fixed to `189G` as `5290113` |
| `af739_t2_e2` | PENDING | task2 `core_edgar_seed2` gap-fill, resubmitted as `5290112` |
| `af739_t3_e2` | PENDING | task3 `core_edgar_seed2` gap-fill, timeout-repaired as `5290366` |

## Incomplete Active Models

| Tier | Models | Current state |
| --- | --- | --- |
| structural OOM | `XGBoost@159`, `XGBoostPoisson@157` | known unfixable gaps |
| AutoFit gap-fill | `AutoFitV739@132` | fully covered by 5 live af739 jobs (`0R + 5PD`) |
| old TSLib gap-fill | `ETSformer`, `LightTS`, `Pyraformer`, `Reformer`, `Crossformer`, `MSGNet`, `MambaSimple`, `PAttn` | covered by accel_v2 queue |
| Phase 15 valid models | `CARD`, `DUET`, `FiLM`, `FilterTS`, `FreTS`, `Fredformer`, `ModernTCN`, `NonstationaryTransformer`, `PDF`, `PIR`, `SCINet`, `SRSNet`, `SegRNN`, `TimeRecipe`, `xPatch` | all at `78/160`, covered by accel_v2 queue |

**Operational conclusion:** aside from the two structural OOM tabular models, there is currently **no known mandatory non-structural gap that is completely unqueued**.

## Live Queue Reality

Detailed per-job progress/ETA snapshot: `docs/RUN_QUEUE_PROGRESS_CURRENT.md`

| Slice | Value | Notes |
| --- | ---: | --- |
| gpu RUNNING | 0 | no current gpu jobs are running for `npin` |
| gpu PENDING | 6 | `af739_t1_e2`, `af739_t1_s2`, `af739_t2_e2`, `af739_t2_s2`, `af739_t3_e2`, `gpu_cos2_t2` |
| l40s RUNNING | 3 | `l2_ac_t2_s2`, `l2_ac_t2_e2`, `l2_ac_t3_ct` |
| l40s PENDING | 14 | overflow / resume-safe accel_v2 backlog |
| hopper RUNNING | 2 | `h2_ac_t3_ct`, `h2_ac_t1_co` on iris-197 |
| hopper PENDING | 15 | priority-limited overflow backlog |
| **total** | **40** | **5 RUNNING + 35 PENDING** |

### Current Throughput Interpretation

- `gpu` remains the main critical path, but it is currently in an all-pending phase: four already-resubmitted V739 gap-fill jobs plus the newly requeued `af739_t3_e2` and `gpu_cos2_t2`.
- there are currently **no** live local-only V740 jobs in queue; the first corrected local compare and the larger-slice `h=90` audit both completed successfully.
- `l40s` remains the best overflow partition when memory needs fit within the 15G/CPU rule and a resume-safe script exists.
- `hopper` is not memory-constrained in the way older docs implied; the real limiter is priority/preemption. It should be treated as opportunistic overflow, not as the sole critical path.
- `ModernTCN` remains the dominant throughput bottleneck for non-`e2` accel jobs.

## Queue Repairs Executed on 2026-03-27

1. **Read `sacct` before trusting old queue docs.** The earlier “5 af739 RUNNING” statement had already gone stale: `af739_t1_e2` and `af739_t2_e2` had TIMEOUTed, while `af739_t1_s2` and `af739_t2_s2` had OOMed.
2. **Resubmitted the four initially missing V739 jobs**:
   - `5290110` `af739_t1_e2`
   - `5290111` `af739_t1_s2`
   - `5290112` `af739_t2_e2`
   - `5290113` `af739_t2_s2`
3. **Raised both seed2 jobs to `189G`** after observing `MaxRSS ≈ 157.3G` on the failed `150G` attempts.
4. **Resubmitted both newly timed-out gpu critical-path jobs**:
   - `5290366` `af739_t3_e2`
   - `5290365` `gpu_cos2_t2`
5. **Rebuilt** `docs/benchmarks/phase9_current_snapshot.{json,md}` and `docs/BLOCK3_RESULTS.md` using the insider Python environment after the new landings.
6. **Recovered the first corrected local V739 vs V740 compare** (`mb_t1_core_edgar_is_funded_h14`), which currently favors V739/`PatchTST` over V740-alpha on that audited binary EDGAR slice.

## Resource Policy (Current)

| Partition | When to use | Current guidance |
| --- | --- | --- |
| `gpu` | mainline critical path | primary partition for af739 and accel_v2 |
| `l40s` | overflow with correct CPU/memory sizing | preferred secondary partition when jobs are resume-safe |
| `hopper` | opportunistic overflow only | use only with checkpoint/requeue-safe scripts; do not rely on it as sole critical path |

## Immediate Next Actions

1. Let the six pending gpu jobs (`af739_t1_e2`, `af739_t1_s2`, `af739_t2_e2`, `af739_t2_s2`, `af739_t3_e2`, `gpu_cos2_t2`) start draining in priority order.
2. Continue letting accel_v2 drain on `gpu + l40s + hopper`; do not touch the canonical benchmark harness beyond necessary resubmissions.
3. Rebuild `docs/benchmarks/phase9_current_snapshot.*` and `docs/BLOCK3_RESULTS.md` again after the next meaningful landing increment.
4. Keep V740 work on the local-only audited side path until it can beat V739 on more than one corrected representative slice.
