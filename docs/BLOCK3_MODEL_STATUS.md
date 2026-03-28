# Block 3 Model Benchmark Status

> Last updated: 2026-03-28 17:10 CET
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
| live jobs | **40** | 9R + 31PD (gpu 6R+0PD, l40s 3R+14PD, hopper 0R+17PD) |
| clean full comparable frontier | **55** | post-filter non-retired models at shared 160/160 |
| text embeddings | available | 5,774,931 rows, 64 PCA dims |

## V739 Status

| Fact | Value | Evidence |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | Root `AGENTS.md` |
| landed conditions | `132/160` | co=28, ce=28, ct=28, fu=28, s2/e2 gap-filling |
| missing total | `28` | direct `all_results.csv` diff against expected 160 cells |
| missing breakdown | `t1_e2=9`, `t1_s2=8`, `t2_e2=4`, `t2_s2=4`, `t3_e2=3` | direct query |
| live V739 jobs | `5` | 5 RUNNING + 0 PENDING |
| quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| mean rank (last computed stable slice) | **#13** | current valid AutoFit baseline only |
| V734-V738 | retired | oracle test-set leakage |

### Current V739 Queue Surface

| Job | State | Purpose |
| --- | --- | --- |
| `af739_t1_e2` | RUNNING | task1 `core_edgar_seed2` gap-fill, running as `5290110` |
| `af739_t1_s2` | RUNNING | task1 `core_only_seed2` gap-fill, running as `5290111` at `189G` |
| `af739_t2_s2` | RUNNING | task2 `core_only_seed2` gap-fill, running as `5290113` at `189G` |
| `af739_t2_e2` | RUNNING | task2 `core_edgar_seed2` gap-fill, running as `5290112` |
| `af739_t3_e2` | RUNNING | task3 `core_edgar_seed2` gap-fill, timeout-repaired as `5290366` |

## Incomplete Active Models

| Tier | Models | Current state |
| --- | --- | --- |
| structural OOM | `XGBoost@159`, `XGBoostPoisson@157` | known unfixable gaps |
| AutoFit gap-fill | `AutoFitV739@132` | fully covered by 5 live af739 jobs (`5R + 0PD`) |
| old TSLib gap-fill | `ETSformer`, `LightTS`, `Pyraformer`, `Reformer`, `Crossformer`, `MSGNet`, `MambaSimple`, `PAttn` | covered by accel_v2 queue |
| Phase 15 valid models | `CARD`, `DUET`, `FiLM`, `FilterTS`, `FreTS`, `Fredformer`, `ModernTCN`, `NonstationaryTransformer`, `PDF`, `PIR`, `SCINet`, `SRSNet`, `SegRNN`, `TimeRecipe`, `xPatch` | all at `78/160`, covered by accel_v2 queue |

**Operational conclusion:** aside from the two structural OOM tabular models, there is currently **no known mandatory non-structural gap that is completely unqueued**.

## Live Queue Reality

Detailed per-job progress/ETA snapshot: `docs/RUN_QUEUE_PROGRESS_CURRENT.md`

| Slice | Value | Notes |
| --- | ---: | --- |
| gpu RUNNING | 6 | `af739_t1_e2`, `af739_t1_s2`, `af739_t2_e2`, `af739_t2_s2`, `af739_t3_e2`, `gpu_cos2_t2` |
| gpu PENDING | 0 | no local-only jobs remain queued on gpu |
| l40s RUNNING | 3 | `l2_ac_t2_e2`, `l2_ac_t3_co`, `l2_ac_t3_ct` |
| l40s PENDING | 14 | overflow / resume-safe accel_v2 backlog |
| hopper RUNNING | 0 | no current hopper jobs are running for `npin` |
| hopper PENDING | 17 | priority-limited overflow backlog |
| bigmem RUNNING | 0 | no current bigmem benchmark-side work |
| **total** | **40** | **9 RUNNING + 31 PENDING** |

### Current Throughput Interpretation

- `gpu` is again the live critical path: all five V739 gap-fill jobs plus `gpu_cos2_t2` are currently running.
- local-only model-clear surface has advanced:
  - `5294242` `v740_samf_clr` completed successfully
  - `5294254` `v740_prop_std` completed successfully on `bigmem`
  - `5294255` `v740_tpfn26c` completed successfully on `gpu`
- `l40s` remains the best overflow partition when memory needs fit within the 15G/CPU rule and a resume-safe script exists.
- `hopper` is not memory-constrained in the way older docs implied; the real limiter is priority/preemption. It should be treated as opportunistic overflow, not as the sole critical path.
- `ModernTCN` remains the dominant throughput bottleneck for non-`e2` accel jobs.

## Queue Repairs Executed on 2026-03-27 / 2026-03-28

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
5. **Resubmitted the timed-out `l40s` overflow job**:
   - `5294241` `l2_ac_t2_s2`
   - the direct log shows it progressed into `investors_count` work and died at the 2-day wall, so this is a throughput continuation, not a structural failure
6. **Advanced the local-only model-clear lane**:
   - `5294242` `v740_samf_clr` (`SAMformer`) → completed successfully
   - `5294243` `v740_prop_clr` (`Prophet`) → failed due invalid `quick+h30` preset/horizon combination
   - `5294254` `v740_prop_std` (`Prophet`) → corrected CPU-only resubmission completed successfully
   - `5294255` `v740_tpfn26c` (`TabPFNClassifier`) → first latest-source 2.6 narrow clear completed successfully
7. **Rebuilt** `docs/benchmarks/phase9_current_snapshot.{json,md}` and `docs/BLOCK3_RESULTS.md` using the insider Python environment after the new landings.
8. **Recovered the first corrected local V739 vs V740 compare** (`mb_t1_core_edgar_is_funded_h14`), which currently favors V739/`PatchTST` over V740-alpha on that audited binary EDGAR slice.

## Resource Policy (Current)

| Partition | When to use | Current guidance |
| --- | --- | --- |
| `gpu` | mainline critical path | primary partition for af739 and accel_v2 |
| `l40s` | overflow with correct CPU/memory sizing | preferred secondary partition when jobs are resume-safe |
| `hopper` | opportunistic overflow only | use only with checkpoint/requeue-safe scripts; do not rely on it as sole critical path |

## Immediate Next Actions

1. Let the six running gpu jobs (`af739_t1_e2`, `af739_t1_s2`, `af739_t2_e2`, `af739_t2_s2`, `af739_t3_e2`, `gpu_cos2_t2`) keep draining in place.
2. Continue letting accel_v2 drain on `gpu + l40s + hopper`; do not touch the canonical benchmark harness beyond necessary resubmissions.
3. Rebuild `docs/benchmarks/phase9_current_snapshot.*` and `docs/BLOCK3_RESULTS.md` again after the next meaningful landing increment.
4. Use the newly completed local-only model-clear results (`SAMformer`, `Prophet`, `TabPFN 2.6`) to decide which comparator deserves the next canonical expansion step.
