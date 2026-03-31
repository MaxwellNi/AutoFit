# Block 3 Model Benchmark Status

> Last updated: 2026-03-31 03:03 CEST
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: direct scan of `runs/benchmarks/block3_phase9_fair/`, `all_results.csv`, live `squeue`, `sacct`, and benchmark aggregation scripts.

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw records | **16152** | direct scan 2026-03-30 |
| raw models (all) | 137 | 116 non-retired + 21 retired AutoFit legacy lines |
| raw complete @160 | 75 | direct unique-condition scan |
| active leaderboard models | **92** | 116 non-retired raw models - 24 audit-excluded |
| active complete @160 | **62** | 75 raw complete - 13 excluded complete models |
| incomplete active models | **30** | 92 - 62 |
| post-filter records in `all_results.csv` | **12300** | fairness_only=True, min_coverage=0.98 |
| post-filter distinct models | 107 | includes 21 retired AutoFit legacy lines |
| post-filter non-retired models | 86 | `all_results.csv` minus retired AutoFit legacy lines |
| conditions per full model | 160 | t1(72) + t2(48) + t3(40) |
| live jobs | **42** | 13R + 29PD (gpu 6R+2PD, l40s 4R+13PD, hopper 3R+14PD) |
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
| `af739_t1_e2` | RUNNING | task1 `core_edgar_seed2` gap-fill, timeout-repaired again as `5298285` |
| `af739_t1_s2` | RUNNING | task1 `core_only_seed2` gap-fill, now running as `5299888` at `224G / 9 CPU` |
| `af739_t2_s2` | RUNNING | task2 `core_only_seed2` gap-fill; `5298048` also OOMed at `200G / 8 CPU`, now repaired and running as `5300059` at `224G / 9 CPU` |
| `af739_t2_e2` | RUNNING | task2 `core_edgar_seed2` gap-fill, timeout-repaired again as `5298286` |
| `af739_t3_e2` | RUNNING | task3 `core_edgar_seed2` gap-fill, timeout-repaired again as `5298287` |

## Incomplete Active Models

| Tier | Models | Current state |
| --- | --- | --- |
| structural OOM | `XGBoost@159`, `XGBoostPoisson@157` | known unfixable gaps |
| AutoFit gap-fill | `AutoFitV739@132` | fully covered by 5 live af739 jobs (`4R + 1PD`) |
| old TSLib gap-fill | `ETSformer`, `LightTS`, `Pyraformer`, `Reformer`, `Crossformer`, `MSGNet`, `MambaSimple`, `PAttn` | covered by accel_v2 queue |
| Phase 15 valid models | `CARD`, `DUET`, `FiLM`, `FilterTS`, `FreTS`, `Fredformer`, `ModernTCN`, `NonstationaryTransformer`, `PDF`, `PIR`, `SCINet`, `SRSNet`, `SegRNN`, `TimeRecipe`, `xPatch` | all at `78/160`, covered by accel_v2 queue |

**Operational conclusion:** aside from the two structural OOM tabular models, there is currently **no known mandatory non-structural gap that is completely unqueued**.

## Live Queue Reality

Detailed per-job progress/ETA snapshot: `docs/RUN_QUEUE_PROGRESS_CURRENT.md`

| Slice | Value | Notes |
| --- | ---: | --- |
| gpu RUNNING | 6 | `af739_t2_s2`, `af739_t1_s2`, `af739_t1_e2`, `af739_t2_e2`, `af739_t3_e2`, `gpu_cos2_t2` |
| gpu PENDING | 2 | `v740_samf_bin4h`, `v740_lgts_f4h` |
| l40s RUNNING | 4 | `l2_ac_t1_e2`, `l2_ac_t3_ce`, `l2_ac_t1_ct`, `l2_ac_t3_fu` |
| l40s PENDING | 13 | overflow / resume-safe accel_v2 backlog |
| hopper RUNNING | 3 | `h2_ac_t1_e2`, `h2_ac_t1_fu`, `h2_ac_t1_s2` |
| hopper PENDING | 14 | priority-limited overflow backlog |
| bigmem RUNNING | 0 | no current bigmem benchmark-side work |
| **total** | **40** | **13 RUNNING + 27 PENDING** |

### Current Throughput Interpretation

- `gpu` is again the live critical path: both heavy V739 seed2 jobs (`af739_t1_s2`, `af739_t2_s2`), three V739 `e2` jobs, and `gpu_cos2_t2` are currently running together.
- local-only model-clear surface has advanced:
  - `5294242` `v740_samf_clr` completed successfully
  - `5294254` `v740_prop_std` completed successfully on `bigmem`
  - `5294255` `v740_tpfn26c` completed successfully on `gpu`
  - `5294259` `v740_tpfn26r_fu` has now completed and proves the funding regressor path runs cleanly through the narrow-clear harness
  - `5294260` `v740_tpfn26r_inv` has now completed too, but the resulting audited slice fails the fairness gate and therefore is not promotable
- `l40s` remains the best overflow partition when memory needs fit within the 15G/CPU rule and a resume-safe script exists.
- the repaired `gpu` e2/cos2 copies are now live again, while `hopper` has also reactivated three `t1` overflow jobs, so the queue has swung back toward active multi-partition draining rather than pure backlog.
- `l2_ac_t3_co` is now back in queue:
  - `5269761 l2_ac_t3_co` timed out at the 2-day wall on `l40s`
  - log evidence shows it was cleanly continuing the resumable `investors_count` section
  - it has now been resubmitted as `5298506`
- `LightGTS` has now been promoted from wrapper-only status to a completed local-clear entrant:
  - tiny real-data smoke is non-fallback on `task2_forecast/core_edgar/funding_raised_usd/h=30`
  - first narrow benchmark-clear attempt `5298059` completed without model execution because the compute node could not see `/tmp/LightGTS`
  - repaired rerun `5298289` completed successfully after switching to `~/.cache/block3_optional_repos/LightGTS`
  - current narrow-clear result: `MAE = 201930.5506`, `fairness_pass = true`
- `OLinear` has now moved from docs-only into the same local-clear lane:
  - official repo is audited and pinned locally
  - Block 3 wrapper exists in `src/narrative/block3/models/olinear_model.py`
  - first tiny funding smoke is non-fallback on `task2_forecast/core_edgar/funding_raised_usd/h=30`
  - tiny investors-count smoke on `h=14` falls back to constant on that small slice
  - first narrow benchmark-clear rerun `5298296` completed successfully
  - current narrow-clear result: `MAE = 131288.8062`, `fairness_pass = true`
- `ElasTST` has now joined the same local-clear lane:
  - official ProbTS repo is audited locally
  - Block 3 wrapper exists in `src/narrative/block3/models/elastst_model.py`
  - first tiny funding smoke is non-fallback on `task2_forecast/core_edgar/funding_raised_usd/h=30`
  - default-context tiny investors slice had no windows, but a shorter-context rerun is also non-fallback
  - first narrow benchmark-clear `5298399` completed successfully
  - current narrow-clear result: `MAE = 201925.6990`, `fairness_pass = true`
  - second narrow benchmark-clear `5298437` now proves the shorter-context count slice executes through the real harness, but it returns `fairness_pass = false`
  - a more conservative count-side repair probe has now also completed:
    - `5299641 v740_elas_i21_clr`
    - `task2_forecast / core_edgar / investors_count / h=14`
    - `input_size = 21`, `l_patch_size = 3_6`
    - `MAE = 124.4985`
    - `fairness_pass = false`
  - current honest state is therefore:
    - funding side = clean local-clear
    - investors/count side = still not promotable
- `UniTS` has now joined the same local-clear lane:
  - official repo audited locally
  - Block 3 forecasting-only wrapper exists in `src/narrative/block3/models/units_model.py`
  - first tiny funding smoke is non-fallback
  - first narrow benchmark-clear `5298457` completed successfully
  - current narrow-clear result: `MAE = 131725.2212`, `fairness_pass = true`
  - count-side narrow clear `5298559` has also completed, but with `fairness_pass = false`
- `V740-alpha` now includes a first audited `CASA + TimeEmb`-inspired mechanism pass:
  - lightweight local-context block + static/dynamic/source fusion gate live in `src/narrative/block3/models/v740_alpha.py`
  - audited local smokes now show:
    - funding `h=30`: `MAE = 182493.8486`
    - binary `h=14`: `MAE = 0.2853`
    - funding `h=60`: `MAE = 176137.8275`
  - these remain local-only engineering results, not canonical benchmark lines
- `hopper` is not memory-constrained in the way older docs implied; the real limiter is priority/preemption. It should be treated as opportunistic overflow, not as the sole critical path.
- `ModernTCN` remains the dominant throughput bottleneck for non-`e2` accel jobs.

## Queue Repairs Executed on 2026-03-27 / 2026-03-28

1. **Read `sacct` before trusting old queue docs.** The earlier “5 af739 RUNNING” statement had already gone stale: `af739_t1_e2` and `af739_t2_e2` had TIMEOUTed, while `af739_t1_s2` and `af739_t2_s2` had OOMed.
2. **Resubmitted the four initially missing V739 jobs**:
   - `5290110` `af739_t1_e2`
   - `5290111` `af739_t1_s2`
   - `5290112` `af739_t2_e2`
   - `5290113` `af739_t2_s2`
3. **The intermediate `189G` repair still failed**:
   - `5290111 af739_t1_s2` OOM with `MaxRSS=198181112K`
   - `5290113 af739_t2_s2` OOM with `MaxRSS=198179092K`
4. **Raised both seed2 jobs again to the current practical setting**:
   - `5298049 af739_t1_s2` at `200G / 8 CPU`
   - `5298048 af739_t2_s2` at `200G / 8 CPU`
5. **Resubmitted both newly timed-out gpu critical-path jobs**:
   - `5290366` `af739_t3_e2`
   - `5290365` `gpu_cos2_t2`
6. **Resubmitted the timed-out `l40s` overflow job**:
   - `5294241` `l2_ac_t2_s2`
   - the direct log shows it progressed into `investors_count` work and died at the 2-day wall, so this is a throughput continuation, not a structural failure
7. **Advanced the local-only model-clear lane**:
   - `5294242` `v740_samf_clr` (`SAMformer`) → completed successfully
   - `5294243` `v740_prop_clr` (`Prophet`) → failed due invalid `quick+h30` preset/horizon combination
   - `5294254` `v740_prop_std` (`Prophet`) → corrected CPU-only resubmission completed successfully
   - `5294255` `v740_tpfn26c` (`TabPFNClassifier`) → first latest-source 2.6 narrow clear completed successfully
   - `5294259` `v740_tpfn26r_fu` (`TabPFNRegressor`) → completed funding narrow clear (`fairness_pass=true`, quality weak)
   - `5294260` `v740_tpfn26r_inv` (`TabPFNRegressor`) → completed investors-count narrow clear (`fairness_pass=false`, red-flag result)
   - `5298059` `v740_lgts_clr` (`LightGTS`) → completed without model execution because the compute node could not see `/tmp/LightGTS`
   - `5298289` `v740_lgts_clr` (`LightGTS`) → repaired rerun completed successfully after switching to `~/.cache/block3_optional_repos/LightGTS`
   - `5298296` `v740_olnr_clr` (`OLinear`) → first audited funding narrow clear completed successfully (`fairness_pass=true`)
   - `5299636` `v740_samf_fu_clr` (`SAMformer`) → funding promotion probe completed successfully (`MAE = 130514.7325`, `fairness_pass = true`)
   - `5299637` `v740_lgts_fu_clr` (`LightGTS`) → fuller-source funding promotion probe completed successfully (`MAE = 201930.5506`, `fairness_pass = true`)
   - `5299641` `v740_elas_i21_clr` (`ElasTST`) → conservative count-side repair completed, but still `fairness_pass = false`
8. **Rebuilt** `docs/benchmarks/phase9_current_snapshot.{json,md}` and `docs/BLOCK3_RESULTS.md` using the insider Python environment after the new landings.
9. **Recovered the first corrected local V739 vs V740 compare** (`mb_t1_core_edgar_is_funded_h14`), which currently favors V739/`PatchTST` over V740-alpha on that audited binary EDGAR slice.
10. **A second timeout wave hit the GPU e2/cos2 critical path on 2026-03-30 and was repaired immediately**:
   - `5290110 af739_t1_e2` → `TIMEOUT`
   - `5290112 af739_t2_e2` → `TIMEOUT`
   - `5290366 af739_t3_e2` → `TIMEOUT`
   - `5290365 gpu_cos2_t2` → `TIMEOUT`
   - repaired copies now live as:
     - `5298285 af739_t1_e2`
     - `5298286 af739_t2_e2`
     - `5298287 af739_t3_e2`
     - `5298288 gpu_cos2_t2`
11. **`af739_t1_s2` also required one more memory repair after the `200G / 8 CPU` copy still OOMed**:
   - `5298049 af739_t1_s2` → `OUT_OF_MEMORY` after `1-04:38:15`
   - repaired again as:
     - `5299888 af739_t1_s2`
     - `gpu / 224G / 9 CPU / 2d`
12. **`af739_t2_s2` has now shown the same fourth-round memory pattern as `t1_s2`.**
   - `5298048 af739_t2_s2` → `OUT_OF_MEMORY` after `1-07:46:25`
   - `MaxRSS = 209713460K`
   - repaired again as:
     - `5300059 af739_t2_s2`
     - `gpu / 224G / 9 CPU / 2d`
   - this repaired copy is now **RUNNING**
13. **`SAMformer` and `LightGTS` are no longer waiting at the single-probe stage.**
   - `5300057 v740_samf_lane` has now completed cleanly and gives a promotion-positive funding-side result for `SAMformer`
   - `5300058 v740_lgts_lane` has now completed cleanly and gives a promotion-positive fuller-source funding-side result for `LightGTS`
14. **The dedicated V740 text-gain audit lane has now completed cleanly.**
   - `5300635 v740_txtgain_bin`
   - scope:
     - `task1_outcome / {core_edgar, full} / is_funded / h={14,60}`
   - result:
     - `full` beats `core_edgar` on both audited binary horizons
     - `h=14`: `0.3264 < 0.3281`
     - `h=60`: `0.3056 < 0.3741`
   - current honest interpretation:
     - on these audited fuller-source binary slices, text is now a **real positive signal**
     - the remaining open text-gain question is narrower and now mainly concerns
       funding/count behavior rather than binary viability
15. **Two more serious promotion-lane probes are now queued.**
   - `5300644 v740_samf_bin4h`
     - `task1_outcome / core_edgar / is_funded / h={1,7,14,30}`
   - `5300645 v740_lgts_f4h`
     - `task2_forecast / full / funding_raised_usd / h={1,7,14,30}`
   - purpose:
     - move both entrants beyond single-condition promotion-positive evidence
       into a first multi-horizon lane expansion
16. **`OLinear` is no longer blocked by “missing artifact generation” as an abstract issue.**
    The current state is:
    - vendor repo audited locally,
    - Block 3 wrapper implemented,
    - tiny real-data smoke exists,
    - first funding narrow clear exists,
    - count-side evidence is still weak and should not yet be oversold.
17. **`l2_ac_t3_co` was the only newly verified canonical task that should run but was absent from the queue.**
    It has now been resubmitted as `5298506` on the same justified `l40s / 120G / 8 CPU / 1 GPU / 2d` resume-safe envelope.

## Resource Policy (Current)

| Partition | When to use | Current guidance |
| --- | --- | --- |
| `gpu` | mainline critical path | primary partition for af739 and accel_v2 |
| `l40s` | overflow with correct CPU/memory sizing | preferred secondary partition when jobs are resume-safe |
| `hopper` | opportunistic overflow only | use only with checkpoint/requeue-safe scripts; do not rely on it as sole critical path |

## Immediate Next Actions

1. Let the six running gpu jobs (`af739_t2_s2`, `af739_t1_s2`, `af739_t1_e2`, `af739_t2_e2`, `af739_t3_e2`, `gpu_cos2_t2`) keep draining in place.
2. Continue letting accel_v2 drain on `gpu + l40s + hopper`; do not touch the canonical benchmark harness beyond necessary resubmissions.
3. Rebuild `docs/benchmarks/phase9_current_snapshot.*` and `docs/BLOCK3_RESULTS.md` again after the next meaningful landing increment.
4. Continue using the completed local-only model-clear results (`SAMformer`, `Prophet`, `TabPFN 2.6`, `LightGTS`, `OLinear`, `ElasTST`, `UniTS`) plus the now-completed wider lane jobs (`5300057`, `5300058`) to decide which comparator deserves the next canonical expansion step.
5. Read out `5300644 v740_samf_bin4h` and `5300645 v740_lgts_f4h` next to decide whether either entrant is ready to move beyond the current local-clear lane.
