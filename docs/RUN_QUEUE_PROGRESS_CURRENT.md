# Current Queue Progress

> Snapshot time: 2026-03-31 03:03 CEST
> Source: live `squeue -u npin,cfisch`, `squeue --start -u npin,cfisch`, and `sacct -u npin -S 2026-03-28T00:00:00`

## Summary

| Metric | Value |
| --- | ---: |
| Total jobs | 42 |
| Running | 13 |
| Pending | 29 |
| gpu running | 6 |
| gpu pending | 2 |
| bigmem running | 0 |
| l40s running | 4 |
| l40s pending | 13 |
| hopper running | 3 |
| hopper pending | 14 |

## Running Jobs

| jobid | job | partition | elapsed | limit | progress | time_left | reason/node |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| 5300059 | af739_t2_s2 | gpu | 2:51:45 | 2-00:00:00 | 5.9% | 1d21h08m | iris-186 |
| 5299888 | af739_t1_s2 | gpu | 4:12:46 | 2-00:00:00 | 8.8% | 1d19h47m | iris-179 |
| 5298285 | af739_t1_e2 | gpu | 1-01:27:16 | 2-00:00:00 | 53.0% | 22h32m | iris-185 |
| 5298286 | af739_t2_e2 | gpu | 1-01:27:16 | 2-00:00:00 | 53.0% | 22h32m | iris-185 |
| 5298287 | af739_t3_e2 | gpu | 1-01:27:16 | 2-00:00:00 | 53.0% | 22h32m | iris-186 |
| 5298288 | gpu_cos2_t2 | gpu | 1-01:27:16 | 2-00:00:00 | 53.0% | 22h32m | iris-169 |
| 5269768 | h2_ac_t1_e2 | hopper | 10:43:00 | 2-00:00:00 | 22.4% | 1d13h16m | iris-197 |
| 5269769 | h2_ac_t1_fu | hopper | 10:43:00 | 2-00:00:00 | 22.4% | 1d13h16m | iris-197 |
| 5269770 | h2_ac_t1_s2 | hopper | 10:40:00 | 2-00:00:00 | 22.2% | 1d13h19m | iris-197 |
| 5279085 | l2_ac_t1_e2 | l40s | 13:55:17 | 2-00:00:00 | 29.1% | 1d10h04m | iris-198 |
| 5269760 | l2_ac_t3_ce | l40s | 1-00:36:45 | 2-00:00:00 | 51.2% | 23h23m | iris-198 |
| 5279084 | l2_ac_t1_ct | l40s | 1-07:47:43 | 2-00:00:00 | 66.2% | 16h12m | iris-199 |
| 5269764 | l2_ac_t3_fu | l40s | 1-09:55:03 | 2-00:00:00 | 70.6% | 14h04m | iris-199 |

## Highest-Value Pending Jobs

| jobid | job | partition | planned_start | reason | notes |
| --- | --- | --- | --- | --- | --- |
| 5300644 | v740_samf_bin4h | gpu | N/A | Priority | first more-serious multi-horizon `SAMformer` lane on `task1_outcome/core_edgar/is_funded/h={1,7,14,30}` |
| 5300645 | v740_lgts_f4h | gpu | N/A | Priority | first more-serious multi-horizon `LightGTS` lane on `task2_forecast/full/funding_raised_usd/h={1,7,14,30}` |
| 5269749 | l2_ac_t1_co | l40s | 2026-03-31 16:07:37 | Priority | next l40s backlog item with a concrete start time |
| 5269752 | l2_ac_t1_fu | l40s | 2026-04-01 12:07:23 | Priority | later l40s restart already scheduled |
| 5294241 | l2_ac_t2_s2 | l40s | 2026-04-03 12:10:00 | Priority | repaired `l40s` timeout continuation on the `investors_count` section |
| 5298506 | l2_ac_t3_co | l40s | N/A | Priority | newly resubmitted after verified 2-day TIMEOUT on `5269761`; resumable continuation, not a structural failure |
| 5269765 | h2_ac_t1_ce | hopper | 2026-04-01 15:19:40 | Priority | opportunistic overflow backlog |

## Pending Breakdown

| partition | pending_jobs | notes |
| --- | ---: | --- |
| gpu | 2 | two new local-only promotion-lane jobs (`5300644`, `5300645`) |
| l40s | 13 | accel_v2 overflow backlog; current live state has four active `l40s` runners including `l2_ac_t1_e2` |
| hopper | 14 | opportunistic overflow backlog; three hopper runners are now active again |

## Canonical Benchmark Recovery Notes

1. The gpu critical path now has all six canonical recovery jobs actively running:
   - RUNNING: `af739_t2_s2`, `af739_t1_s2`, `af739_t1_e2`, `af739_t2_e2`, `af739_t3_e2`, `gpu_cos2_t2`
   - separate local-only pending promotion lanes: `v740_samf_bin4h`, `v740_lgts_f4h`
2. `af739_t1_s2` and `af739_t2_s2` failed once more at `189G` (`MaxRSS≈198.18G`) and were resubmitted as `5298049` / `5298048` at `200G / 8 CPU`.
3. `5298049 af739_t1_s2` then OOMed again after `1-04:38:15` at `200G / 8 CPU`, so it has now been resubmitted as:
   - `5299888 af739_t1_s2`
   - `gpu / 224G / 9 CPU / 2d`
4. `5298048 af739_t2_s2` has now shown the same fourth-round memory failure class:
   - `5298048 af739_t2_s2` → `OUT_OF_MEMORY`
   - `MaxRSS = 209713460K`
   - repaired as:
     - `5300059 af739_t2_s2`
     - `gpu / 224G / 9 CPU / 2d`
5. `l2_ac_t2_s2` is the confirmed latest `l40s` timeout casualty on 2026-03-28; log evidence shows it progressed deep into `investors_count` before the 2-day wall, so it was resubmitted as `5294241`.
6. `l2_ac_t3_co` is now also a confirmed resumable timeout continuation:
   - `5269761 l2_ac_t3_co` hit the 2-day wall on 2026-03-30
   - the log shows it was cleanly continuing `investors_count` work when SLURM cancelled it for time limit
   - it has now been resubmitted as `5298506`
7. The `l40s` lane is now running four jobs. At this snapshot:
   - `l2_ac_t1_e2`
   - `l2_ac_t3_ce`
   - `l2_ac_t1_ct`
   - `l2_ac_t3_fu`
   are all RUNNING, while `hopper` has also resumed three `t1` overflow jobs,
   so the current queue is `13R+29PD`.
8. Local-only V740 model-clear status has advanced:
   - `5294242` `v740_samf_clr` → completed successfully
   - `5294243` `v740_prop_clr` → failed before model execution because `quick` preset cannot run `h=30`
   - `5294254` `v740_prop_std` → corrected `Prophet` resubmission, completed successfully
   - `5294255` `v740_tpfn26c` → first `TabPFN 2.6` narrow clear, completed successfully
   - `5294259` `v740_tpfn26r_fu` → completed as the first `TabPFNRegressor` funding clear (`fairness_pass=true`, quality weak)
   - `5294260` `v740_tpfn26r_inv` → completed as the first `TabPFNRegressor` investors-count clear, but with `fairness_pass=false`
9. These local-only jobs use separate output roots and do not count as canonical benchmark results.
10. `5298059` `v740_lgts_clr` is no longer pending. It completed without model execution because the compute node could not see `/tmp/LightGTS`.
11. The repaired `LightGTS` rerun `5298289` completed successfully after switching the vendor repo to the persistent path `~/.cache/block3_optional_repos/LightGTS`.
12. `5298296` `v740_olnr_clr` has also completed successfully as the first audited `OLinear` funding narrow clear:
   - `task2_forecast / core_edgar / funding_raised_usd / h=30`
   - `MAE = 131288.8062`
   - `fairness_pass = true`
   - this remains a local-only side path and does not count as a canonical benchmark landing
13. `5298399` `v740_elas_clr` has now completed successfully as the first `ElasTST` narrow benchmark-clear probe:
   - `task2_forecast / core_edgar / funding_raised_usd / h=30`
   - `MAE = 201925.6990`
   - `RMSE = 205733.8504`
   - `prediction_coverage_ratio = 1.0`
   - `fairness_pass = true`
   - `peak_rss_gb = 47.06`
   - this remains a local-only clear path and does not count as a canonical benchmark result
14. `5298437` `v740_elas_inv_clr` has now completed as the second `ElasTST` narrow probe:
   - `task2_forecast / core_edgar / investors_count / h=14`
   - `input_size=30`, `l_patch_size=6_10`
   - `MAE = 125.1687`
   - `prediction_coverage_ratio = 1.0`
   - `fairness_pass = false`
   - this is a useful negative result and keeps `ElasTST` in the local-clear lane rather than promoting it
15. `5298457` `v740_units_clr` has now completed successfully as the first `UniTS` narrow benchmark-clear probe:
   - `task2_forecast / core_edgar / funding_raised_usd / h=30`
   - `MAE = 131725.2212`
   - `fairness_pass = true`
16. `V740-alpha` has now advanced to a first audited `CASA + TimeEmb`-inspired mechanism pass:
   - funding `h=30` audited smoke: `MAE = 182493.8486`
   - binary `h=14` audited smoke: `MAE = 0.2853`
   - funding `h=60` audited smoke: `MAE = 176137.8275`
   - these remain local-only engineering audits and do not count as canonical benchmark results
15. The current 2026-03-30 10:45 recheck found **no additional missing canonical jobs** beyond the already repaired `5298506 l2_ac_t3_co`.
16. The same `CASA + TimeEmb` audit line now also has fuller-source `full` evidence:
   - `task1_outcome / full / is_funded / h=14`:
     - `MAE = 0.2758`
     - `text_source_density = 1.0`
     - still non-constant
   - `task1_outcome / full / is_funded / h=60`:
     - `MAE = 0.3296`
     - `text_source_density = 1.0`
     - still non-constant
   - `task2_forecast / full / funding_raised_usd / h=60`:
     - `MAE = 176137.0694`
     - `text_source_density = 1.0`
     - still non-constant
   - `task2_forecast / full / funding_raised_usd / h=30`:
     - `MAE = 182491.8726`
     - `text_source_density = 1.0`
     - still non-constant
   - current honest interpretation:
     - the first `full` binary slice improves slightly over `core_edgar`
     - the `full` binary line also stays non-degenerate at `h=60`, but is
       clearly harder than the matching `h=14` slice
     - the fuller-source funding `h=60` slice is effectively tied with
       `core_edgar`, and the newer fuller-source funding `h=30` slice is also
       only a tie / slight edge
     - so text is source-covered there but not yet proven to be the main driver
17. `SAMformer` is no longer a single-slice local-clear entrant:
   - a second tiny audited funding smoke has now landed on
     `task2_forecast / core_edgar / funding_raised_usd / h=30`
   - result:
     - `MAE = 265368.0`
     - `constant_prediction = false`
   - the first promotion probe `5299018 v740_samf_fu_clr` failed immediately
     because the submission still used `--preset quick`, which cannot run
     `task2 / h=30`
   - the repaired funding promotion probe has now completed successfully:
     - `5299636 v740_samf_fu_clr`
     - `MAE = 130514.7325`
     - `RMSE = 174074.8917`
     - `prediction_coverage_ratio = 1.0`
     - `fairness_pass = true`
18. `LightGTS` has now also crossed the “single audited slice only” boundary:
   - a fuller-source tiny funding smoke has landed on
     `task2_forecast / full / funding_raised_usd / h=30`
   - result:
     - `MAE = 445368.0`
     - `constant_prediction = false`
   - the fuller-source promotion probe has now completed successfully:
     - `5299637 v740_lgts_fu_clr`
     - `MAE = 201930.5506`
     - `RMSE = 205737.7476`
     - `prediction_coverage_ratio = 1.0`
     - `fairness_pass = true`
19. `5299641 v740_elas_i21_clr` has now completed as a bounded count-side repair probe for `ElasTST`:
   - scope:
     - `task2_forecast / core_edgar / investors_count / h=14`
     - `input_size = 21`, `l_patch_size = 3_6`
   - rationale:
     - the more conservative local smoke remained non-constant,
     - the queue still had room for one additional local-only GPU decision job,
     - and this is the cleanest way to decide whether `ElasTST` count-side can recover or should stay funding-only
   - audited result:
     - `MAE = 124.4985`
     - `RMSE = 124.4985`
     - `prediction_coverage_ratio = 1.0`
     - `fairness_pass = false`
20. `5298523` `v740_olnr_inv` has now completed as the first count-side `OLinear` narrow clear:
   - `task2_forecast / core_edgar / investors_count / h=14`
   - `input_size = 30`, `temp_patch_len = 6`, `temp_stride = 6`
   - `MAE ≈ 4.58e-05`
   - `prediction_coverage_ratio = 1.0`
   - `fairness_pass = false`
   - log evidence shows a constant `137.0` prediction on the audited slice
   - current honest interpretation:
     - `OLinear` funding-side evidence remains promising,
     - but the count-side lane is not promotable right now
21. `5298559 v740_units_inv_clr` has now completed, and the result is decisive:
   - scope:
     - `task2_forecast / core_edgar / investors_count / h=14`
     - `input_size = 30`, `patch_len = 5`, `stride = 5`
   - audited result:
     - `MAE = 6.1035e-05`
     - `prediction_coverage_ratio = 1.0`
     - `fairness_pass = false`
   - log evidence shows a constant `136.9999` prediction on the audited slice
   - current honest interpretation:
     - the richer local smoke was only a screening signal,
     - `UniTS` remains promotable on the funding lane only,
     - its current count-side lane is not promotable
22. Additional partition-specific duplicates were correctly **not** launched for `5298559`:
   - `gpu` already had a concrete same-day start time and then completed quickly
   - a `hopper` test-only duplicate would start much later (`2026-05-06 03:18:22`)
   - the current `l40s` submission path for this local-only pattern is rejected under the present account/QoS context
   - so not duplicating it avoided extra queue pressure without losing time
23. The wider local promotion lanes for `SAMformer` and `LightGTS` are now already read out:
   - `5300057 v740_samf_lane` → completed cleanly, current harness surface landed:
     - `task2_forecast / core_edgar / funding_raised_usd / h=30`
     - `MAE = 130514.7325`
     - `fairness_pass = true`
   - `5300058 v740_lgts_lane` → completed cleanly, current harness surface landed:
     - `task2_forecast / full / funding_raised_usd / h=30`
     - `MAE = 201930.5506`
     - `fairness_pass = true`
   - current honest label:
     - both are now promotion-positive funding-side lines
24. The paired V740 text-gain audit has now completed cleanly:
   - `5300635 v740_txtgain_bin`
   - scope:
     - `task1_outcome / {core_edgar, full} / is_funded / h={14,60}`
   - audited result:
     - `full / h=14`: `MAE = 0.3264`
     - `core_edgar / h=14`: `MAE = 0.3281`
     - `full / h=60`: `MAE = 0.3056`
     - `core_edgar / h=60`: `MAE = 0.3741`
   - current honest interpretation:
     - on these audited fuller-source binary slices, text is now a **real gain**
       rather than only stable source coverage
25. Two more-serious local-only promotion lanes are now queued:
   - `5300644 v740_samf_bin4h`
   - `5300645 v740_lgts_f4h`
   - current purpose:
     - move `SAMformer` and `LightGTS` beyond single-condition promotion-positive
       probes into the first multi-horizon lane expansion
