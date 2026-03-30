# Current Queue Progress

> Snapshot time: 2026-03-30 10:32 CEST
> Source: live `squeue -u npin,cfisch`, `squeue --start -u npin,cfisch`, and `sacct -u npin -S 2026-03-28T00:00:00`

## Summary

| Metric | Value |
| --- | ---: |
| Total jobs | 40 |
| Running | 9 |
| Pending | 31 |
| gpu running | 6 |
| gpu pending | 0 |
| bigmem running | 0 |
| l40s running | 3 |
| l40s pending | 14 |
| hopper running | 0 |
| hopper pending | 17 |

## Running Jobs

| jobid | job | partition | elapsed | limit | progress | time_left | reason/node |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| 5298285 | af739_t1_e2 | gpu | 9:54:07 | 2-00:00:00 | 20.6% | 1d14h5m | iris-185 |
| 5298286 | af739_t2_e2 | gpu | 9:54:07 | 2-00:00:00 | 20.6% | 1d14h5m | iris-185 |
| 5298287 | af739_t3_e2 | gpu | 9:54:07 | 2-00:00:00 | 20.6% | 1d14h5m | iris-186 |
| 5298288 | gpu_cos2_t2 | gpu | 9:54:07 | 2-00:00:00 | 20.6% | 1d14h5m | iris-169 |
| 5298048 | af739_t2_s2 | gpu | 20:14:16 | 2-00:00:00 | 42.2% | 1d3h45m | iris-186 |
| 5298049 | af739_t1_s2 | gpu | 20:14:16 | 2-00:00:00 | 42.2% | 1d3h45m | iris-170 |
| 5269760 | l2_ac_t3_ce | l40s | 9:03:36 | 2-00:00:00 | 18.9% | 1d14h56m | iris-198 |
| 5279084 | l2_ac_t1_ct | l40s | 16:14:34 | 2-00:00:00 | 33.8% | 1d7h45m | iris-199 |
| 5269764 | l2_ac_t3_fu | l40s | 18:21:54 | 2-00:00:00 | 38.3% | 1d5h38m | iris-199 |

## Highest-Value Pending Jobs

| jobid | job | partition | planned_start | reason | notes |
| --- | --- | --- | --- | --- | --- |
| 5279085 | l2_ac_t1_e2 | l40s | 2026-03-31 14:20:21 | Resources | first l40s slot with a concrete restart time |
| 5269749 | l2_ac_t1_co | l40s | 2026-03-31 16:07:37 | Priority | next l40s backlog item with a concrete start time |
| 5269752 | l2_ac_t1_fu | l40s | 2026-04-02 14:30:00 | Priority | later l40s restart already scheduled |
| 5294241 | l2_ac_t2_s2 | l40s | 2026-04-04 14:30:00 | Priority | resumed overflow copy after 2-day TIMEOUT on `5269759` |
| 5298506 | l2_ac_t3_co | l40s | N/A | Priority | newly resubmitted after verified 2-day TIMEOUT on `5269761`; resumable continuation, not a structural failure |
| 5269765 | h2_ac_t1_ce | hopper | 2026-03-31 03:20:00 | Priority | opportunistic overflow backlog |
| 5269771 | h2_ac_t2_ce | hopper | 2026-03-31 03:20:00 | Priority | opportunistic overflow backlog |
| 5269780 | h2_ac_t3_e2 | hopper | 2026-03-31 03:20:00 | Priority | opportunistic overflow backlog |

## Pending Breakdown

| partition | pending_jobs | notes |
| --- | ---: | --- |
| gpu | 0 | canonical gpu critical path is fully live again; no gpu backlog remains at this snapshot |
| l40s | 14 | accel_v2 overflow backlog; current live state has three active `l40s` runners plus newly requeued `l2_ac_t3_co` |
| hopper | 17 | pure overflow backlog only |

## Canonical Benchmark Recovery Notes

1. The gpu critical path is fully live again: all five `af739_*` gap-fill jobs plus `gpu_cos2_t2` are running.
2. `af739_t1_s2` and `af739_t2_s2` failed once more at `189G` (`MaxRSS≈198.18G`) and were resubmitted as `5298049` / `5298048` at `200G / 8 CPU`.
3. `l2_ac_t2_s2` is the confirmed latest `l40s` timeout casualty on 2026-03-28; log evidence shows it progressed deep into `investors_count` before the 2-day wall, so it was resubmitted as `5294241`.
4. `l2_ac_t3_co` is now also a confirmed resumable timeout continuation:
   - `5269761 l2_ac_t3_co` hit the 2-day wall on 2026-03-30
   - the log shows it was cleanly continuing `investors_count` work when SLURM cancelled it for time limit
   - it has now been resubmitted as `5298506`
5. The `l40s` lane is no longer down to a single live job. At this snapshot:
   - `l2_ac_t3_ce`
   - `l2_ac_t1_ct`
   - `l2_ac_t3_fu`
   are all RUNNING, so the current queue is `9R+30PD`.
6. Local-only V740 model-clear status has advanced:
   - `5294242` `v740_samf_clr` → completed successfully
   - `5294243` `v740_prop_clr` → failed before model execution because `quick` preset cannot run `h=30`
   - `5294254` `v740_prop_std` → corrected `Prophet` resubmission, completed successfully
   - `5294255` `v740_tpfn26c` → first `TabPFN 2.6` narrow clear, completed successfully
   - `5294259` `v740_tpfn26r_fu` → completed as the first `TabPFNRegressor` funding clear (`fairness_pass=true`, quality weak)
   - `5294260` `v740_tpfn26r_inv` → completed as the first `TabPFNRegressor` investors-count clear, but with `fairness_pass=false`
7. These local-only jobs use separate output roots and do not count as canonical benchmark results.
8. `5298059` `v740_lgts_clr` is no longer pending. It completed without model execution because the compute node could not see `/tmp/LightGTS`.
9. The repaired `LightGTS` rerun `5298289` completed successfully after switching the vendor repo to the persistent path `~/.cache/block3_optional_repos/LightGTS`.
10. `5298296` `v740_olnr_clr` has also completed successfully as the first audited `OLinear` funding narrow clear:
   - `task2_forecast / core_edgar / funding_raised_usd / h=30`
   - `MAE = 131288.8062`
   - `fairness_pass = true`
   - this remains a local-only side path and does not count as a canonical benchmark landing
11. `5298399` `v740_elas_clr` has now completed successfully as the first `ElasTST` narrow benchmark-clear probe:
   - `task2_forecast / core_edgar / funding_raised_usd / h=30`
   - `MAE = 201925.6990`
   - `RMSE = 205733.8504`
   - `prediction_coverage_ratio = 1.0`
   - `fairness_pass = true`
   - `peak_rss_gb = 47.06`
   - this remains a local-only clear path and does not count as a canonical benchmark result
12. `5298437` `v740_elas_inv_clr` has now completed as the second `ElasTST` narrow probe:
   - `task2_forecast / core_edgar / investors_count / h=14`
   - `input_size=30`, `l_patch_size=6_10`
   - `MAE = 125.1687`
   - `prediction_coverage_ratio = 1.0`
   - `fairness_pass = false`
   - this is a useful negative result and keeps `ElasTST` in the local-clear lane rather than promoting it
13. `5298457` `v740_units_clr` has now completed successfully as the first `UniTS` narrow benchmark-clear probe:
   - `task2_forecast / core_edgar / funding_raised_usd / h=30`
   - `MAE = 131725.2212`
   - `fairness_pass = true`
14. `V740-alpha` has now advanced to a first audited `CASA + TimeEmb`-inspired mechanism pass:
   - funding `h=30` audited smoke: `MAE = 182493.8486`
   - binary `h=14` audited smoke: `MAE = 0.2853`
   - funding `h=60` audited smoke: `MAE = 176137.8275`
   - these remain local-only engineering audits and do not count as canonical benchmark results
