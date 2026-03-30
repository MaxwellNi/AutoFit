# Current Queue Progress

> Snapshot time: 2026-03-30 08:57 CEST
> Source: live `squeue -u npin,cfisch`, `squeue --start -u npin,cfisch`, and `sacct -u npin -S 2026-03-28T00:00:00`

## Summary

| Metric | Value |
| --- | ---: |
| Total jobs | 39 |
| Running | 9 |
| Pending | 30 |
| gpu running | 6 |
| gpu pending | 0 |
| bigmem running | 0 |
| l40s running | 3 |
| l40s pending | 13 |
| hopper running | 0 |
| hopper pending | 17 |

## Running Jobs

| jobid | job | partition | elapsed | limit | progress | time_left | reason/node |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| 5298285 | af739_t1_e2 | gpu | 8:22:04 | 2-00:00:00 | 17.4% | 1d15h37m | iris-185 |
| 5298286 | af739_t2_e2 | gpu | 8:22:04 | 2-00:00:00 | 17.4% | 1d15h37m | iris-185 |
| 5298287 | af739_t3_e2 | gpu | 8:22:04 | 2-00:00:00 | 17.4% | 1d15h37m | iris-186 |
| 5298288 | gpu_cos2_t2 | gpu | 8:22:04 | 2-00:00:00 | 17.4% | 1d15h37m | iris-169 |
| 5298048 | af739_t2_s2 | gpu | 18:42:13 | 2-00:00:00 | 39.0% | 1d5h17m | iris-186 |
| 5298049 | af739_t1_s2 | gpu | 18:42:13 | 2-00:00:00 | 39.0% | 1d5h17m | iris-170 |
| 5269760 | l2_ac_t3_ce | l40s | 7:31:33 | 2-00:00:00 | 15.7% | 1d16h28m | iris-198 |
| 5279084 | l2_ac_t1_ct | l40s | 14:42:31 | 2-00:00:00 | 30.6% | 1d9h17m | iris-199 |
| 5269764 | l2_ac_t3_fu | l40s | 16:49:51 | 2-00:00:00 | 35.1% | 1d7h10m | iris-199 |

## Highest-Value Pending Jobs

| jobid | job | partition | planned_start | reason | notes |
| --- | --- | --- | --- | --- | --- |
| 5279085 | l2_ac_t1_e2 | l40s | 2026-03-31 14:20:21 | Resources | first l40s slot with a concrete restart time |
| 5269749 | l2_ac_t1_co | l40s | 2026-03-31 16:07:37 | Priority | next l40s backlog item with a concrete start time |
| 5269752 | l2_ac_t1_fu | l40s | 2026-04-02 14:30:00 | Priority | later l40s restart already scheduled |
| 5294241 | l2_ac_t2_s2 | l40s | 2026-04-04 14:30:00 | Priority | resumed overflow copy after 2-day TIMEOUT on `5269759` |
| 5269765 | h2_ac_t1_ce | hopper | 2026-03-31 03:20:00 | Priority | opportunistic overflow backlog |
| 5269771 | h2_ac_t2_ce | hopper | 2026-03-31 03:20:00 | Priority | opportunistic overflow backlog |
| 5269780 | h2_ac_t3_e2 | hopper | 2026-03-31 03:20:00 | Priority | opportunistic overflow backlog |

## Pending Breakdown

| partition | pending_jobs | notes |
| --- | ---: | --- |
| gpu | 0 | canonical gpu critical path is fully live again; no gpu backlog remains at this snapshot |
| l40s | 13 | accel_v2 overflow backlog; current live state has three active `l40s` runners |
| hopper | 17 | pure overflow backlog only |

## Canonical Benchmark Recovery Notes

1. The gpu critical path is fully live again: all five `af739_*` gap-fill jobs plus `gpu_cos2_t2` are running.
2. `af739_t1_s2` and `af739_t2_s2` failed once more at `189G` (`MaxRSS≈198.18G`) and were resubmitted as `5298049` / `5298048` at `200G / 8 CPU`.
3. `l2_ac_t2_s2` is the confirmed latest `l40s` timeout casualty on 2026-03-28; log evidence shows it progressed deep into `investors_count` before the 2-day wall, so it was resubmitted as `5294241`.
4. The `l40s` lane is no longer down to a single live job. At this snapshot:
   - `l2_ac_t3_ce`
   - `l2_ac_t1_ct`
   - `l2_ac_t3_fu`
   are all RUNNING, so the current queue is `9R+30PD`.
5. Local-only V740 model-clear status has advanced:
   - `5294242` `v740_samf_clr` → completed successfully
   - `5294243` `v740_prop_clr` → failed before model execution because `quick` preset cannot run `h=30`
   - `5294254` `v740_prop_std` → corrected `Prophet` resubmission, completed successfully
   - `5294255` `v740_tpfn26c` → first `TabPFN 2.6` narrow clear, completed successfully
   - `5294259` `v740_tpfn26r_fu` → completed as the first `TabPFNRegressor` funding clear (`fairness_pass=true`, quality weak)
   - `5294260` `v740_tpfn26r_inv` → completed as the first `TabPFNRegressor` investors-count clear, but with `fairness_pass=false`
6. These local-only jobs use separate output roots and do not count as canonical benchmark results.
7. `5298059` `v740_lgts_clr` is no longer pending. It completed without model execution because the compute node could not see `/tmp/LightGTS`.
8. The repaired `LightGTS` rerun `5298289` completed successfully after switching the vendor repo to the persistent path `~/.cache/block3_optional_repos/LightGTS`.
9. `5298296` `v740_olnr_clr` has also completed successfully as the first audited `OLinear` funding narrow clear:
   - `task2_forecast / core_edgar / funding_raised_usd / h=30`
   - `MAE = 131288.8062`
   - `fairness_pass = true`
   - this remains a local-only side path and does not count as a canonical benchmark landing
10. `5298399` `v740_elas_clr` has now completed successfully as the first `ElasTST` narrow benchmark-clear probe:
   - `task2_forecast / core_edgar / funding_raised_usd / h=30`
   - `MAE = 201925.6990`
   - `RMSE = 205733.8504`
   - `prediction_coverage_ratio = 1.0`
   - `fairness_pass = true`
   - `peak_rss_gb = 47.06`
   - this remains a local-only clear path and does not count as a canonical benchmark result
11. `5298437` `v740_elas_inv_clr` has now completed as the second `ElasTST` narrow probe:
   - `task2_forecast / core_edgar / investors_count / h=14`
   - `input_size=30`, `l_patch_size=6_10`
   - `MAE = 125.1687`
   - `prediction_coverage_ratio = 1.0`
   - `fairness_pass = false`
   - this is a useful negative result and keeps `ElasTST` in the local-clear lane rather than promoting it
