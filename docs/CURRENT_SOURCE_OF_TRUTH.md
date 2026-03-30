# Current Source of Truth

> Last verified: 2026-03-30 14:12 CEST
> Verified by direct scans of `runs/benchmarks/block3_phase9_fair/`, live `squeue -u npin`, `sacct`, benchmark aggregation scripts.

This file is the authoritative documentation entry point for the current Block 3 project state.
If any other document disagrees with this file, prefer this file and the evidence paths cited below.

## Authoritative Sources (Read in This Order)

1. Root `AGENTS.md`
2. `.local_mandatory_preexec.md`
3. `docs/CURRENT_SOURCE_OF_TRUTH.md`
4. `docs/PHASE9_V739_FACT_ALIGNMENT.md`
5. `docs/BLOCK3_MODEL_STATUS.md`
6. `docs/BLOCK3_RESULTS.md`
7. `docs/benchmarks/phase9_current_snapshot.md`
8. `docs/V739_CURRENT_RUN_MONITOR.md`
9. `docs/PHASE12_TEXT_RERUN_EXECUTION.md`
10. `runs/benchmarks/block3_phase9_fair/all_results.csv`
11. `runs/benchmarks/block3_phase9_fair/REPLICATION_MANIFEST.json`

## Verified Current Facts

| Fact | Current value | Evidence |
| --- | --- | --- |
| Canonical benchmark directory | `runs/benchmarks/block3_phase9_fair/` | direct scan |
| Raw metric records | `16152` | direct scan 2026-03-30 |
| Raw models materialized | `137` | direct scan (116 real + 21 retired AutoFit@1) |
| Audit-excluded models | `24` | AUDIT_EXCLUDED_MODELS (17 old + 7 Finding H) |
| Active (leaderboard) models | `92` | 116 raw - 24 excluded |
| Raw complete models (`@160`) | `75` | direct unique-condition scan |
| Active complete models (`@160`) | `62` | 75 raw complete - 13 excluded complete models |
| Incomplete active models | `30` | 92 - 62 |
| Per-ablation | co=2849, s2=2139, ce=2780, e2=2778, ct=2764, fu=2767 | direct scan 2026-03-26 |
| Conditions per model | `160` | t1(72) + t2(48) + t3(40) |
| Current AutoFit baseline | `AutoFitV739` only | Root `AGENTS.md` |
| V739 landed conditions | `132/160` | co=28, ce=28, ct=28, fu=28, s2/e2 gap-filling still in progress |
| V739 quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| V739 mean rank | **#13** (top 14%, 92 active models) | per-condition ranking (last computed) |
| Post-filter distinct models in `all_results.csv` | `107` | includes 21 retired AutoFit legacy lines that still pass fairness/coverage filters |
| Post-filter non-retired models | `86` | `all_results.csv` minus retired AutoFit legacy lines |
| Text embedding artifacts | `AVAILABLE` | `runs/text_embeddings/embedding_metadata.json` |
| Phase 12 text reruns | `48/48 COMPLETED` | core_text+full 91/91 models |
| Phase 15 new models | 23 submitted, 15 valid, 8 excluded (Finding H), 78/160 | direct scan |
| Live jobs | `41` (10R + 31PD) | direct `squeue -u npin` 2026-03-30 14:12: gpu 6R+1PD, l40s 4R+13PD, hopper 0R+17PD |
| Clean full comparable frontier | `55` models @ shared `160/160` | post-filter `all_results.csv`, non-retired only |

## What the Current Benchmark Means

1. Phase 7 and Phase 8 results are historical only. They are not valid current benchmark evidence.
2. V734-V738 are retired because of oracle test-set leakage. They must not be used as baselines, ranking references, or implementation templates.
3. The current clean AutoFit line starts at V739, which uses validation-based selection (`val_raw`) instead of oracle tables.
4. The current physical Phase 9 fair benchmark has 6 ablations:
   - `core_only` (co) — baseline numeric features
   - `core_only_seed2` (s2) — seed stability check
   - `core_edgar` (ce) — EDGAR features added
   - `core_edgar_seed2` (e2) — EDGAR + seed2
   - `core_text` (ct) — text embeddings added
   - `full` (fu) — all features combined
5. Text embedding artifacts exist. Phase 12 text reruns COMPLETED (48/48 jobs). `core_text` / `full` reflect real text features.
6. Phase 15 added 23 new TSLib models. 8 produce 100% constant predictions (Finding H) and are excluded. 15 valid models remain at ~67/160 conditions.

## Current Execution Reality

1. Live queue snapshot verified on 2026-03-30 14:12 CEST:
   - `10 RUNNING` = `6 gpu + 4 l40s + 0 hopper`
   - `31 PENDING` = `1 gpu + 13 l40s + 17 hopper`
   - **41 total**
   - Current gpu runners:
     - `af739_t1_s2` (`5298049`)
     - `af739_t2_s2` (`5298048`)
     - `af739_t1_e2` (`5298285`)
     - `af739_t2_e2` (`5298286`)
     - `af739_t3_e2` (`5298287`)
     - `gpu_cos2_t2` (`5298288`)
   - Current l40s runners:
     - `l2_ac_t1_e2`
     - `l2_ac_t3_ce`
     - `l2_ac_t1_ct`
     - `l2_ac_t3_fu`
   - Current hopper runners: none; hopper remains pure overflow backlog right now
   - **ModernTCN bottleneck** remains the dominant throughput limiter for non-e2 accel jobs
   - Partition constraints (`sinfo` verified): `gpu=756G`, `l40s=515G`, `hopper=2063754MB (~2.06TB)`; the earlier `hopper=201G` claim was a unit-reading error
2. V739 status:
   - **132/160 conditions landed** (+1 from 131, s2/e2 gap-filling)
   - 5 af739 jobs live: `5 RUNNING + 0 PENDING`
   - RUNNING:
     - `t1_e2` as `5298285`
     - `t1_s2` as `5298049`
     - `t2_e2` as `5298286`
     - `t2_s2` as `5298048`
     - `t3_e2` as `5298287`
   - Missing structure: `t1_e2=9`, `t1_s2=8`, `t2_e2=4`, `t2_s2=4`, `t3_e2=3` (28 total)
   - V739 is empirically valid: 0 NaN/Inf, 0 fallback, 100% fairness pass
   - The two seed2 jobs (`t1_s2`, `t2_s2`) OOMed once at `150G` and then again at `189G` (`MaxRSS ≈ 198.18G` on the second repaired copies)
   - they are now running as:
     - `5298049 af739_t1_s2` at `200G / 8 CPU`
     - `5298048 af739_t2_s2` at `200G / 8 CPU`
   - `t1_e2`, `t2_e2`, `t3_e2`, and `gpu_cos2_t2` all hit the 2-day wall again in their repaired copies on 2026-03-29 and were explicitly resubmitted as:
     - `5298285 af739_t1_e2`
     - `5298286 af739_t2_e2`
     - `5298287 af739_t3_e2`
     - `5298288 gpu_cos2_t2`
3. Finding H (discovered 2026-03-22):
   - 8 P15 models produce 100% constant predictions (0% fairness pass)
   - CFPT, DeformableTST, MICN, PathFormer, SEMPO, SparseTSF, TimeBridge, TimePerceiver
   - Added to AUDIT_EXCLUDED_MODELS in aggregate_block3_results.py
   - Removed from accel_v2 scripts → ~30% faster per job
4. accel_v2 progress (current live reality):
   - the gpu critical path is still actively running: `gpu_cos2_t2` plus the five af739 gap-fill jobs are all on live GPU nodes
   - active accel_v2 runtime is now concentrated on `l40s`; hopper remains queued but not live
   - live `l40s` runners at this moment are:
     - `l2_ac_t3_ce`
     - `l2_ac_t1_ct`
     - `l2_ac_t3_fu`
   - `l2_ac_t2_s2` timed out at the 2-day wall on 2026-03-28 after advancing into the `investors_count` section and was resubmitted as `5294241`
   - `l2_ac_t3_co` also timed out at the 2-day wall on 2026-03-30 while cleanly continuing the resumable `investors_count` section and has now been resubmitted as `5298506`
   - the earlier short-lived `l40s` burst has not fully vanished; three `l40s` overflow jobs are still live while the rest are back in the pending backlog
   - **ModernTCN remains the universal bottleneck** for non-`e2` accel paths and is still the main reason resumable copies keep timing out before the backlog clears
   - `_requeue_handler()` remains present in the script surface; current progress still depends on partition availability more than on correctness bugs
5. Critical gaps:
   - s2 (core_only_seed2): `gpu_cos2_t2` remains the canonical task2 seed2 gap-fill and has now been **resubmitted again** as `5298288` after the latest 2-day timeout on `5290365`
   - e2 (core_edgar_seed2): 2778 records (+54 from 2724), accel_v2 e2 scripts producing the bulk of recent growth
   - V739: 28 missing s2+e2 conditions, and **all 5 required gap-fill jobs are now live** (`5 RUNNING + 0 PENDING`)
6. Text embeddings:
   - `runs/text_embeddings/text_embeddings.parquet` — 5,774,931 rows, 64 PCA dims
   - Phase 12 all 48/48 complete. core_text+full coverage: 91/91 models
7. Audit-excluded models: 24 total (was 23, added NegativeBinomialGLM as Structural)

8. Local-only V740 truth that is now settled enough to cite:
   - the first corrected local head-to-head for `mb_t1_core_edgar_is_funded_h14` has landed
   - `V739` beats `V740-alpha` on that slice:
     - `V739`: `MAE = 0.1623`, selected model `PatchTST`, fit time `301.3s`
     - `V740-alpha`: `MAE = 0.2016`, fit time `38.9s`
   - interpretation: `V740-alpha` is still much cheaper, but it is **not yet** strong enough to replace V739 on this audited binary EDGAR slice
9. Local-only model-clear queue state:
   - `5294242` `v740_samf_clr` **completed successfully** for a narrow `SAMformer` benchmark-clear probe
   - a second `SAMformer` funding-side tiny smoke is now also non-fallback:
     - `task2_forecast / core_edgar / funding_raised_usd / h=30`
     - `MAE = 265368.0`
     - `constant_prediction = false`
   - that evidence was strong enough to justify a second narrow-clear probe:
     - `5299018` `v740_samf_fu_clr`
     - current state: `PENDING`
   - `5294243` `v740_prop_clr` **failed before model execution** because `quick` preset does not support `h=30`
   - `5294254` `v740_prop_std` **completed successfully** as the corrected `Prophet` resubmission on `bigmem`
   - `5294255` `v740_tpfn26c` **completed successfully** as the first narrow `TabPFNClassifier` + official 2.6 checkpoint benchmark-clear probe
   - `5294259` `v740_tpfn26r_fu` **completed successfully** as the first `TabPFNRegressor` funding narrow clear
   - `5294260` `v740_tpfn26r_inv` **completed successfully**, but returned `fairness_pass = false` on the audited investors-count slice and therefore must be treated as a red-flag result rather than a promotable clear
   - all of these write to isolated output roots under `runs/benchmarks/block3_phase9_localclear_20260328/` or `runs/benchmarks/block3_phase9_localclear_20260330/` and do **not** count as canonical benchmark results
10. `LightGTS` local integration has now crossed both the first real-data gate and the first narrow benchmark-clear gate:
   - the vendor repo audit and import smoke had already passed
   - the first tiny real-data funding smoke at `input_size=96` produced **no training windows** and therefore fell back to a constant predictor
   - a second tiny real-data funding smoke at `input_size=60`, `patch_len=15`, `stride=15` is now **non-fallback and non-constant**
   - slice:
     - `task2_forecast / core_edgar / funding_raised_usd / h=30`
     - `4 entities / 300 train rows`
   - result:
     - `fit_seconds = 4.60`
     - `prediction_std = 10802.78`
     - `constant_prediction = false`
     - `MAE = 445367.99`
   - the first narrow benchmark-clear attempt (`5298059`) completed without model execution because the compute node could not see the vendor repo path (`/tmp/LightGTS`)
   - after moving the official repo to the persistent user-local path `~/.cache/block3_optional_repos/LightGTS`, the repaired rerun (`5298289`) completed successfully
   - narrow benchmark-clear result on the audited slice:
     - `task2_forecast / core_edgar / funding_raised_usd / h=30`
     - `MAE = 201930.5506`
     - `RMSE = 205737.7476`
     - `prediction_coverage_ratio = 1.0`
     - `fairness_pass = true`
     - `peak_rss_gb = 47.40`
   - interpretation:
     - `LightGTS` is now genuinely past the “import only” stage
     - it has a real narrow harness clear
     - it is still a local-clear side path and does **not** yet count as a canonical benchmark entrant
11. `OLinear` has now crossed the same first serious local-only integration gates:
   - the official repo is audited and pinned locally at:
     - `~/.cache/block3_optional_repos/OLinear`
     - HEAD `f168e01a3e0e316ad98330b5e77afed1f77b0af5`
   - local integration is no longer blocked only on paper by artifact generation:
     - a Block 3 vendor-path helper exists in `src/narrative/block3/models/optional_runtime.py`
     - a real local wrapper exists in `src/narrative/block3/models/olinear_model.py`
     - the model is now locally registered through `src/narrative/block3/models/deep_models.py`
   - first tiny real-data funding smoke on:
     - `task2_forecast / core_edgar / funding_raised_usd / h=30`
     - is **non-fallback** and **non-constant**
     - `fit_seconds = 15.35`
     - `MAE = 265368.0`
12. `ElasTST` is no longer docs-only:
   - the official ProbTS repo is audited locally at:
     - `~/.cache/block3_optional_repos/ProbTS`
   - a local vendor-import wrapper now exists at:
     - `src/narrative/block3/models/elastst_model.py`
   - the first funding smoke on:
     - `task2_forecast / core_edgar / funding_raised_usd / h=30`
     - is **non-fallback** and **non-constant**
     - `fit_seconds = 2.02`
     - `MAE = 445367.47`
   - the first shorter-context investors smoke on:
     - `task2_forecast / core_edgar / investors_count / h=14`
     - `input_size = 30`, `l_patch_size = 6_10`
     - is also **non-fallback**
     - `fit_seconds = 2.30`
     - `MAE = 47.52`
   - the first narrow benchmark-clear job has already completed successfully:
     - `5298399 v740_elas_clr`
     - `task2_forecast / core_edgar / funding_raised_usd / h=30`
     - `MAE = 201925.6990`
     - `RMSE = 205733.8504`
     - `fairness_pass = true`
     - `prediction_coverage_ratio = 1.0`
     - `peak_rss_gb = 47.06`
   - the second narrow benchmark-clear job has also completed:
     - `5298437 v740_elas_inv_clr`
     - `task2_forecast / core_edgar / investors_count / h=14`
     - `input_size = 30`, `l_patch_size = 6_10`
     - `MAE = 125.1687`
     - `RMSE = 125.1687`
     - `fairness_pass = false`
     - `prediction_coverage_ratio = 1.0`
     - `peak_rss_gb = 46.24`
   - this remains a local-only side path and does **not** count as a canonical benchmark result
   - first tiny real-data investors-count smoke on:
     - `task2_forecast / core_edgar / investors_count / h=14`
     - falls back to a constant predictor on that tiny slice
   - first narrow benchmark-clear rerun:
     - `5298296` `v740_olnr_clr`
     - completed successfully on the real harness
13. `UniTS` is also no longer docs-only:
   - the official repo is audited locally at:
     - `~/.cache/block3_optional_repos/UniTS`
     - HEAD `0e0281482864017cac8832b2651906ff5375a34e`
   - a forecasting-only Block 3 wrapper now exists at:
     - `src/narrative/block3/models/units_model.py`
   - first tiny funding smoke on:
     - `task2_forecast / core_edgar / funding_raised_usd / h=30`
     - is **non-fallback**
     - `fit_seconds = 14.95`
     - `MAE = 265368.0`
   - first narrow benchmark-clear:
     - `5298457 v740_units_clr`
     - `MAE = 131725.2212`
     - `RMSE = 174105.8744`
     - `prediction_coverage_ratio = 1.0`
     - `fairness_pass = true`
   - `peak_rss_gb = 47.73`
   - this remains a local-only side path and does **not** count as a canonical benchmark landing
14. `OLinear` count-side is now settled enough for an honest decision:
   - a richer local smoke on:
     - `task2_forecast / core_edgar / investors_count / h=14`
     - `input_size = 30`, `temp_patch_len = 6`, `temp_stride = 6`
     - became **non-fallback** and **non-constant**
     - `MAE = 0.0206`
   - but the matching audited harness narrow clear:
     - `5298523 v740_olnr_inv`
     - finishes with `fairness_pass = false`
     - and the log explicitly reports a constant `137.0` prediction on the audited slice
   - the correct operational conclusion is therefore:
     - keep `OLinear` as a promising **funding-side** entrant
     - do **not** promote its count-side lane yet
15. The first `CASA + TimeEmb`-inspired V740 alpha implementation pass is now audited on real freeze-backed slices:
   - code surface:
     - `src/narrative/block3/models/v740_alpha.py`
   - new lightweight mechanisms:
     - `CASALocalContextBlock` for cheap condition-aware local context mixing
     - `StaticDynamicTimeFusion` for TimeEmb-style static/dynamic/source fusion
   - three audited local smokes now exist:
     - `task2_forecast / core_edgar / funding_raised_usd / h=30`
       - `MAE = 182493.8486`
       - `constant_prediction = false`
     - `task1_outcome / core_edgar / is_funded / h=14`
       - `MAE = 0.2853`
       - `constant_prediction = false`
     - `task2_forecast / core_edgar / funding_raised_usd / h=60`
       - `MAE = 176137.8275`
       - `constant_prediction = false`
   - relative to the immediately prior audited alpha objective line, these new smokes show real directional improvement on the same funding and binary audited slices while staying non-degenerate
16. The same `CASA + TimeEmb` alpha line now also has fuller-source audited evidence:
   - `task1_outcome / full / is_funded / h=14`
     - `MAE = 0.2758`
     - `RMSE = 0.4962`
     - `text_source_density = 1.0`
     - `constant_prediction = false`
   - `task1_outcome / full / is_funded / h=60`
     - `MAE = 0.3296`
     - `RMSE = 0.5366`
     - `text_source_density = 1.0`
     - `constant_prediction = false`
   - `task2_forecast / full / funding_raised_usd / h=60`
     - `MAE = 176137.0694`
     - `RMSE = 367849.7154`
     - `text_source_density = 1.0`
     - `constant_prediction = false`
   - current honest interpretation:
     - the first fuller-source binary slice is slightly stronger than the
       matching `core_edgar` audit,
     - the fuller-source binary line stays non-degenerate even at `h=60`, but
        is clearly harder than the matching `h=14` slice,
     - while the fuller-source funding `h=60` slice is effectively tied with
        `core_edgar`, so text is source-covered there but not yet proven as the
        main gain source.
17. `UniTS` count-side now has a real harness-side decision:
   - richer local smoke on:
     - `task2_forecast / core_edgar / investors_count / h=14`
     - `input_size = 30`, `patch_len = 5`, `stride = 5`
     - had looked promising (`MAE = 0.0214`, non-constant)
   - the follow-up real harness clear has now landed:
     - `5298559 v740_units_inv_clr`
   - audited result:
     - `MAE = 6.1035e-05`
     - `prediction_coverage_ratio = 1.0`
     - `fairness_pass = false`
   - log evidence shows a constant `136.9999` prediction on the audited slice
   - current honest interpretation:
     - `UniTS` remains promising on the funding lane,
     - but its count-side lane is not promotable right now

## Current Priorities

1. ~~Land the first valid V739 results.~~ ✅ DONE (124/160 conditions landed, 112 co+ce+ct+fu complete).
2. ~~Finish gap-fill for partial models.~~ ✅ Current full-160 active frontier is 62 models; remaining incomplete active models are covered by accel_v2 or structural OOM exceptions.
3. ~~Submit real text-enabled reruns.~~ ✅ Phase 12 DONE (48/48 COMPLETED).
4. ~~Phase 12 text reruns to land.~~ ✅ DONE. core_text+full 91/91 models.
5. ~~Phase 15 new TSLib models.~~ ✅ Submitted. 15 valid, 8 excluded (Finding H).
6. ~~Cancel old v1 l40s/hopper jobs.~~ ✅ DONE. 8 old v1 jobs cancelled, freed L40S for v2.
7. Complete V739 s2/e2 gap-fill (currently `5 RUNNING + 0 PENDING`, all 5 live).
8. Complete e2 gap for ETSformer/LightTS/Pyraformer/Reformer (0/28 e2 each — covered by accel_v2).
9. Complete P15 model gap-fill to 160/160 via accel_v2 (54 total jobs, auto-requeue).
10. g2_ac_v2 auto-requeue: e2/ct complete first run; co/s2/ce/fu need 2-3 requeues, with `gpu_cos2_t2` now corrected to the trimmed 23-model list.
11. Local-only V740 work is now permitted via resumable side-paths that do **not** touch the canonical benchmark harness. The current completed local-only milestones are:
   - first corrected local `V739 vs V740` head-to-head (`mb_t1_core_edgar_is_funded_h14`)
   - larger-slice `full / funding_raised_usd / h=90 / input_size=120` audit
   Both completed successfully on 2026-03-27 and are now recorded in the V740 research notes.

## What Is No Longer Current

1. Everything under `docs/_legacy_repo/` is historical archive material.
2. The old V72/V73 truth-pack line under `docs/benchmarks/LEGACY__block3_truth_pack__v72_v73/` is historical evidence, not the current operational truth for Phase 9 / V739.
3. Research/reference notes under `docs/references/` are background knowledge only. They are useful for design, but they are not status documents.
4. Any archived local checklist copies under `docs/_legacy_repo/` are historical only and must not be treated as current execution truth.
5. The archived large result table at `docs/_legacy_repo/BLOCK3_RESULTS_table_20260314.md` is preserved for traceability, not for current operational reading.

## Validation Commands

```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/build_phase9_current_snapshot.py
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/aggregate_block3_results.py
squeue -u npin,cfisch
for jid in $(squeue -u npin,cfisch -h -o '%i %j' | awk '$2 ~ /(v739|af739)/ {print $1}'); do
  scontrol show job "$jid" | egrep 'JobId=|JobName=|Command=|WorkDir=|StdOut=|StdErr='
done
```
