# V740-alpha Selective / DistDF Audit (2026-03-30)

This note records the second implementation pass on the current
`Selective Learning + DistDF-style` objective scaffold inside
`src/narrative/block3/models/v740_alpha.py`.

It stays intentionally narrow:

- no canonical benchmark claims,
- no runtime-selector reintroduction,
- only freeze-backed audited local smokes,
- and only enough objective complexity to answer whether the new weighting and
  alignment logic remain stable and directionally useful.

## 1. What changed in the second pass

Compared with the first 2026-03-30 scaffold, the second pass makes two focused
changes.

### 1.1 Selective-learning weights are now source-aware at the window level

The first pass only used:

- target magnitude,
- history volatility,
- global source-density scalars.

The second pass now also uses:

- EDGAR recent-token activity,
- EDGAR recency-bucket activity,
- text recent-token activity,
- text recency-bucket activity,
- binary event / transition emphasis,
- horizon-sensitive weighting for non-binary targets.

This is still deliberately conservative. The allowed per-window weight range
remains clipped to `0.6–1.3`.

### 1.2 DistDF-style alignment is now slightly richer

The first pass aligned:

- horizon mean,
- horizon dispersion.

The second pass also aligns:

- horizon endpoint,
- step-difference shape.

This is still **not** a full paper-faithful DistDF implementation. It is a
light engineering approximation intended to test whether richer multistep
structure can be added without destabilizing alpha.

## 2. Funding sanity rerun on the same audited slice

Artifact:

- `docs/references/v740_alpha_selective_smoke_20260330/t2_core_edgar_funding_h30_v2.json`

Slice:

- task: `task2_forecast`
- ablation: `core_edgar`
- target: `funding_raised_usd`
- horizon: `30`
- max entities: `4`
- max rows: `300`
- max epochs: `2`

Result:

- `constant_prediction = false`
- `prediction_std = 3508.63`
- `edgar_source_density = 1.0`
- `MAE = 441156.71`
- `RMSE = 634565.63`
- `wall_time_seconds = 44.21`

Reference point from the earlier same-day first pass:

- previous artifact:
  - `docs/references/v740_alpha_selective_smoke_20260330/t2_core_edgar_funding_h30.json`
- previous result:
  - `MAE = 441572.68`

Interpretation:

- the second pass gives a **small real improvement**
- more importantly, it does **not** destabilize the current V740 execution path

## 3. Binary audited smoke on a hard EDGAR slice

Artifact:

- `docs/references/v740_alpha_selective_smoke_20260330/t1_core_edgar_is_funded_h14_v2.json`

Slice:

- task: `task1_outcome`
- ablation: `core_edgar`
- target: `is_funded`
- horizon: `14`
- max entities: `4`
- max rows: `300`
- max epochs: `2`

Result:

- `constant_prediction = false`
- `prediction_std = 0.01236`
- `binary_train_rate = 0.9048`
- `binary_event_rate = 1.0`
- `binary_transition_rate = 0.1667`
- `binary_temperature = 0.85`
- `task_mod_enabled = false`
- `binary_teacher_weight = 0.1280`
- `binary_event_weight = 0.1700`
- `teacher_logistic_mix = 0.3217`
- `teacher_tree_mix = 0.6783`
- `edgar_source_density = 0.7367`
- `MAE = 0.3089`
- `RMSE = 0.5484`
- `wall_time_seconds = 40.61`

Interpretation:

- the second-pass scaffold stays clean on a hard binary slice
- the binary regime logic still behaves as intended:
  - task modulation bypassed
  - teacher/event weights remain active
  - no constant collapse

This is still not enough to claim that V740 now beats V739 on the audited
binary EDGAR line. It only proves that the stronger objective scaffold remains
stable there.

## 4. Long-horizon audited smoke (`h=60`)

Artifact:

- `docs/references/v740_alpha_selective_smoke_20260330/t2_core_edgar_funding_h60_v2.json`

Slice:

- task: `task2_forecast`
- ablation: `core_edgar`
- target: `funding_raised_usd`
- horizon: `60`
- max entities: `8`
- max rows: `800`
- max epochs: `2`

Result:

- `constant_prediction = false`
- `prediction_std = 12579.79`
- `edgar_source_density = 1.0`
- `MAE = 232810.09`
- `RMSE = 445120.97`
- `wall_time_seconds = 34.26`

Interpretation:

- the second-pass scaffold also remains stable on a source-covered longer
  horizon slice
- this is useful because it shows the revised weighting/alignment logic is not
  restricted to short-horizon funding behavior

It should still **not** be over-read as final long-horizon evidence, because:

- the slice is local-only,
- the entity/row caps differ from earlier long-h runs,
- and the current question is still stability and directionality, not final
  benchmark superiority.

## 5. Third-pass mechanism add-on: lightweight CASA + TimeEmb

A later same-day engineering pass then added two **strictly lightweight**
mechanism stubs that had previously remained in the design docs only:

1. `CASALocalContextBlock`
   - a cheap condition-aware local-context gate over the dynamic sequence
   - intentionally **not** a paper-faithful CASA reimplementation
   - goal: give alpha a local-context refinement path without turning the trunk
     into a heavy attention model

2. `StaticDynamicTimeFusion`
   - a TimeEmb-inspired static/dynamic/source fusion gate
   - combines:
     - dynamic summary (`pooled`, `memory_feat`, `value_feat`)
     - source/static summary (`static_feat`, `edgar_feat`, `text_feat`)
     - condition token (`task/target/horizon/ablation`)
   - goal: let alpha modulate source/static information more explicitly rather
     than only concatenating it downstream

These two additions keep the one-model, one-inference-path constraint intact.

### 5.1 Funding rerun on the same audited `h=30` slice

Artifact:

- `docs/references/v740_alpha_selective_smoke_20260330/t2_core_edgar_funding_h30_v3.json`

Result:

- `constant_prediction = false`
- `prediction_std = 14231.45`
- `edgar_source_density = 1.0`
- `MAE = 182493.85`
- `RMSE = 364527.28`
- `wall_time_seconds = 63.12`

Relative to the second-pass result:

- previous artifact:
  - `docs/references/v740_alpha_selective_smoke_20260330/t2_core_edgar_funding_h30_v2.json`
- previous `MAE`:
  - `441156.71`

Interpretation:

- this is a **material improvement**, not just a no-regression check
- on this audited funding slice, the new local-context + fusion logic appears
  to matter more than the earlier objective-only refinements

### 5.2 Binary rerun on the hard EDGAR audited slice

Artifact:

- `docs/references/v740_alpha_selective_smoke_20260330/t1_core_edgar_is_funded_h14_v3.json`

Result:

- `constant_prediction = false`
- `prediction_std = 0.1060`
- `binary_train_rate = 0.9177`
- `binary_event_rate = 0.9348`
- `binary_transition_rate = 0.0217`
- `binary_temperature = 3.0`
- `task_mod_enabled = false`
- `binary_teacher_weight = 0.1395`
- `binary_event_weight = 0.1790`
- `teacher_logistic_mix = 0.2846`
- `teacher_tree_mix = 0.7154`
- `edgar_source_density = 0.9093`
- `MAE = 0.2853`
- `RMSE = 0.5011`
- `wall_time_seconds = 70.66`

Relative to the second-pass result:

- previous `MAE = 0.3089`

Interpretation:

- binary behavior remains non-degenerate
- and the new mechanism pass also gives a **real incremental gain** on this
  hard audited slice

### 5.3 Long-horizon rerun (`h=60`)

Artifact:

- `docs/references/v740_alpha_selective_smoke_20260330/t2_core_edgar_funding_h60_v3.json`

Result:

- `constant_prediction = false`
- `prediction_std = 12997.48`
- `edgar_source_density = 1.0`
- `MAE = 176137.83`
- `RMSE = 367851.11`
- `wall_time_seconds = 39.09`

Relative to the second-pass result:

- previous `MAE = 232810.09`

Interpretation:

- the first `CASA + TimeEmb` pass is not only helping short-horizon funding
  and binary slices
- it also improves the currently audited source-covered `h=60` funding slice

### 5.4 Fuller-source binary rerun (`full`, `h=14`)

Artifact:

- `docs/references/v740_alpha_selective_smoke_20260330/t1_full_is_funded_h14_v3.json`

Result:

- `constant_prediction = false`
- `prediction_std = 0.0848`
- `binary_temperature = 3.0`
- `edgar_source_density = 0.9093`
- `text_source_density = 1.0`
- `MAE = 0.2758`
- `RMSE = 0.4962`
- `wall_time_seconds = 66.18`

Relative to the matching core-edgar v3 binary audit:

- core-edgar artifact:
  - `docs/references/v740_alpha_selective_smoke_20260330/t1_core_edgar_is_funded_h14_v3.json`
- core-edgar `MAE`:
  - `0.2853`

Interpretation:

- the first fuller-source binary slice is also **directionally better** than
  the matching `core_edgar` slice
- this is useful because it shows the current static/dynamic/source fusion
  logic can absorb the extra text channel without collapsing binary behavior

### 5.5 Fuller-source long-horizon funding rerun (`full`, `h=60`)

Artifact:

- `docs/references/v740_alpha_selective_smoke_20260330/t2_full_funding_h60_v3.json`

Result:

- `constant_prediction = false`
- `prediction_std = 12997.13`
- `edgar_source_density = 1.0`
- `text_source_density = 1.0`
- `MAE = 176137.0694`
- `RMSE = 367849.7154`
- `wall_time_seconds = 38.75`

Relative to the matching `core_edgar` v3 long-horizon audit:

- core-edgar artifact:
  - `docs/references/v740_alpha_selective_smoke_20260330/t2_core_edgar_funding_h60_v3.json`
- core-edgar `MAE`:
  - `176137.8275`

Interpretation:

- this is effectively a **tie** with the matching `core_edgar` slice
- so the current honest statement is:
  - the `full` route is now source-covered and stable on this audited long-h
    slice,
  - but text is **not yet** clearly the main gain driver there

### 5.6 Fuller-source longer-horizon binary rerun (`full`, `h=60`)

Artifact:

- `docs/references/v740_alpha_selective_smoke_20260330/t1_full_is_funded_h60_v3.json`

Result:

- `constant_prediction = false`
- `prediction_std = 0.2045`
- `binary_temperature = 1.0`
- `edgar_source_density = 0.9093`
- `text_source_density = 1.0`
- `MAE = 0.3296`
- `RMSE = 0.5366`
- `wall_time_seconds = 42.71`

Relative to the matching fuller-source binary `h=14` audit:

- fuller-source `h=14` artifact:
  - `docs/references/v740_alpha_selective_smoke_20260330/t1_full_is_funded_h14_v3.json`
- fuller-source `h=14` `MAE`:
  - `0.2758`

Interpretation:

- this harder longer-h binary slice is still **non-degenerate** and fully
  source-covered,
- but it is clearly harder than the matching `h=14` slice,
- so the current honest claim is:
  - the `CASA + TimeEmb` pass now stays stable on a harder long-h fuller-source
    binary slice too,
- but this is not yet evidence of a decisive long-h binary breakthrough

### 5.7 Fuller-source short-horizon funding rerun (`full`, `h=30`)

Artifact:

- `docs/references/v740_alpha_selective_smoke_20260330/t2_full_funding_h30_v3.json`

Result:

- `constant_prediction = false`
- `prediction_std = 14233.66`
- `edgar_source_density = 1.0`
- `text_source_density = 1.0`
- `MAE = 182491.8726`
- `RMSE = 364524.0365`
- `wall_time_seconds = 35.2`

Relative to the matching `core_edgar` v3 short-horizon funding audit:

- core-edgar artifact:
  - `docs/references/v740_alpha_selective_smoke_20260330/t2_core_edgar_funding_h30_v3.json`
- core-edgar `MAE`:
  - `182493.8486`

Interpretation:

- this fuller-source funding slice is also **stable and source-covered**
- it is only a **tie / slight edge** over the matching `core_edgar` slice
- so the current honest statement remains:
  - text survives the new mechanism pass cleanly on this audited funding line,
  - but text is **not yet** proven to be the main gain driver there

## 6. Current honest takeaway

The current `Selective + DistDF + CASA + TimeEmb` alpha line is now supported
by ten real local facts:

1. second-pass objective-only funding smoke improves slightly,
2. second-pass hard binary EDGAR smoke stays non-degenerate,
3. second-pass longer-horizon funding smoke stays non-degenerate,
4. third-pass `h=30` funding smoke improves materially,
5. third-pass binary EDGAR smoke improves incrementally,
6. third-pass `h=60` funding smoke also improves materially,
7. third-pass fuller-source binary smoke improves slightly over `core_edgar`,
8. third-pass fuller-source `h=60` funding smoke stays stable and tied with the
   matching `core_edgar` slice,
9. third-pass fuller-source `h=60` binary smoke stays stable and non-degenerate,
   but remains clearly harder than the matching `h=14` fuller-source slice.
10. third-pass fuller-source `h=30` funding smoke also stays stable and
    effectively tied with the matching `core_edgar` slice.

So the current honest label is:

- **objective scaffold and first mechanism add-ons are both stronger than before**
- **fuller-source slices now also stay stable under the same mechanism pass**
- **the same mechanism pass now also survives a harder long-h fuller-source binary slice**
- **text is now repeatedly source-covered on fuller-source funding and binary slices**
- **but text is still not proven to be the dominant gain source on audited funding slices**
- **still pre-benchmark**
- **worth continuing**
- **not yet a decisive performance breakthrough**
