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

## 5. Current honest takeaway

The second-pass objective refinement is now supported by three real local facts:

1. same-slice funding smoke improves slightly,
2. hard binary EDGAR smoke stays non-degenerate,
3. longer-horizon funding smoke also stays non-degenerate.

So the current honest label is:

- **objective scaffold stronger than before**
- **still pre-benchmark**
- **worth continuing**
- **not yet a decisive performance breakthrough**
