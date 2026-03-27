# V740 Alpha Long-Horizon Local Smoke (2026-03-26)

> Date: 2026-03-26
> Scope: local-only prototype expansion beyond the current benchmark horizon grid
> Status: first real `h>30` V740-alpha evidence, now extended with `45` and `180`
>
> This note does **not** change the current clean benchmark protocol.
> The official comparable benchmark surface remains `h in {1, 7, 14, 30}`.
> The purpose of this note is to record the first verified local-only V740-alpha
> runs at longer forecast horizons after moving the prototype away from a tiny
> hard-coded horizon table and toward a mixed continuous-plus-bucket horizon
> conditioning path.

## 1. Why this exists

As of 2026-03-26, the current benchmark and the earlier V740 local tools only
produced directly comparable results on `h = 1, 7, 14, 30`. The project config
already contained longer context and `k` values, but that was not the same as
having verified V740 behavior at forecast horizons above 30.

This local-only experiment exists to answer the narrower engineering question:

> Can the current V740-alpha prototype run correctly at `h = 60` without
> collapsing, and if so, what does the first real-data result look like?

## 2. Implementation change required

Before this line of experiments, `V740AlphaPrototypeWrapper` only mapped a
small discrete horizon table and the local smoke runner only accepted
`{1, 7, 14, 30, 60, 90}`.

On 2026-03-26 the local prototype was extended in two ways:

1. the horizon conditioning path no longer depends on a tiny fixed lookup table;
   instead it now uses:
   - continuous horizon features
   - coarse horizon buckets
   - context-to-horizon ratio features
2. the local smoke runner now accepts arbitrary positive integer horizons
   rather than a short hard-coded choice list.

This matters because the entrepreneurial-finance research horizons that are
actually interesting are not limited to `60/90`. The intended research ladder
now is:

- `45` days: campaign follow-up / traction
- `60` days: medium-term progression
- `90` days: quarter-scale evolution
- `180` days: half-year financing progression
- `365` days: annual status change

This remains a **local-only prototype extension**. It is not yet part of the
official benchmark execution grid.

## 3. First verified `h=60` run

Command:

```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 \
  scripts/run_v740_alpha_smoke_slice.py \
  --task task2_forecast \
  --ablation core_only \
  --target funding_raised_usd \
  --horizon 60 \
  --max-entities 12 \
  --max-rows 1500 \
  --max-epochs 2 \
  --output-json docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_funding_h60.json
```

Observed result:

- task: `task2_forecast`
- ablation: `core_only`
- target: `funding_raised_usd`
- horizon: `60`
- train rows: `1500`
- val rows: `1008`
- test rows: `972`
- train matrix rows: `838`
- test matrix rows: `972`
- feature count: `77`
- selected entities: `12`
- coverage ratio: `1.0`
- constant prediction: `false`
- prediction std: `3790.47`
- fit time: included in total wall time
- wall time: `42.142s`

Metrics:

- `MAE = 360123.8448`
- `RMSE = 399021.8354`
- `sMAPE = 197.9286`
- `MAPE = 99.4656`

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_funding_h60.json`

## 4. What this does and does not prove

This run proves the following:

1. V740-alpha now has a real, exercised `h=60` code path.
2. The longer-horizon local prototype does not trivially fail on this slice.
3. The resulting prediction is not a constant-collapse artifact.

This run does **not** prove the following:

1. that `h=60` is better than `h=30`,
2. that longer forecast horizons help on the official benchmark,
3. that this should be folded into the current clean benchmark protocol,
4. that binary or multimodal `h>30` behavior is already understood.

## 5. Same-slice `h=30` control run

To avoid over-interpreting a single `h=60` number, the same local-only slice
was also run at `h=30`:

```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 \
  scripts/run_v740_alpha_smoke_slice.py \
  --task task2_forecast \
  --ablation core_only \
  --target funding_raised_usd \
  --horizon 30 \
  --max-entities 12 \
  --max-rows 1500 \
  --max-epochs 2 \
  --output-json docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_funding_h30.json
```

Observed result:

- constant prediction: `true`
- prediction std: `0.0`
- `MAE = 361820.0564`
- `RMSE = 400352.1908`
- `wall_time = 39.362s`

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_funding_h30.json`

Within this exact local slice, the first comparison therefore looks like:

| Horizon | Constant | MAE | RMSE | Wall time |
|---|---:|---:|---:|---:|
| `30` | `true` | `361820.06` | `400352.19` | `39.36s` |
| `60` | `false` | `360123.84` | `399021.84` | `42.14s` |

This is not enough evidence to claim that `h=60` is globally better than
`h=30`. It is enough to say that on this first verified local-only V740 slice,
the longer horizon is at least viable and, in this case, behaves slightly more
usefully than the shorter-horizon control.

## 6. Additional same-day `investors_count` check

To avoid overfitting the interpretation to one funding slice, the same local
protocol was also applied to `task2_forecast / core_only / investors_count`
with the same `12 entities / 1500 train rows / 2 epochs` budget.

Observed results:

| Target | Horizon | Constant | MAE | RMSE | Wall time |
|---|---:|---|---:|---:|---:|
| `investors_count` | `30` | `false` | `48.6373` | `55.8018` | `43.925s` |
| `investors_count` | `60` | `true` | `50.0833` | `57.0665` | `33.341s` |

Artifacts:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_investors_h30.json`
- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_investors_h60.json`

This second check matters because it shows that the first `funding_raised_usd`
signal does **not** generalize automatically. On this count target, the local
`h=30` slice is healthier than `h=60`, and the longer horizon collapses to a
constant prediction.

## 7. Current best interpretation

The correct interpretation is conservative:

- `h>30` is no longer purely hypothetical for V740-alpha,
- but it is still only a **prototype local-only** path,
- and it should be evaluated first as an engineering/research extension, not as
  a benchmark claim.

The same-day evidence is now mixed:

- on `funding_raised_usd / core_only / task2`, `h=60` looked slightly better
  than the `h=30` control and avoided constant collapse,
- on `investors_count / core_only / task2`, `h=60` looked worse than `h=30`
  and did collapse to a constant prediction.

So the most truthful current statement is not "longer horizon is better", but:

> `h>30` is now a real exercised V740 path, and it may help on some continuous
> funding slices, but it is not yet robust across targets.

The next sensible steps are:

1. repeat the `30 vs 60` comparison on `core_edgar`,
2. only then consider whether `h=90` is worth testing,
3. keep all such runs outside the official benchmark until the evidence is
   strong enough to justify a protocol expansion.

## 8. First same-day `h=90` checks

To avoid leaving `h=90` at the purely conceptual stage, two additional
local-only runs were executed on the same day using the same narrow budget
(`12 entities / 1500 train rows / 2 epochs`).

### 8.1 `funding_raised_usd / core_only / task2 / h=90`

Command:

```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 \
  scripts/run_v740_alpha_smoke_slice.py \
  --task task2_forecast \
  --ablation core_only \
  --target funding_raised_usd \
  --horizon 90 \
  --max-entities 12 \
  --max-rows 1500 \
  --max-epochs 2 \
  --output-json docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_funding_h90.json
```

Observed result:

- constant prediction: `true`
- prediction std: `0.0`
- `MAE = 134873.4536`
- `RMSE = 186562.7618`
- wall time: `28.896s`
- training mode: `fallback-only`

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_funding_h90.json`

### 8.2 `investors_count / core_only / task2 / h=90`

Command:

```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 \
  scripts/run_v740_alpha_smoke_slice.py \
  --task task2_forecast \
  --ablation core_only \
  --target investors_count \
  --horizon 90 \
  --max-entities 12 \
  --max-rows 1500 \
  --max-epochs 2 \
  --output-json docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_investors_h90.json
```

Observed result:

- constant prediction: `true`
- prediction std: `0.0`
- `MAE = 23.7500`
- `RMSE = 28.8227`
- wall time: `31.025s`
- training mode: `fallback-only`

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_investors_h90.json`

## 9. `core_edgar` and `full` 30-vs-60 controls

To avoid treating the first `core_only` result as representative, the same
local-only protocol was extended to `core_edgar` and `full`.

### 9.1 `core_edgar / funding_raised_usd`

| Horizon | Constant | MAE | RMSE |
|---|---:|---:|---:|
| `30` | `false` | `179797.74` | `359612.87` |
| `60` | `false` | `178084.08` | `370856.79` |

Artifacts:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_funding_h30.json`
- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_funding_h60.json`

### 9.2 `core_edgar / investors_count`

| Horizon | Constant | MAE | RMSE |
|---|---:|---:|---:|
| `30` | `false` | `56.5055` | `64.2714` |
| `60` | `false` | `57.0610` | `64.7228` |

Artifacts:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_investors_h30.json`
- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_investors_h60.json`

### 9.3 `full / funding_raised_usd`

| Horizon | Constant | MAE | RMSE |
|---|---:|---:|---:|
| `30` | `false` | `179797.75` | `359612.87` |
| `60` | `false` | `178084.08` | `370856.79` |

Artifacts:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_full_funding_h30.json`
- `docs/references/v740_alpha_longh_smoke_20260326/t2_full_funding_h60.json`

### 9.4 `full / investors_count`

| Horizon | Constant | MAE | RMSE |
|---|---:|---:|---:|
| `30` | `false` | `56.4737` | `64.2513` |
| `60` | `false` | `57.0722` | `64.7305` |

Artifacts:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_full_investors_h30.json`
- `docs/references/v740_alpha_longh_smoke_20260326/t2_full_investors_h60.json`

The important empirical point is that `full` is currently almost identical to
`core_edgar` on these narrow long-horizon slices. On the tested local slices,
the present text path is **not** producing a measurable additional benefit.

## 10. Context-scaled `h=90` probes

To test whether the earlier `h=90` failure was simply a context-length issue,
two context-scaled local probes were run at `input_size=120`.

### 10.1 `core_edgar / funding_raised_usd / h=90 / input_size=120`

- constant prediction: `false`
- `MAE = 175230.96`
- `RMSE = 368945.47`

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_funding_h90_ctx120.json`

### 10.2 `core_edgar / investors_count / h=90 / input_size=120`

- constant prediction: `true`
- `MAE = 21.8138`
- `RMSE = 36.5540`

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_investors_h90_ctx120.json`

### 10.3 `core_only / funding_raised_usd / h=90 / input_size=120`

- constant prediction: `true`
- `MAE = 134820.11`
- fallback-only

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_funding_h90_ctx120.json`

This sharpens the earlier interpretation:

- the first viable `h=90` path currently appears only on a funding slice,
- simply increasing context is not enough to rescue every target,
- and count-like long horizons still look weaker than funding-like long
  horizons in the current alpha.

## 11. New entrepreneurial-finance horizon probes: `45` and `180`

The updated horizon-conditioning path was then used to probe two more
startup-financing-motivated horizons.

### 11.1 `core_edgar / funding_raised_usd / h=45 / input_size=90`

- constant prediction: `false`
- `MAE = 230911.99`
- `RMSE = 448447.85`
- `horizon_to_context_ratio = 0.5`

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_funding_h45_ctx90.json`

This is the first direct proof that V740-alpha can now exercise a
non-benchmark entrepreneurial-finance horizon between `30` and `60` under the
new continuous horizon-conditioning path.

### 11.2 `core_edgar / funding_raised_usd / h=180 / input_size=180`

- constant prediction: `true`
- `MAE = 297084.33`
- `RMSE = 505184.47`
- fallback-only
- `horizon_to_context_ratio = 1.0`

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_funding_h180_ctx180.json`

This is useful negative evidence. It shows that simply matching context length
to a half-year horizon is still not enough on a tiny local slice. The current
prototype can represent the horizon, but it does not yet have enough effective
windows or enough long-range skill to make `h=180` viable under this budget.

## 12. A critical caution: the earlier source-density readings were a logging bug

After the first long-horizon sweep, the source-density diagnostics in the
funding/count artifacts were re-audited against the implementation. The root
cause turned out to be a logging bug: `edgar_source_density` and
`text_source_density` had been computed only inside the binary branch of
`V740AlphaPrototypeWrapper.fit()`, so non-binary long-h artifacts falsely
reported `0.0`.

That bug is now fixed in `src/narrative/block3/models/v740_alpha.py`, and two
representative reruns were executed to refresh the evidence:

- `task2_forecast / core_edgar / funding_raised_usd / h=60`
  - `edgar_source_density = 1.0`
  - `MAE = 177810.09`
- `task2_forecast / full / funding_raised_usd / h=60`
  - `edgar_source_density = 1.0`
  - `text_source_density = 1.0`
  - `MAE = 177810.09`

So the correct interpretation is now narrower and more accurate:

- the tested `core_edgar` and `full` long-h slices do contain source-covered
  rows,
- the current `h=60` funding path remains non-constant after the density fix,
- but this still does **not** prove that the source-native sparse event-memory
  path is already the dominant reason for the gain.

What the refreshed artifacts support is:

- horizon/context design is beginning to matter,
- EDGAR/text-covered feature regimes are genuinely present on the tested slices,
- but source-native sparse event-memory attribution still requires dedicated
  ablation beyond these narrow local probes.

## 13. Current bottom line

The current honest bottom line is now:

1. V740-alpha no longer has a hard-coded "small horizon table" limitation.
2. `h=45`, `h=60`, `h=90`, and `h=180` are now real exercised local-only
   prototype paths.
3. `h=60` is promising on funding slices, but not uniformly on count slices.
4. The first viable `h=90` path exists, but only on a funding slice with
   longer context.
5. `h=180` is still fallback-only on the tested narrow slice.
6. Current long-h improvements should be interpreted as evidence about
   horizon/context design on source-covered slices, not yet as proof that
   source-native event memory itself is fully carrying the gain.

### 8.3 What the `h=90` runs imply

These first `h=90` checks are still useful, even though they are not yet good
results. They show that, on the current narrow local slices:

1. the `90` horizon token path is wired correctly end-to-end,
2. but the tested slices do not yet produce enough usable windows for the main
   learning path,
3. so the prototype falls back to a degenerate prediction regime.

This is exactly why longer-horizon work must be treated as a joint design
problem:

- longer forecast horizon,
- sufficient input context,
- sufficient per-entity sequence length,
- and representative local sampling

must all scale together. Simply adding a bigger horizon token is not enough.

## 9. Updated best interpretation

The current best interpretation is now stricter than Section 7 alone:

- `h=60` is a real, partially promising prototype path,
- but its benefit is target-dependent even on the tested core-only slices,
- and `h=90` is not yet ready for meaningful engineering comparison on the
  current narrow local setup because it falls back to constant predictions.

So the most truthful current statement is:

> V740-alpha has now exercised `h = 60` and `h = 90` locally. `h = 60` shows
> mixed but potentially useful behavior, while `h = 90` currently exposes a
> window/context limitation rather than a mature long-horizon capability.

## 10. Same-day `core_edgar` checks

To test whether the first `core_only` signals were merely artifacts of the
endogenous-only setup, the same local protocol was also applied to
`task2_forecast / core_edgar` on the same day.

Observed results:

| Target | Horizon | Constant | MAE | RMSE | Wall time |
|---|---:|---|---:|---:|---:|
| `funding_raised_usd` | `30` | `false` | `179797.7425` | `359612.8705` | `51.257s` |
| `funding_raised_usd` | `60` | `false` | `178084.0833` | `370856.7862` | `37.888s` |
| `investors_count` | `30` | `false` | `56.5055` | `64.2714` | `49.867s` |
| `investors_count` | `60` | `false` | `57.0610` | `64.7228` | `48.516s` |

Artifacts:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_funding_h30.json`
- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_funding_h60.json`
- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_investors_h30.json`
- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_investors_h60.json`

These checks matter for two reasons:

## 14. Larger-slice `full / funding_raised_usd / h=90` result

After the earlier narrow `h=90` probes and the context-scaled `core_edgar`
funding success, the next useful question was whether a richer `full` slice
could also run cleanly at `h=90` when given more entities and more rows.

That larger local-only SLURM audit has now completed with:

- task: `task2_forecast`
- ablation: `full`
- target: `funding_raised_usd`
- horizon: `90`
- input size: `120`
- train rows: `3000`
- val rows: `2016`
- test rows: `1944`
- train matrix rows: `1080`
- test matrix rows: `1944`
- feature count: `169`
- selected entities: `24`
- `edgar_source_density = 0.9997`
- `text_source_density = 1.0`
- constant prediction: `false`
- prediction std: `5611.63`
- `MAE = 397737.1872`
- `RMSE = 993272.4963`
- wall time: `35.581s`

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_full_funding_h90_ctx120_rows3000.json`

This result is valuable, but only in a narrow sense:

1. it confirms that a larger, source-covered `full / h=90` path now executes
   cleanly and non-degenerately;
2. it does **not** yet show that text improves long-horizon performance;
3. it should not be directly compared with the earlier smaller-slice `h=90`
   probes without noting the slice-size change.

The right interpretation is therefore:

> V740-alpha now has a bigger `full / funding / h=90` path that is stable
> enough to study. That is an engineering milestone, not yet a performance
> victory claim.

1. unlike the `core_only / investors_count / h=60` slice, the `core_edgar`
   count slice does **not** collapse to a constant prediction,
2. but the target-dependent story remains: `h=60` is slightly better on the
   tested funding slice and slightly worse on the tested count slice.

So EDGAR currently looks more like a **stabilizer** of the longer-horizon path
than a universal long-horizon booster.

## 11. Same-day `full` checks

To test whether adding the current text pathway changes the local long-horizon
picture, the same protocol was also applied to `task2_forecast / full`.

Observed results:

| Target | Horizon | Constant | MAE | RMSE | Wall time |
|---|---:|---|---:|---:|---:|
| `funding_raised_usd` | `30` | `false` | `179797.7506` | `359612.8716` | `44.630s` |
| `funding_raised_usd` | `60` | `false` | `178084.0833` | `370856.7862` | `39.146s` |
| `investors_count` | `30` | `false` | `56.4737` | `64.2513` | `54.674s` |
| `investors_count` | `60` | `false` | `57.0722` | `64.7305` | `51.252s` |

Artifacts:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_full_funding_h30.json`
- `docs/references/v740_alpha_longh_smoke_20260326/t2_full_funding_h60.json`
- `docs/references/v740_alpha_longh_smoke_20260326/t2_full_investors_h30.json`
- `docs/references/v740_alpha_longh_smoke_20260326/t2_full_investors_h60.json`

The important practical observation is that these `full` results are nearly
identical to the matching `core_edgar` results on the same narrow local slices.
On the currently tested local regime, the text pathway is therefore **not
showing a measurable incremental long-horizon benefit**.

## 12. Updated best interpretation

After the `core_only`, `core_edgar`, and `full` checks together, the strongest
truthful
statement is now:

> V740-alpha's local `h=60` path is real and increasingly credible on the
> tested narrow slices, especially for funding targets. However, its benefit is
> still target-dependent, and `h=90` remains immature because the current
> window/context regime is too weak to support it reliably.

That is a more defensible conclusion than either of the two naive extremes:

- "`h>30` is always better", or
- "`h>30` is pointless."

At this stage, the right engineering target is:

1. keep `h=60` under active local-only development,
2. expand it across a few more representative slices,
3. treat `h=90` as a context/window-design problem before treating it as a
   forecasting benchmark problem.

On the currently tested long-horizon local slices, the clearest structural
pattern is:

- `EDGAR` can stabilize some longer-horizon behavior,
- `text` currently does not add a clear incremental gain,
- and `funding_raised_usd` remains the most promising target for pushing the
  first robust `h>30` capability.

## 13. Same-day `h=90` context-scaling probes

To test whether the `h=90` failure mode was mainly caused by the default short
context, two additional narrow local-only probes were run with:

- `input_size = 120`
- `max_rows = 2000`

### 13.1 `core_edgar / funding_raised_usd / h=90 / input_size=120`

Observed result:

- constant prediction: `false`
- prediction std: `9596.17`
- `MAE = 175230.9597`
- `RMSE = 368945.4672`
- wall time: `37.864s`

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_edgar_funding_h90_ctx120.json`

### 13.2 `core_only / funding_raised_usd / h=90 / input_size=120`

Observed result:

- constant prediction: `true`
- prediction std: `0.0`
- `MAE = 134820.1141`
- `RMSE = 186435.1094`
- wall time: `27.23s`
- training mode: `fallback-only`

Artifact:

- `docs/references/v740_alpha_longh_smoke_20260326/t2_core_only_funding_h90_ctx120.json`

### 13.3 What this isolates

These two probes are especially informative because they separate two possible
stories:

1. "`h=90` just needs a longer context", and
2. "`h=90` needs a longer context plus a richer source regime."

The evidence now supports the second interpretation more strongly.

- Longer context alone is **not** enough on the tested `core_only` slice.
- Longer context **plus** EDGAR produces a non-constant learned path on the
  tested funding slice.

So the current best engineering interpretation is:

> the first viable `h=90` path in V740-alpha seems to require both more usable
> context and richer source support. This is not yet a general long-horizon
> capability; it is an early sign of where longer-horizon viability may come
> from.
