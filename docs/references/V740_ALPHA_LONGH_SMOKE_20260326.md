# V740 Alpha Long-Horizon Local Smoke (2026-03-26)

> Date: 2026-03-26
> Scope: local-only prototype expansion beyond the current benchmark horizon grid
> Status: first real `h>30` V740-alpha evidence
>
> This note does **not** change the current clean benchmark protocol.
> The official comparable benchmark surface remains `h in {1, 7, 14, 30}`.
> The purpose of this note is to record the first verified local-only V740-alpha
> run at a longer forecast horizon after explicitly wiring `60/90` support into
> the prototype's horizon conditioning path.

## 1. Why this exists

As of 2026-03-26, the current benchmark and the earlier V740 local tools only
produced directly comparable results on `h = 1, 7, 14, 30`. The project config
already contained longer context and `k` values, but that was not the same as
having verified V740 behavior at forecast horizons above 30.

This local-only experiment exists to answer the narrower engineering question:

> Can the current V740-alpha prototype run correctly at `h = 60` without
> collapsing, and if so, what does the first real-data result look like?

## 2. Implementation change required

Before this run, `V740AlphaPrototypeWrapper` only mapped horizon condition
tokens for `{1, 7, 14, 30}`.

On 2026-03-26 the local prototype was extended to support:

- `60`
- `90`

at the horizon-token level in `src/narrative/block3/models/v740_alpha.py`, and
the local smoke runner was updated to allow those horizons in
`scripts/run_v740_alpha_smoke_slice.py`.

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
