# OLinear Integration Audit

> Last updated: 2026-03-30
> Scope: factual integration audit for the official `OLinear` repo on Block 3.

## 1. Official Source

- official repo:
  - `https://github.com/jackyue1994/OLinear`
- persistent local vendor path:
  - `~/.cache/block3_optional_repos/OLinear`
- audited local HEAD:
  - `f168e01a3e0e316ad98330b5e77afed1f77b0af5`

## 2. What Was Previously Blocking Integration

`OLinear` was previously tracked as a high-value missing efficient entrant, but
it was not benchmark-ready because the model depends on dataset-specific
artifacts:

- `Q_mat`
- `Q_out_mat`
- and optionally channel-correlation artifacts for heavier variants

That meant the old status was not “missing code”, but “missing a Block 3
artifact-generation path”.

## 3. What Exists Now

### Public code surface

- runtime helper:
  - `src/narrative/block3/models/optional_runtime.py`
- local wrapper:
  - `src/narrative/block3/models/olinear_model.py`
- local registration path:
  - `src/narrative/block3/models/deep_models.py`

### Current wrapper strategy

The current Block 3 wrapper does **not** attempt to replicate every OLinear
variant. It starts with the base `OLinear` path and generates the necessary
time-domain correlation artifacts from real Block 3 training windows.

This is the current pragmatic integration choice because it keeps the first
clear path:

- lightweight,
- auditable,
- and compatible with the existing local smoke / narrow-clear workflow.

## 4. Real Local Smoke Evidence

### Funding slice

- artifact:
  - `docs/references/local_model_smoke_20260330/olinear_t2_core_edgar_funding_h30.json`
- slice:
  - `task2_forecast / core_edgar / funding_raised_usd / h=30`
- result:
  - `fit_seconds = 15.35`
  - `prediction_std = 300966.34`
  - `constant_prediction = false`
  - `MAE = 265368.00`

### Investors-count slice

- artifact:
  - `docs/references/local_model_smoke_20260330/olinear_t2_core_edgar_investors_h14.json`
- slice:
  - `task2_forecast / core_edgar / investors_count / h=14`
- result:
  - fallback-only on that tiny slice
  - `constant_prediction = true`
  - `MAE = 11.25`

## 5. First Narrow Benchmark-Clear

- job:
  - `5298296` `v740_olnr_clr`
- output root:
  - `runs/benchmarks/block3_phase9_localclear_20260330/olinear_funding_h30/`
- status:
  - **COMPLETED**

### Narrow-clear metrics

- model: `OLinear`
- slice:
  - `task2_forecast / core_edgar / funding_raised_usd / h=30`
- `MAE = 131288.8062`
- `RMSE = 174092.4956`
- `prediction_coverage_ratio = 1.0`
- `fairness_pass = true`
- `peak_rss_gb = 47.49`
- `train_time_seconds = 3.01`
- `inference_time_seconds = 0.0050`

## 6. Current Honest Status

`OLinear` is now **past** all of these old states:

- paper-only
- repo-only
- blocked-only-on-artifact-generation
- wrapper-scaffold-only

It has now reached:

- official repo audited,
- Block 3 wrapper implemented,
- tiny real-data funding smoke,
- first non-fallback funding smoke,
- first audited narrow benchmark-clear.

## 7. What Is Still Not Proven

This does **not** yet prove that `OLinear` should be promoted into the canonical
benchmark lane immediately.

The biggest remaining gap is:

- current count-side evidence is weak,
- the tiny `investors_count` slice fell back to a constant predictor,
- so we do not yet have a broad enough local-clear matrix to claim it is ready
  for canonical expansion.

## 8. Recommended Next Step

The next honest move is:

1. widen the local-clear matrix first,
2. especially across:
   - `funding_raised_usd`
   - `investors_count`
   - `core_only`
   - `core_edgar`
3. only then decide whether `OLinear` deserves a true canonical benchmark
   promotion attempt.
