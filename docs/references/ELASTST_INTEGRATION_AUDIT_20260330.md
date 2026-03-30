# ElasTST Integration Audit (2026-03-30)

This note records the first real Block 3 integration pass for `ElasTST`.
It is intentionally narrow and factual. The goal is to decide whether
`ElasTST` is only a literature reference or a genuine next-wave comparator
worth continuing through local-clear and then narrow benchmark-clear.

## 1. Official source path audited

- official repo path:
  - `https://github.com/microsoft/ProbTS`
- local audited repo:
  - `~/.cache/block3_optional_repos/ProbTS`
- audited HEAD during this pass:
  - `6975a97669955b5debcfa71007244b8ed7f50161`

Verified relevant files:

- `config/multi_hor/elastst.yaml`
- `scripts/run_elastst.sh`
- `probts/model/forecaster/point_forecaster/elastst.py`
- `probts/model/nn/arch/ElasTSTModule/ElasTST_backbone.py`

## 2. What mattered technically

The official implementation does exist inside ProbTS, but importing the full
`probts` package surface is not the right first integration path for Block 3.

The first blocker sequence was:

1. package-level import pulled in wider ProbTS data-manager code we do not
   need for local clear work
2. the full import path initially surfaced a `GLIBCXX_3.4.31` issue
3. after preloading the insider env `libstdc++`, the remaining problem was not
   missing code but an integration mismatch around the expected ElasTST input
   shape

The key correction from direct execution was:

> the official `ElasTST_backbone` treats the input as a **single-channel
> time-by-variable image**, not as a multi-channel covariate tensor.

So the wrapper must instantiate the vendor backbone with:

- `in_channels = 1`

and allow the variable axis to remain the patched dimension.

## 3. Local integration path used

Block 3 now uses a source-level vendor import path rather than full-package
ProbTS import for this model.

New local implementation:

- `src/narrative/block3/models/elastst_model.py`

Supporting runtime helper:

- `src/narrative/block3/models/optional_runtime.py`

Registry exposure:

- `src/narrative/block3/models/deep_models.py`

This path deliberately keeps the first pass lightweight:

1. vendor repo audit
2. import path audit
3. tiny real-data smokes
4. only then decide whether narrow benchmark-clear is worth paying for

## 4. Real smoke results obtained today

### 4.1 Funding slice: first non-fallback success

Artifact:

- `docs/references/local_model_smoke_20260330/elastst_t2_core_edgar_funding_h30.json`

Slice:

- model: `ElasTST`
- task: `task2_forecast`
- ablation: `core_edgar`
- target: `funding_raised_usd`
- horizon: `30`
- max entities: `4`
- max rows: `300`
- wrapper config:
  - `input_size=60`
  - `l_patch_size=8_16`
  - `hidden_size=64`
  - `d_inner=128`
  - `n_heads=4`

Result:

- `fit_seconds = 2.02`
- `predict_seconds = 0.020`
- `prediction_std = 10801.88`
- `constant_prediction = false`
- `MAE = 445367.47`
- `RMSE = 639269.97`

This is the first meaningful proof that the Block 3 wrapper is not just
importable; it can actually train and predict on a freeze-backed real slice.

### 4.2 Investors slice: default context was too long on the tiny slice

Artifact:

- `docs/references/local_model_smoke_20260330/elastst_t2_core_edgar_investors_h14.json`

Slice:

- task: `task2_forecast`
- ablation: `core_edgar`
- target: `investors_count`
- horizon: `14`
- config:
  - `input_size=60`
  - `l_patch_size=8_16`

Result:

- `No training windows`
- fallback-only
- `constant_prediction = true`
- `MAE = 11.25`

This is not a model crash. It means the tiny audited slice plus the longer
context did not leave usable windows for this target/horizon combination.

### 4.3 Investors slice: shorter context restored a real training path

Artifact:

- `docs/references/local_model_smoke_20260330/elastst_t2_core_edgar_investors_h14_ctx30.json`

Adjusted config:

- `input_size=30`
- `l_patch_size=6_10`

Result:

- `fit_seconds = 2.30`
- `predict_seconds = 0.048`
- `prediction_std = 1.90e-4`
- `constant_prediction = false`
- `MAE = 47.52`
- `RMSE = 48.95`

This tells us something useful and specific:

> on tiny Block 3 count slices, `ElasTST` is context-sensitive and should not
> be evaluated only with the longer default varied-horizon context.

## 5. Current honest status

As of this pass, the correct label for `ElasTST` is:

- **locally integrated**
- **real-data smoke passed on funding**
- **real-data smoke passed on investors after context shortening**
- **first narrow benchmark-clear completed successfully**
- **not yet canonical benchmark-landed**

### 5.1 First narrow benchmark-clear

Job:

- `5298399` `v740_elas_clr`

Output root:

- `runs/benchmarks/block3_phase9_localclear_20260330/elastst_funding_h30/`

Result:

- task: `task2_forecast`
- ablation: `core_edgar`
- target: `funding_raised_usd`
- horizon: `30`
- `MAE = 201925.6990`
- `RMSE = 205733.8504`
- `prediction_coverage_ratio = 1.0`
- `fairness_pass = true`
- `peak_rss_gb = 47.06`
- `train_time_seconds = 4.02`
- `inference_time_seconds = 0.012`

This means `ElasTST` is now past the “tiny smoke only” stage.

### 5.2 Second narrow benchmark-clear: investors-count remains a red flag

Job:

- `5298437` `v740_elas_inv_clr`

Output root:

- `runs/benchmarks/block3_phase9_localclear_20260330/elastst_investors_h14_ctx30/`

Configuration:

- `task2_forecast`
- `core_edgar`
- `investors_count`
- `h=14`
- `input_size=30`
- `l_patch_size=6_10`

Result:

- `MAE = 125.1687`
- `RMSE = 125.1687`
- `prediction_coverage_ratio = 1.0`
- `fairness_pass = false`
- `peak_rss_gb = 46.24`
- `train_time_seconds = 3.36`
- `inference_time_seconds = 0.014`

This is a useful negative result. It means the shorter-context path is enough
to make `ElasTST` executable on the count slice, but **not** enough to make it
promotable. At the moment, the correct label is:

- funding narrow clear: **clean**
- investors narrow clear: **runs, but fails fairness**

## 6. Practical next step

The most defensible next step is:

1. keep `ElasTST` in the local-clear lane
2. treat the count-side fairness failure as a tuning problem, not as proof of
   general readiness
3. widen the local-clear matrix beyond the first funding slice
4. only then decide whether it deserves:
   - a broader local-clear matrix, or
   - a more formal comparator expansion

The key engineering lesson from the first pass is now explicit:

> `ElasTST` is not blocked by missing source code anymore. The main tuning
> issue is now **context / patch-size compatibility on Block 3 audited slices**.
