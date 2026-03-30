# Local Model Smoke Notes (2026-03-30)

This note records the first round of generic local smokes run through
`scripts/run_block3_local_model_smoke.py`. These are **not** canonical
benchmark results. They are small, freeze-backed audited slices used to decide
whether a newly integrated model is healthy enough to justify more expensive
benchmark work.

Supporting implementation added in this round:

- generic runner:
  - `scripts/run_block3_local_model_smoke.py`
- optional dependency bootstrap:
  - `scripts/bootstrap_optional_model_deps.py`
- runtime helper for local vendor path / `libstdc++` preload:
  - `src/narrative/block3/models/optional_runtime.py`

## 1. SAMformer

### Command status

- local generic smoke: **passed**
- output artifact:
  - `docs/references/local_model_smoke_20260328/samformer_t1_core_edgar_is_funded_h14.json`

### Slice

- model: `SAMformer`
- task: `task1_outcome`
- ablation: `core_edgar`
- target: `is_funded`
- horizon: `14`
- max entities: `4`
- max rows: `300`

### Result

- `fit_seconds = 11.6`
- `predict_seconds = 0.002`
- `prediction_std = 0.1254`
- `constant_prediction = false`
- `MAE = 0.3863`
- `RMSE = 0.5688`

### Narrow benchmark-clear update

- job: `5294242` `v740_samf_clr`
- status: **COMPLETED**
- output root:
  - `runs/benchmarks/block3_phase9_localclear_20260328/samformer/`

Benchmark-harness metrics on the narrow audited slice:

- `MAE = 0.3485`
- `RMSE = 0.3893`
- `prediction_coverage_ratio = 1.0`
- `fairness_pass = true`
- `peak_rss_gb = 49.16`

### Interpretation

SAMformer is now past the “wrapper exists” stage and past the “generic local
smoke only” stage. It has a first **real benchmark-harness narrow clear** on a
freeze-backed audited slice. It is still **not** a canonical benchmark entrant
yet, but it now has a credible path to the next integration step.

## 2. Prophet

### Current factual status

- local wrapper: **exists**
  - `src/narrative/block3/models/statistical.py`
- local vendor install path:
  - `~/.cache/block3_optional_pydeps/py312`
- bootstrap route:
  - `scripts/bootstrap_optional_model_deps.py --group prophet`
- local generic smoke: **passed**
  - output artifact:
    - `docs/references/local_model_smoke_20260328/prophet_t2_core_only_funding_h30.json`

### Slice

- model: `Prophet`
- task: `task2_forecast`
- ablation: `core_only`
- target: `funding_raised_usd`
- horizon: `30`
- max entities: `4`
- max rows: `300`

### Result

- `fit_seconds = 5.24`
- `predict_seconds = 0.209`
- `prediction_std = 183038.48`
- `constant_prediction = false`
- `MAE = 5300.86`
- `RMSE = 7245.31`

### Narrow benchmark-clear update

- first SLURM probe: `5294243` `v740_prop_clr`
- status: **FAILED**
- real root cause:
  - `--preset quick` only allows horizons `[7, 14]`
  - the job requested `--horizons-filter 30`
  - harness aborted before model execution with:
    `No horizons left after applying horizons filter: [30]`
- corrected follow-up:
  - `5294254` `v740_prop_std`
  - `bigmem`, `4 CPU`, `64G`, `2h`
  - status: **COMPLETED**
  - output root:
    `runs/benchmarks/block3_phase9_localclear_20260328/prophet_standard_h30/`

Benchmark-harness metrics on the narrow audited slice:

- `MAE = 1599245.3772`
- `RMSE = 2134851.9455`
- `prediction_coverage_ratio = 1.0`
- `fairness_pass = true`
- `peak_rss_gb = 47.81`

### Interpretation

Prophet is no longer blocked by wrapper absence or dependency availability.
The current practical path is a **repo/user-local vendor install**, not
mutating the shared insider env. The first narrow benchmark-clear failed for a
submission-surface reason, not a model/runtime reason, and the corrected
CPU-only resubmission has now completed cleanly. The conclusion is therefore
clear: Prophet is benchmark-harness-compatible on a narrow audited slice, but
its performance on this specific Block 3 funding task is weak.

## 3. TabPFN / TabPFN-TS

### Current factual status

- local generic wrappers: **already exist**
  - `src/narrative/block3/models/traditional_ml.py`
- `GLIBCXX_3.4.31` blocker: **resolved**
  - preloading the insider env `libstdc++.so.6.0.34` fixes the runtime loader
- Hugging Face access: **working**
  - authenticated account resolves both `Prior-Labs/tabpfn_2_5` and
    `Prior-Labs/tabpfn_2_6`
- official latest checkpoint line: **2.6**
  - classifier:
    `tabpfn-v2.6-classifier-v2.6_default.ckpt`
  - regressor:
    `tabpfn-v2.6-regressor-v2.6_default.ckpt`
- shared env package was too old:
  - shared `insider` runtime had `tabpfn 6.3.2`
  - that runtime only knew `V2` / `V2.5`
  - trying to load 2.6 checkpoints there produced `KeyError: 'tabpfn_v2_6'`
- latest-source runtime is now installed locally:
  - vendor path:
    `~/.cache/block3_optional_pydeps/py312_tabpfn_latest`
  - installed version:
    `tabpfn 7.0.1`
- current next step:
  - the first narrow benchmark-clear probe `5294255` `v740_tpfn26c` has now
    **completed successfully**
  - it used the local 2.6 classifier checkpoint plus the latest-source vendor
    runtime

Benchmark-harness metrics on the narrow audited slice:

- `MAE = 0.1611`
- `RMSE = 0.2621`
- `prediction_coverage_ratio = 1.0`
- `fairness_pass = true`
- `peak_rss_gb = 47.84`

### Interpretation

TabPFN is no longer blocked by runtime/toolchain compatibility, and it is no
longer blocked by Hugging Face authentication. The important correction is
that the current target should be **TabPFN 2.6**, not the older 2.5 default.
That proof has now been obtained: the latest-source vendor runtime plus the
cached 2.6 checkpoint clear a real narrow benchmark slice cleanly. The
remaining question is no longer “can it run?”, but whether it deserves wider
canonical benchmark expansion.

## 4. LightGTS

### Current factual status

- local wrapper scaffold: **exists**
  - `src/narrative/block3/models/lightgts_model.py`
- official repo audit: **done**
  - `docs/references/LIGHTGTS_INTEGRATION_AUDIT_20260328.md`
- vendor import + minimal instantiation smoke: **passed**
- first real-data tiny smoke: **completed twice**
  - artifacts:
    - `docs/references/local_model_smoke_20260328/lightgts_t2_core_edgar_funding_h30.json`
    - `docs/references/local_model_smoke_20260328/lightgts_t2_core_edgar_funding_h30_ctx60.json`

### Slice

- model: `LightGTS`
- task: `task2_forecast`
- ablation: `core_edgar`
- target: `funding_raised_usd`
- horizon: `30`
- max entities: `4`
- max rows: `300`

### Result

#### First attempt: default-ish longer context

- `input_size = 96`
- `patch_len = 24`
- `stride = 24`
- outcome:
  - `No training windows, using fallback-only mode`
  - `constant_prediction = true`
  - `MAE = 439131.00`

#### Second attempt: smaller context matched to the tiny slice

- `input_size = 60`
- `patch_len = 15`
- `stride = 15`
- result:
  - `fit_seconds = 4.60`
  - `predict_seconds = 0.018`
  - `prediction_std = 10802.78`
  - `constant_prediction = false`
  - `MAE = 445367.99`
  - `RMSE = 639270.56`

### Interpretation

LightGTS has now crossed the most important first gate:

- it is no longer just an audited vendor repo,
- it is no longer just an import/instantiation smoke,
- and it is no longer fallback-only by default.

The main lesson from this first real-data pass is precise:

> on tiny Block 3 audited slices, `LightGTS` needs a shorter context than the
> repo-style default before it even has valid training windows.

So the next correct step is **not** “register it into the canonical benchmark
right now”. The correct next step is:

1. keep the current wrapper out of the canonical leaderboard,
2. register it only as a local-clear entrant,
3. attempt a narrow benchmark-clear job.

That next step has now completed in two stages:

### First narrow benchmark-clear attempt

- job: `5298059` `v740_lgts_clr`
- outcome: **not a valid clear**
- real blocker:
  - the compute node could not see the vendor repo because the job exported
    `BLOCK3_LIGHTGTS_REPO=/tmp/LightGTS`
  - the harness exited with:
    `LightGTS official repo not found`

### Repaired narrow benchmark-clear

- repaired job: `5298289` `v740_lgts_clr`
- repair:
  - moved the official repo to the persistent user-local path
    `~/.cache/block3_optional_repos/LightGTS`
- scope:
  - `task2_forecast / core_edgar / funding_raised_usd / h=30`
- output root:
  - `runs/benchmarks/block3_phase9_localclear_20260329/lightgts_funding_h30/`
- result:
  - `MAE = 201930.5506`
  - `RMSE = 205737.7476`
  - `prediction_coverage_ratio = 1.0`
  - `fairness_pass = true`
  - `peak_rss_gb = 47.40`

### Interpretation update

LightGTS is now past all of these gates:

- vendor repo audit
- import/instantiation smoke
- non-fallback tiny real-data smoke
- first real narrow benchmark-clear

That still does **not** make it a canonical benchmark entrant yet. It does mean
that the next honest question is no longer “can it run?” but “does it deserve a
wider local-clear matrix or a true canonical expansion attempt?”.

## 5. OLinear

### Current factual status

- local wrapper: **exists**
  - `src/narrative/block3/models/olinear_model.py`
- vendor/runtime helper:
  - `src/narrative/block3/models/optional_runtime.py`
- official repo audit: **done**
  - persistent local repo:
    `~/.cache/block3_optional_repos/OLinear`
  - audited HEAD:
    `f168e01a3e0e316ad98330b5e77afed1f77b0af5`
- local generic smokes: **completed**
  - artifacts:
    - `docs/references/local_model_smoke_20260330/olinear_t2_core_edgar_funding_h30.json`
    - `docs/references/local_model_smoke_20260330/olinear_t2_core_edgar_investors_h14.json`

### Funding slice

- model: `OLinear`
- task: `task2_forecast`
- ablation: `core_edgar`
- target: `funding_raised_usd`
- horizon: `30`
- max entities: `4`
- max rows: `300`

#### Result

- `fit_seconds = 15.35`
- `predict_seconds = 0.0869`
- `prediction_std = 300966.34`
- `constant_prediction = false`
- `MAE = 265368.00`
- `RMSE = 528267.79`

### Investors-count slice

- model: `OLinear`
- task: `task2_forecast`
- ablation: `core_edgar`
- target: `investors_count`
- horizon: `14`
- max entities: `4`
- max rows: `300`

#### Result

- wrapper logs:
  - `No training windows, using fallback-only mode`
- `fit_seconds = 0.044`
- `predict_seconds = 0.00003`
- `prediction_std = 0.0`
- `constant_prediction = true`
- `MAE = 11.25`
- `RMSE = 14.64`

### Narrow benchmark-clear update

- job: `5298296` `v740_olnr_clr`
- status: **COMPLETED**
- output root:
  - `runs/benchmarks/block3_phase9_localclear_20260330/olinear_funding_h30/`

Benchmark-harness metrics on the audited slice:

- `MAE = 131288.8062`
- `RMSE = 174092.4956`
- `prediction_coverage_ratio = 1.0`
- `fairness_pass = true`
- `peak_rss_gb = 47.49`
- `train_time_seconds = 3.01`
- `inference_time_seconds = 0.0050`

### Interpretation

OLinear is no longer in the old “docs-only / artifact-blocked” state.
It now has all of the following:

- an audited official repo path,
- a Block 3 wrapper,
- a real non-fallback funding smoke,
- and a first audited narrow benchmark-clear on the real harness.

The honest remaining limitation is also clear:

- the current count-side evidence is weak,
- the tiny `investors_count` slice fell back to a constant predictor,
- so OLinear should be treated as a **promising local-clear entrant**, not yet
  as a broadly proven canonical comparator.

## 6. Narrow benchmark-clear status

As of 2026-03-30, four local-only narrow benchmark-clear lines have completed
through the real benchmark harness with isolated output roots:

- `5294242` `v740_samf_clr`
  - `SAMformer`
  - `task1_outcome / core_edgar / is_funded / h=14`
  - **completed successfully**
- `5294254` `v740_prop_std`
  - `Prophet`
  - `task2_forecast / core_only / funding_raised_usd / h=30`
  - **completed successfully**
- `5294255` `v740_tpfn26c`
  - `TabPFNClassifier`
  - `task1_outcome / core_edgar / is_funded / h=14`
  - **completed successfully**
- `5298289` `v740_lgts_clr`
  - `LightGTS`
  - `task2_forecast / core_edgar / funding_raised_usd / h=30`
  - **completed successfully** after repairing the vendor repo path
- `5298296` `v740_olnr_clr`
  - `OLinear`
  - `task2_forecast / core_edgar / funding_raised_usd / h=30`
  - **completed successfully**

These are resumable SLURM jobs under
`runs/benchmarks/block3_phase9_localclear_20260328/`. They are intended to
clear the models against the real benchmark harness without polluting the
canonical leaderboard.

## 7. TabPFNRegressor 2.6 Follow-up Probes

The next two highest-value follow-up probes have now both **completed** so that
`TabPFN 2.6` is no longer judged only from a single binary slice.

### Funding narrow clear

- job: `5294259` `v740_tpfn26r_fu`
- model: `TabPFNRegressor`
- slice:
  - `task2_forecast`
  - `core_edgar`
  - `funding_raised_usd`
  - `h=30`
- output root:
  - `runs/benchmarks/block3_phase9_localclear_20260328/tabpfn2_6_regressor_funding/`

#### Result

- `MAE = 1389577.1089`
- `RMSE = 1827933.2039`
- `prediction_coverage_ratio = 1.0`
- `fairness_pass = true`
- `peak_rss_gb = 46.66`
- `train_time_seconds = 1.68`
- `inference_time_seconds = 0.66`

#### Interpretation

This funding probe proves that the latest-source `TabPFN 2.6` regressor path is
not limited to the earlier binary clear. The harness path is clean and fully
covered on a real audited funding slice, but the quality on this slice is weak.
So the current conclusion is not “promote to canonical now”, but rather:
`TabPFNRegressor` is **runnable and auditable**, yet still needs more evidence
before wider benchmark expansion.

### Investors-count narrow clear

- job: `5294260` `v740_tpfn26r_inv`
- model: `TabPFNRegressor`
- slice:
  - `task2_forecast`
  - `core_edgar`
  - `investors_count`
  - `h=14`
- output root:
  - `runs/benchmarks/block3_phase9_localclear_20260328/tabpfn2_6_regressor_investors/`

#### Result

- `MAE = 0.0`
- `RMSE = 0.0`
- `prediction_coverage_ratio = 1.0`
- `fairness_pass = false`
- `peak_rss_gb = 48.30`
- `train_time_seconds = 1.64`
- `inference_time_seconds = 0.002`

#### Interpretation

This probe is a **red flag, not a success story**. The numerical score is
perfect, but the run fails the fairness gate. It therefore cannot be promoted
as a valid benchmark-clear result, and it should be treated as evidence that
`TabPFNRegressor` still needs closer auditing on count-style targets before any
canonical expansion is considered.

## 8. ElasTST

### Current factual status

- official ProbTS repo audited locally:
  - `~/.cache/block3_optional_repos/ProbTS`
- dedicated local wrapper now exists:
  - `src/narrative/block3/models/elastst_model.py`
- integration audit:
  - `docs/references/ELASTST_INTEGRATION_AUDIT_20260330.md`

### Funding slice: first non-fallback smoke

- output artifact:
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

### Investors slice: default context too long on the tiny slice

- output artifact:
  - `docs/references/local_model_smoke_20260330/elastst_t2_core_edgar_investors_h14.json`

Result:

- `No training windows`
- fallback-only
- `constant_prediction = true`
- `MAE = 11.25`

### Investors slice: shorter context restores a real path

- output artifact:
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

### Interpretation

ElasTST is now beyond the “paper only” stage:

- the local wrapper is real,
- funding and shorter-context investors slices both execute end-to-end,
- and the remaining question is no longer “can it run?” but
  “which context / patch settings make it worth paying for narrow clear?”

So the honest next step is:

1. keep it in local-clear status,
2. start with a funding narrow clear first,
3. widen only if the first clear is healthy.

That first narrow clear has now already completed successfully:

- job: `5298399` `v740_elas_clr`
- output root:
  - `runs/benchmarks/block3_phase9_localclear_20260330/elastst_funding_h30/`
- audited harness result:
  - `MAE = 201925.6990`
  - `RMSE = 205733.8504`
  - `prediction_coverage_ratio = 1.0`
  - `fairness_pass = true`
  - `peak_rss_gb = 47.06`

So `ElasTST` has now crossed the same first serious local-clear gate that
`LightGTS` and `OLinear` already crossed earlier.

That said, the second narrow clear is now also real and is **not** clean:

- job: `5298437` `v740_elas_inv_clr`
- output root:
  - `runs/benchmarks/block3_phase9_localclear_20260330/elastst_investors_h14_ctx30/`
- audited harness result:
  - `MAE = 125.1687`
  - `RMSE = 125.1687`
  - `prediction_coverage_ratio = 1.0`
  - `fairness_pass = false`
  - `peak_rss_gb = 46.24`

So the current honest summary for `ElasTST` is:

- funding side: clean narrow clear exists
- count side: executable, but not yet promotable

## 9. V740-alpha selective / DistDF scaffold v2

The latest V740-only sanity pass stayed deliberately lightweight and auditable.

- output artifact:
  - `docs/references/v740_alpha_selective_smoke_20260330/t2_core_edgar_funding_h30_v2.json`
- slice:
  - `task2_forecast`
  - `core_edgar`
  - `funding_raised_usd`
  - `h=30`

Second-pass implementation changes:

- selective-learning weights now use window-level source activity from the
  EDGAR/text recent + bucket memories instead of only global source density
- the non-binary DistDF-style auxiliary now also aligns:
  - horizon endpoint
  - step-difference shape

Result:

- `constant_prediction = false`
- `prediction_std = 3508.63`
- `edgar_source_density = 1.0`
- `MAE = 441156.71`
- `RMSE = 634565.63`

Compared with the earlier first-pass scaffold (`MAE = 441572.68`), this is a
small but real improvement. The stronger claim is still the engineering one:
the second scaffold runs cleanly and does not destabilize the current V740
prototype.
