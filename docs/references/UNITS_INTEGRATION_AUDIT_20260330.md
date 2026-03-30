# UniTS Integration Audit (2026-03-30)

This note records the first real Block 3 integration pass for `UniTS`.
It is intentionally narrow and factual. The goal is to decide whether `UniTS`
should stay a literature reference or move into the same local-clear lane now
used by `LightGTS`, `OLinear`, and `ElasTST`.

## 1. Official source path audited

- official repo path:
  - `https://github.com/mims-harvard/UniTS`
- local audited repo:
  - `~/.cache/block3_optional_repos/UniTS`
- audited HEAD during this pass:
  - `0e0281482864017cac8832b2651906ff5375a34e`

Verified relevant files:

- `README.md`
- `Tutorial.md`
- `run.py`
- `models/UniTS.py`
- `models/UniTS_zeroshot.py`
- `data_provider/multi_task.yaml`
- `data_provider/zeroshot_task.yaml`

## 2. What mattered technically

The official repo is a genuine unified multi-task foundation-style project, but
Block 3 does not need that whole surface for the first integration step.

The key engineering decision from direct inspection is:

> start with a **forecasting-only wrapper** around the official `models/UniTS.py`
> backbone rather than trying to reproduce the repo’s full multi-task training
> scripts on day one.

That choice is justified because:

1. our immediate comparator need is forecasting on Block 3, not anomaly
   detection / imputation / classification,
2. the official `Model` class already supports forecasting through
   `task_name='long_term_forecast'`,
3. the minimum required constructor surface is small enough to vendor-import
   cleanly:
   - `prompt_num`
   - `d_model`
   - `patch_len`
   - `stride`
   - `e_layers`
   - `n_heads`
   - `dropout`

This means UniTS is operationally closer to a high-complexity second-wave
entrant than to a docs-only curiosity.

## 3. Local integration path used

New local implementation:

- `src/narrative/block3/models/units_model.py`

Supporting runtime helper:

- `src/narrative/block3/models/optional_runtime.py`

Registry exposure:

- `src/narrative/block3/models/deep_models.py`

Current scope of this wrapper is intentionally limited to:

1. forecasting only,
2. one-task local training,
3. tiny smoke,
4. first narrow benchmark-clear.

It does **not** yet claim:

- prompt tuning,
- pretraining,
- zero-shot new-length transfer,
- or the full official multi-task curriculum.

## 4. Real smoke result obtained today

Artifact:

- `docs/references/local_model_smoke_20260330/units_t2_core_edgar_funding_h30.json`

Slice:

- model: `UniTS`
- task: `task2_forecast`
- ablation: `core_edgar`
- target: `funding_raised_usd`
- horizon: `30`
- max entities: `4`
- max rows: `300`

Wrapper config:

- `input_size = 60`
- `patch_len = 10`
- `stride = 10`
- `d_model = 64`
- `prompt_num = 4`
- `e_layers = 2`
- `n_heads = 4`
- `batch_size = 16`
- `max_epochs = 3`

Result:

- `fit_seconds = 14.95`
- `predict_seconds = 0.174`
- `prediction_std = 300966.34`
- `constant_prediction = false`
- `MAE = 265368.00`
- `RMSE = 528267.79`

This is enough to establish three things:

1. the forecasting-only wrapper is real,
2. the official backbone can train on a freeze-backed Block 3 slice,
3. the result is non-fallback and non-constant.

That is not a quality claim yet. It is a clean execution claim.

## 5. Narrow benchmark-clear status

The first narrow benchmark-clear has now **completed successfully**:

- job: `5298457` `v740_units_clr`
- partition: `gpu`
- resources:
  - `1 GPU`
  - `7 CPU`
  - `120G`
  - `4h`
  - `--qos=normal`
  - `--requeue`
- output root:
  - `runs/benchmarks/block3_phase9_localclear_20260330/units_funding_h30/`

Scope:

- `task2_forecast`
- `core_edgar`
- `funding_raised_usd`
- `h=30`
- `standard` preset
- `max_entities=16`
- `max_rows=1600`

Audited result:

- `MAE = 131725.2212`
- `RMSE = 174105.8744`
- `prediction_coverage_ratio = 1.0`
- `fairness_pass = true`
- `peak_rss_gb = 47.73`
- `train_time_seconds = 15.11`
- `inference_time_seconds = 0.018`

## 6. Current honest status

As of this pass, the correct label for `UniTS` is:

- no longer docs-only,
- locally integrated,
- tiny-smoke passed,
- first narrow benchmark-clear passed,
- but not yet canonical benchmark-landed.

That is exactly the kind of state we wanted before spending broader benchmark
budget on it.
