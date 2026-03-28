# Local Model Smoke Notes (2026-03-28)

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

### Interpretation

This upgrades SAMformer from “wrapper exists” to “wrapper plus generic
freeze-backed local smoke passes on a real slice.” It is still **not**
canonical-benchmark-cleared. The next step should be a narrow benchmark-clear
path, not a large submission.

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

### Interpretation

Prophet is no longer blocked by wrapper absence or dependency availability.
The current practical path is a **repo/user-local vendor install**, not
mutating the shared insider env.

## 3. TabPFN / TabPFN-TS

### Current factual status

- local generic wrappers: **already exist**
  - `src/narrative/block3/models/traditional_ml.py`
- `GLIBCXX_3.4.31` blocker: **resolved**
  - preloading the insider env `libstdc++.so.6.0.34` makes `tabpfn 6.3.2`
    importable
- wrapper support added:
  - `BLOCK3_TABPFN_MODEL_PATH` may now be used to point to a local checkpoint
- current blocker:
  - official model download is **gated**
  - tiny local smoke now reaches model loading, then fails with
    Hugging Face access-control instructions for `Prior-Labs/tabpfn_2_5`

### Interpretation

TabPFN is no longer blocked by runtime/toolchain compatibility. It is currently
blocked by **gated model-weight access** unless a local checkpoint path is
provided or the active Hugging Face account has actually accepted the
`Prior-Labs/tabpfn_2_5` access terms. A successful `hf auth login` by itself is
not enough if the account still lacks the repo-level gated approval.

## 4. Narrow benchmark-clear queue status

As of 2026-03-28, two local-only narrow benchmark-clear jobs are now queued
through the canonical harness with isolated output roots:

- `5294242` `v740_samf_clr`
  - `SAMformer`
  - `task1_outcome / core_edgar / is_funded / h=14`
- `5294243` `v740_prop_clr`
  - `Prophet`
  - `task2_forecast / core_only / funding_raised_usd / h=30`

These are resumable SLURM jobs under
`runs/benchmarks/block3_phase9_localclear_20260328/`. They are intended to
clear the models against the real benchmark harness without polluting the
canonical leaderboard.
