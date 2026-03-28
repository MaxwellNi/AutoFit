# Local Model Smoke Notes (2026-03-28)

This note records the first round of generic local smokes run through
`scripts/run_block3_local_model_smoke.py`. These are **not** canonical
benchmark results. They are small, freeze-backed audited slices used to decide
whether a newly integrated model is healthy enough to justify more expensive
benchmark work.

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
- blocker:
  - insider env import fails with
    `ModuleNotFoundError: No module named 'prophet'`

### Interpretation

Prophet is no longer blocked by wrapper absence. It is currently blocked by
environment dependency availability.

## 3. TabPFN / TabPFN-TS

### Current factual status

- local generic wrappers: **already exist**
  - `src/narrative/block3/models/traditional_ml.py`
- blocker:
  - insider env import fails with
    `libstdc++.so.6: version 'GLIBCXX_3.4.31' not found`

### Interpretation

TabPFN is currently blocked by runtime/toolchain compatibility, not by missing
model glue code.
