# LightGTS Integration Audit

> Last updated: 2026-03-29
> Scope: direct official-repo audit to decide the most realistic Block 3
> integration path for `LightGTS`.

## 1. Why this audit was needed

`LightGTS` is currently one of the highest-value missing efficient comparators
for Block 3 and for the V740 design line. But “official code exists” is not the
same as “the model is ready to drop into our benchmark harness”.

This note records the direct same-day audit of the official source tree so that
future integration work starts from verified facts rather than assumptions.

## 2. Official source verified today

- official repo:
  - `https://github.com/decisionintelligence/LightGTS`
- official HEAD verified on 2026-03-28:
  - `36ad5bfb4c71ce11bf0372f8e4433c29e1ea5ff5`
- paper:
  - `https://arxiv.org/abs/2506.06005`

## 3. What the official repo actually contains

Direct source audit confirms the repo is a **script-driven PyTorch project**,
not a clean package-style benchmark library.

Verified top-level files:

- `finetune.py`
- `zero_shot.py`
- `pretrain.py`
- `datautils.py`
- `src/models/LightGTS.py`
- `src/models/LightGTS_resample.py`
- `checkpoints/`

The README explicitly advertises:

- zero-shot and full-shot usage through CLI scripts,
- pretraining / transfer settings,
- released checkpoints,
- public-benchmark datasets such as `ETT`, `Weather`, `Traffic`, `Exchange`,
  `Solar`, and `PEMS`.

## 4. Practical integration implications for Block 3

### What is good news

1. The repo has a real official implementation.
2. The model class is visible in source rather than hidden behind binaries.
3. Released checkpoints exist, so both full-shot and checkpoint-based paths are
   imaginable.

### What is not plug-and-play

1. The official repo expects its own dataset names and dataset loader stack.
2. The training scripts hardcode public-benchmark assumptions and even set
   `CUDA_VISIBLE_DEVICES` at script level.
3. The repo ships a broad and heavy `requirements.txt`, so “install the whole
   thing into the shared env” would be a bad operational choice.

## 5. Best integration path

The current best path is:

1. **do not** treat `LightGTS` as a package lane like `StatsForecast`;
2. **do not** try to reuse the official CLI / `datautils.py` stack directly for
   Block 3;
3. instead, build a **custom vendor wrapper** that:
   - imports the LightGTS model class from the official repo,
   - maps Block 3 panel windows into the expected tensor layout,
   - uses our own audited train/val/test slicing,
   - and keeps output writing inside the Block 3 harness contract.

This makes `LightGTS` much closer to the current `SAMformer` integration path
than to a “just install a package and call it” path.

## 6. Current status in Block 3 terms

- official repo: **verified**
- code path quality: **high enough to proceed**
- vendor-path helper: **now exists** in
  `src/narrative/block3/models/optional_runtime.py`
- local wrapper scaffold: **now exists** in
  `src/narrative/block3/models/lightgts_model.py`
- vendor import + minimal instantiation smoke: **passed** on 2026-03-29
- first tiny freeze-backed real-data smoke: **completed** on 2026-03-29
  - first pass (`input_size=96`) produced **no training windows** and therefore
    fell back to a constant predictor
  - second pass (`input_size=60`, `patch_len=15`, `stride=15`) became
    **non-fallback and non-constant**
  - audited slice:
    `task2_forecast / core_edgar / funding_raised_usd / h=30`
  - result:
    - `fit_seconds = 4.60`
    - `prediction_std = 10802.78`
    - `constant_prediction = false`
    - `MAE = 445367.99`
- registry entry: **not added yet**
- registry entry: **now added locally** under the `transformer_sota` lane on 2026-03-29
- narrow benchmark-clear:
  - first attempt `5298059` completed without model execution because the
    compute node could not see `/tmp/LightGTS`
  - repaired rerun `5298289` completed successfully after switching to the
    persistent user-local path `~/.cache/block3_optional_repos/LightGTS`
  - result on the audited slice:
    - `MAE = 201930.5506`
    - `RMSE = 205737.7476`
    - `prediction_coverage_ratio = 1.0`
    - `fairness_pass = true`
    - `peak_rss_gb = 47.40`
- canonical benchmark landing: **none yet**

## 7. Recommended next engineering step

The next concrete step should now be:

1. treat the repo-path issue as solved and keep using the persistent vendor path
   instead of `/tmp/LightGTS`,
2. inspect whether this first successful narrow clear is strong enough to justify
   a wider local-clear matrix,
3. only then decide whether `LightGTS` deserves a true canonical expansion attempt.

## 8. Bottom line

`LightGTS` is now past the “paper name only” stage.

The verified status is:

> **official repo audited, custom wrapper scaffolded, vendor import smoke
> passed, first real-data tiny smoke is non-fallback with a shorter context,
> and the repaired first narrow benchmark-clear now succeeds on a real audited
> slice; the realistic Block 3 route remains a custom vendor wrapper with a
> widened local-clear gate before any canonical expansion.**
