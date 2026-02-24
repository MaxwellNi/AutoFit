ONE-LINER TS AGENT PROMPT: Build time-series models ONLY with provable zero-leakage (feature availability-time ≤ t_obs_end, correct label alignment to t_target, no centered/forward ops, train-only fit for any stats/preprocess/feature-select/PCA/outlier rules, as-of snapshots for revisable data, no entity/event/near-duplicate split leakage), strict train==infer parity (same pipeline + cache keyed by data_version/split/preproc_hash, eval in deployed inference mode rolling/recursive, correct train/eval modes incl. dropout/BN, past-only padding/interpolation, toy unit tests for window indexing), and deployment-real evaluation evidence (overall + sliced metrics with worst-slice, tail+business metrics, calibration/coverage if probabilistic, drift monitor + retrain/rollback plan) with full reproducibility logs (seeds/env/hashes/commit, leakage+split audits, reproducible run command); if any requirement can’t be met, STOP and report the exact blocker.

# Block3 Execution Contract

- Contract-Version: `2026-02-24`
- Policy-Hash-SHA256: `cae333137aa152bc12f534269c39ca096cf7bfefe129445b113b2df852047bbf`

## Scope

This contract is mandatory for every Block3 run path:

- Iris Slurm submit and preflight scripts
- Local 3090/4090 production runners
- Any benchmark or completion controller scripts for AutoFit V7/V7.1/V7.2

## Non-Negotiable Controls

1. No leakage:
   - Feature availability time must be at or before observation cutoff.
   - Label alignment must match the declared target horizon.
   - No centered or forward-looking transforms.
   - No train/validation/test cross-contamination for fitting preprocessors.
   - Re-issuable sources must use as-of snapshots only.
   - No entity/event/near-duplicate split leakage.
2. Train-infer parity:
   - Same transformation graph in training and inference.
   - Same cache semantics (`data_version`, `split`, `preproc_hash`).
   - Evaluation must execute in deployment-equivalent rolling/recursive mode.
3. Reproducibility:
   - Persist seeds, commit hash, runtime environment, and config hashes.
   - Persist leakage and split audit outputs.
   - Persist exact reproducible run command.
4. Blocker-stop policy:
   - If any mandatory control is not satisfied, abort before submission/execution.

## Runtime Requirements

1. Runtime must use `insider` environment only.
2. Python interpreter must come from the `insider` prefix and be `>=3.11`.
3. Dependencies must be repaired only inside `insider`.
4. No `base` execution, no alternate environment creation for Block3 paths.

## Leakage and Split Audit Requirements

Required checks before any large run:

1. Freeze pointer verification pass.
2. Leakage guard coverage pass.
3. Prediction coverage guard pass.
4. Stability and distribution checks present and readable.
5. No test-feedback-based model or hyperparameter selection.

## Train-Infer Parity Requirements

1. Same pipeline logic for fit and predict.
2. No inference-only feature behavior.
3. Temporal ordering preserved for all rolling/recursive calls.
4. Correct mode toggles for deep models (dropout/normalization behavior).

## Reproducibility Requirements

1. Log deterministic seeds and software versions.
2. Log runtime interpreter path and Python version.
3. Log config hash and contract audit output.
4. Log command line invocation for the run.

## Forbidden Run Modes

1. `ALLOW_UNSAFE_SKIP_PREFLIGHT=1` for normal production runs.
2. Any flag that disables leakage, split, or coverage guards.
3. Any use of test-set metrics for model selection or hyperparameter search.
4. Any run started without successful contract assertion.

## Blocker-Stop Policy

If any mandatory requirement in this contract is not satisfied, the run must stop with a precise blocker message and no job submission.
