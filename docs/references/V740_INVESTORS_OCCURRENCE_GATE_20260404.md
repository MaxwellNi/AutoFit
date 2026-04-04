# V740 Investors Occurrence Gate Note (2026-04-04)

## Why this change landed

The investors/count lane was still using a `relu(outputs["count"])` post-processing path in both training and inference, which left the model with a dead-zone style positive constraint and no learned occurrence mechanism for zero-inflated count behavior.

The current code path now does two things:

1. replaces the count-side positive transform with `softplus`
2. adds a learned occurrence head seeded by recent-activity priors

## What actually survived local validation

The learned occurrence gate did **not** validate as a globally safe default.

Small insider smoke comparisons on representative investors h1 cases showed:

- `task1_outcome / core_only / investors_count / h1`
  - no-gate baseline: `MAE ~= 0.03599`
  - global occurrence gate variants: consistently worse
- `task2_forecast / core_edgar / investors_count / h1`
  - no-gate baseline: `MAE ~= 1.42496`
  - task2-only occurrence gate: `MAE ~= 0.65835`

So the safe current interpretation is:

- keep `softplus` as the global positive count transform
- keep the learned occurrence gate infrastructure in-tree
- activate the learned occurrence gate only for `task2_forecast` on `investors_count`
- leave `task1_outcome` and `task3_risk_adjust` investors lanes on the no-gate path until more evidence exists

## Current implementation contract

`src/narrative/block3/models/v740_alpha.py` now exposes this through regime metadata:

- requested count sparsity gate strength
- effective count sparsity gate strength
- gate source signal
- gate type: `learned_occurrence_with_history_prior_task2_only`

This makes local artifacts auditable even when the requested and effective gate strengths differ.

## Immediate next move

The next focused backfill should stay narrow:

- `task2_forecast`
- `investors_count`
- `horizon=1`
- quick profile
- compare current default vs `--disable-count-sparsity-gate` and one route-off control

That is the shortest path to confirming whether the local task2 gain persists on the benchmark-style duel surface.