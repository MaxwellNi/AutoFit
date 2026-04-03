# V740 Representation Post-Audit Gate (2026-04-02)

This note records only evidence that has actually been executed.
It does not treat queued jobs as completed evidence.

## Newly Executed Post-Audit Probes

### Binary h1 probe

Source: `docs/references/V740_SHARED112_BINARY_H1_POSTAUDIT_20260402.md`

- Scope: shared112, `is_funded`, `h1`, 4 cells
- Result: `2 wins / 0 ties / 2 losses`
- Relative to the earlier h1 probe (`V740_SHARED112_BINARY_PROBE_PATCH3_20260401.md`): `0 improved / 4 worsened`
- Cell deltas:
  - `core_only`: `0.559171 -> 0.560637` (worse)
  - `core_edgar`: `0.267157 -> 0.269477` (worse, but still a win)
  - `core_text`: `0.555625 -> 0.682232` (materially worse)
  - `full`: `0.266948 -> 0.267962` (worse, but still a win)

Interpretation:

- The representation-path fix preserved the existing win structure on `core_edgar` and `full`.
- It did not improve binary h1 overall.
- `core_text` became substantially worse, so the new text path is still not a reliable net gain even on a slice where text should have a chance to help.

### Investors h1 probe

Source: `docs/references/V740_SHARED112_INVESTORS_H1_POSTAUDIT_20260402.md`

- Scope: shared112, `investors_count`, `h1`, 12 cells
- Result: `0 wins / 0 ties / 12 losses`
- Relative to the earlier full investors loop h1 cells (`V740_SHARED112_INVESTORS_LOOP_20260401.md`): `7 improved / 5 worsened`

Largest improvements:

- `task1_outcome / full`: `241.704217 -> 50.473424`
- `task3_risk_adjust / core_text`: `23.058279 -> 15.001366`
- `task2_forecast / full`: `83.562933 -> 79.249995`

Largest regressions:

- `task2_forecast / core_edgar`: `83.781694 -> 297.317808`
- `task2_forecast / core_only`: `7.323342 -> 14.567524`
- `task1_outcome / core_only`: `31.079107 -> 36.513078`

Interpretation:

- The representation-path fix reduced error on several investors h1 cells.
- That improvement never crossed the competitiveness boundary.
- All 12 cells still lost, and the incumbent local MAEs remain near-zero on the hardest cases.
- This is not a "nearly solved" lane. It is still a structural failure lane.

## Previously Executed Wider Evidence

### Binary wider loop

Source: `docs/references/V740_SHARED112_BINARY_LOOP_20260401.md`

- `7 wins / 1 tie / 8 losses`

This shows V740 already had a real binary win structure before the post-audit probe.
The new probe shows that the representation fix did not extend that structure.

### Investors wider loop

Source: `docs/references/V740_SHARED112_INVESTORS_LOOP_20260401.md`

- `0 wins / 48 losses`
- `21 constant predictions`

The new h1 probe removes the constant-prediction symptom on the sampled cells, but not the competitive gap.

### Funding widened best-branch duel

Source directories under `runs/benchmarks/v740_localclear_20260401/v740_funding_bestbranch_duel_20260402/`

- `anchor_only_no_log_a085`: `20 wins / 28 losses`
- `scale_anchor_no_log_a085`: `20 wins / 28 losses`
- `full` funding remained `0 wins / 12 losses`

This means funding branch search has already converged enough to define the ceiling of the current line: it can be partially rescued, but not turned into a dominant lane by small regime patches.

## What The Current Line Can And Cannot Become

### What is still real

- V740 can sustain a real binary win structure on `core_edgar` and `full`.
- V740 can partially rescue funding with the no-log strong-anchor regime.
- Representation cleanup is still worth doing because it can reduce some pathological investors h1 errors.

### What is no longer defensible

- It is not defensible to claim that the current single shared V740 line is on track to become a near-universal champion on the full benchmark.
- It is not defensible to claim that another small round of representation polishing will close the investors lane.
- It is not defensible to claim that the current text path is a stable net positive. The executed binary h1 probe shows the opposite on `core_text`.

## End-State Conclusion

If the constraint is:

- one V740 model package,
- roughly the current size and compute envelope,
- robustness across `is_funded`, `funding_raised_usd`, and `investors_count`,
- and "close to full benchmark overall champion",

then the executed evidence says the current homogeneous V740 line does **not** reach that end-state.

The realistic end of this line is a Pareto point:

- competitive on binary in selected ablations,
- partially competitive on funding,
- structurally non-competitive on investors.

To move past that ceiling, the solution class has to change.
The minimum plausible change is not another representation tweak. It is explicit target-specialization inside the model package, for example:

- target-specific expert heads,
- routed objective-specific decoding,
- or a similar architecture that stops forcing `is_funded`, `funding_raised_usd`, and `investors_count` through one effectively shared prediction geometry.

If that class change is not allowed, then "single-model time-series champion across all Block 3 targets" is not a realistic objective for V740.

## Executed Continuation (2026-04-03)

Job `5304393` (`v740_repr_pa`) completed on `2026-04-03` in `00:37:50`.
It is no longer queued work.

The landed outputs are:

- `docs/references/V740_SHARED112_BINARY_POSTAUDIT_20260402.md` with full binary post-audit results: `10 wins / 0 ties / 6 losses` (`32` JSON)
- `docs/references/V740_SHARED112_INVESTORS_POSTAUDIT_20260402.md` with full investors post-audit results: `0 wins / 0 ties / 48 losses` (`96` JSON)

So the representation-path continuation did execute, but it did not change the end-state conclusion of this note:

- binary improved relative to the original wider `7/2/7` readout, but still remained mixed rather than dominant
- investors stayed a structural failure lane even after the full post-audit rerun