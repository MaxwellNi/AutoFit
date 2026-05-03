# R14 Champion Method Redesign Notes

> Last updated: 2026-05-03 23:25:41 CEST / 2026-05-03 21:25:41 UTC

## Plain-Language Verdict

The current R14 line is not failing because we lack another metric patch. It is failing the oral-grade standard because the main method is not yet proven to learn a reusable business-evolution law.

The honest redesign is: learn business-state atoms first, let each target read those atoms through its own generative process, and treat source/text/EDGAR as sparse event/regime evidence with a no-read fallback. Reliability intervals must be audited separately from point accuracy.

## What The Latest Evidence Says

- Source read-gate is not passed. Source wins many rows by count, but loses on mean absolute error because large-tail regressions dominate.
- Source path activation is not passed. The signed source route produced 15 active rows, all negative; active paired benefit was 5 wins / 10 losses with mean source-minus-control MAE delta +61710.
- Mainline point-value audit is not passed. Current observed R14 pairing is 2 wins / 8 losses on 10 funding-heavy cells, so the shared mainline cannot be claimed as a point champion.
- Hard-cell tail guard is real but limited. It fixed one formal h30/core_text/funding coverage blocker at 0.92139, but width was about 6.48e6, roughly 3.56x the marginal interval width.
- Public-pack full summary passed for 7/7 families, 40/40 results, 0 errors. This proves runner readiness and source-free forecasting survival, not Block 3 target-geometry transfer.

## Method Contract Going Forward

The champion route is now codified in `configs/research/champion_method_contract.json` and audited by `scripts/r14_champion_method_contract_audit.py`.

The method must pass six groups together:

1. Business atoms and runtime wiring.
2. Source as guarded regime evidence, not residual multiplier.
3. Point champion evidence against best non-mainline forecasters.
4. Calibrated reliability with width/sharpness control.
5. External generalization.
6. Frontier mechanism maturity.

## Immediate Execution Strategy

1. Keep source-scaling demoted unless strict read-gate and paired-benefit gates pass.
2. Expand hard-cell tail guard from one formal cell to a funding h14/h30 matrix across ablations.
3. Add width/sharpness gates so wide intervals cannot masquerade as a true solution.
4. Prioritize distributional tail objectives and selective hard-region learning over another source multiplier patch.
5. Use the auditable forecaster protocol as the main packaging path: strongest point forecaster first, calibrated reliability second, strict row-key audit always.
6. Run public-pack and target-geometry transfer separately; do not confuse public forecasting completion with Block 3 target-geometry proof.

## Compute Policy

Use `gpu` as the primary GPU partition, `bigmem` for memory-heavy CPU/formal audit reruns, `l40s` for justified overflow, and `hopper` opportunistically when resume-safe. Current queue inspection at 2026-05-03 23:25 CEST showed only two npin jobs visible and three idle bigmem nodes, so the next formal hard-cell matrix can use bigmem aggressively without touching canonical `runs/` commit boundaries.

Do not commit generated `runs/`, `slurm_logs/`, or `.slurm_scripts` artifacts. Commit only code/config/docs/test surfaces that define reproducible behavior.

## 2026-05-04 00:00 CEST Matrix Follow-Up

The first hard-cell matrix completed cleanly as array `5382390`:

- scope: `funding_raised_usd`, h14/h30 x `core_only/core_text/core_edgar/full`, `MainlineS5FsLogOnly`
- state: 8/8 `COMPLETED`, exit `0:0`
- MaxRSS range: about 82G to 140G, so `bigmem 320G / 13 CPU` is safe but conservative for this path
- post-matrix hard-cell audit: 9 records, mean coverage 0.90422, min coverage 0.88136, 0 below 0.88
- h14 cells passed with width ratio about 1.28-1.35 at upper level 0.975
- h30 cells passed with width ratio about 3.47-3.56 at upper level 0.995

This changes the honest interpretation:

1. Hard-cell reliability is no longer a one-off result; the matrix coverage floor is real enough to use as evidence.
2. It is still not a finished solution because h30 sharpness fails the current width contract (`max_width_ratio_vs_marginal = 3.0`).
3. The correct next experiment is not another source multiplier; it is a tail-level search and/or distributional tail objective to reduce h30 interval width without losing the 0.88 coverage floor.

At 2026-05-04 00:00 CEST / 2026-05-03 22:00 UTC, a second local-only array `5382403` was submitted on `bigmem`:

- scope: h30 x `core_only/core_text/core_edgar/full`
- tested upper levels: 0.990 and 0.985
- purpose: find the smallest predeclared calibration level that preserves coverage >= 0.88 while satisfying the width/sharpness guard

## Why The Last Three Months Did Not Yet Produce A True Champion

The core mistake was not lack of effort. It was that several generations optimized local surfaces before the method contract was strict enough.

The main failure modes are now explicit:

1. **Source was treated too often as an improvement mechanism instead of evidence.** It can win many small rows while losing large-tail rows badly. The latest read-gate has row win rate 0.584 but mean source-minus-core error +11016, so source cannot be promoted.
2. **Reliability and point accuracy were mixed together in the story.** Hard-cell calibration can fix coverage, but it does not make the forecaster better. Width must be audited separately.
3. **Local gates were too easy to overread.** A narrow investors or funding signal did not transfer automatically to the full shared surface.
4. **External generalization was late.** Public-pack now passes as a runner/readiness layer, but target-geometry transfer is still missing.
5. **Frontier mechanisms were catalogued before being operationalized.** Distributional tail objectives and offline distillation are still not implemented local experiments.

## Data Problem Or Method Problem?

Both, but they are different classes of problem.

- Data reality: Block 3 has sparse events, extreme funding tails, uneven source coverage, and target coupling between occurrence/severity/count. These are genuine hard-data properties, not just bugs.
- Method problem: previous routes did not sufficiently separate shared business-state representation, target-isolated generative laws, source read/no-read confidence, and calibrated reliability. That is why metric wins failed to become a robust champion.

The data does not make the problem impossible. It means the winning method must explicitly model tail regimes, no-read regimes, and target-specific observation laws.

## Global/Local Optimality Standard

There is no honest way to guarantee a mathematical global optimum in this research setting. The enforceable substitute is a bounded, auditable search protocol:

1. Define the candidate family before running: point forecaster, distributional objective, hard-region weighting, source-memory policy, calibration policy.
2. Freeze train/calibration/test row keys and ablation surfaces.
3. Require local cell improvement and global no-regression together.
4. Require coverage, width, point accuracy, fairness, and external transfer to pass together.
5. Promote only the candidate whose audit ledger beats strong alternatives under the same frozen protocol.

This is the practical route to a champion that can survive review: not a claim that the search is mathematically exhaustive, but a proof that the promoted method is the best audited candidate under a frozen, leakage-free, multi-surface contract.