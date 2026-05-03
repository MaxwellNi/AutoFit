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