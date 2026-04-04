# V740 Funding Patch-Split Gate (2026-04-02)

This note records the completed controlled funding mechanism split gate submitted as `5304057 v740_fnd_g2`.

It is a local-only engineering note for V740 and does not alter the canonical Phase 9 benchmark line.

## Execution Summary

- job: `5304057 v740_fnd_g2`
- state: `COMPLETED`
- elapsed: `00:19:45`
- batch peak RSS: `~4.90 GB`
- output root: `runs/benchmarks/v740_localclear_20260401/v740_funding_patchsplit_gate_20260402/`
- surface: 6 funding-mechanism variants x 5 focused cells = 30 markdown summaries + 60 JSON artifacts

## Variants Tested

- `baseline_off`: log off, source scaling off, anchor off
- `log_only`: log on, source scaling off, anchor off
- `log_anchor`: log on, source scaling off, anchor on
- `log_scale`: log on, source scaling on, anchor off
- `patch1_full`: log on, source scaling on, anchor on
- `scale_anchor_no_log`: log off, source scaling on, anchor on

## Focused Gate Cells

- `task2_forecast / full / funding_raised_usd / h7`
- `task2_forecast / full / funding_raised_usd / h30`
- `task2_forecast / core_edgar / funding_raised_usd / h30`
- `task2_forecast / core_only / funding_raised_usd / h30`
- `task1_outcome / full / is_funded / h30`

## Outcome Table

| Variant | full h7 | full h30 | core_edgar h30 | core_only h30 | binary guard |
|---|---:|---:|---:|---:|---:|
| `baseline_off` | `158726.3889` | `158458.3889` | `148393.2769` | `351842.4101` | `0.2784` |
| `log_only` | `68004475556.9444` | `158458.3889` | `158458.3889` | `504390394.5444` | `0.2784` |
| `log_anchor` | `68004475556.9444` | `158458.3889` | `158458.3889` | `24086678177.1764` | `0.2784` |
| `log_scale` | `68004475556.9444` | `158458.3889` | `158458.3889` | `504397736.7210` | `0.2784` |
| `patch1_full` | `68004475556.9444` | `158458.3889` | `158458.3889` | `24086683078.2365` | `0.2784` |
| `scale_anchor_no_log` | `25817.3024` | `28596.4352` | `20305.9931` | `58058.6240` | `0.2784` |

Reference incumbent local MAE on the same five cells:

- `full / funding / h7`: `463.6938`
- `full / funding / h30`: `463.6847`
- `core_edgar / funding / h30`: `463.6847`
- `core_only / funding / h30`: `6238.5882`
- `full / is_funded / h30`: `0.2730`

## Main Conclusions

1. `log1p` is the dominant damage source on the current funding path.
   Every log-enabled variant stays catastrophic or explodes further.

2. The best current funding regime is `scale_anchor_no_log`.
   Relative to `baseline_off`, it improves the four funding gate cells by large factors:
   - `full / h7`: `158726.3889 -> 25817.3024`
   - `full / h30`: `158458.3889 -> 28596.4352`
   - `core_edgar / h30`: `148393.2769 -> 20305.9931`
   - `core_only / h30`: `351842.4101 -> 58058.6240`

3. The split gate does **not** show a competitive funding regime yet.
   Even the best variant is still far worse than the incumbent on all four funding cells.

4. The binary guard is non-discriminative for these funding controls.
   All six variants land the same `0.2784` MAE on `task1_outcome / full / is_funded / h30`.

## Interpretation

The new control surface answers the first honest attribution question cleanly:

- keep exploring the no-log branch
- stop spending local gate budget on log-domain funding variants

The next useful question is narrower:

- in the no-log branch, how much lift comes from source scaling alone?
- how much comes from the anchor?
- does anchor strength need to be reduced below `0.85`?

That is the motivation for the next `no-log` sub-split gate.

## Important Metadata Lesson

The binary guard JSON confirms why the guard no longer differentiates these variants:

- `funding_target = false`
- effective `funding_log_domain = false`
- effective `funding_source_scaling = false`
- effective `funding_anchor = false`

So the guard remains useful as a coarse sanity check, but it should not be treated as evidence that one funding variant is safer than another when the effective binary regime is identical.