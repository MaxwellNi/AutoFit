# V740 Funding No-Log Sub-Split Gate (2026-04-02)

This note records the completed no-log funding sub-split gate submitted as `5304068 v740_fnd_g3`.

It is a local-only engineering result for V740 and does not alter the canonical Phase 9 benchmark line.

## Execution Summary

- job: `5304068 v740_fnd_g3`
- state: `COMPLETED`
- elapsed: `00:24:40`
- batch peak RSS: `~4.32 GB`
- output root: `runs/benchmarks/v740_localclear_20260401/v740_funding_nolog_subsplit_gate_20260402/`
- surface: 6 variants x 6 focused cells = 36 markdown summaries + 72 JSON artifacts

## Variants Tested

- `baseline_off`: no log, no scaling, no anchor
- `scale_only_no_log`: scaling only
- `anchor_only_no_log_a035`: anchor only, strength `0.35`
- `anchor_only_no_log_a085`: anchor only, strength `0.85`
- `scale_anchor_no_log_a035`: scaling + anchor `0.35`
- `scale_anchor_no_log_a085`: scaling + anchor `0.85`

## Focused Cells

- `task2_forecast / full / funding_raised_usd / h7`
- `task2_forecast / full / funding_raised_usd / h30`
- `task2_forecast / core_edgar / funding_raised_usd / h7`
- `task2_forecast / core_edgar / funding_raised_usd / h30`
- `task2_forecast / core_only / funding_raised_usd / h30`
- `task1_outcome / full / is_funded / h30`

## Outcome Table

| Variant | full h7 | full h30 | core_edgar h7 | core_edgar h30 | core_only h30 | binary guard |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_off` | `158726.3889` | `158458.3889` | `158726.3889` | `148396.7941` | `351829.5819` | `0.2784` |
| `scale_only_no_log` | `158726.3889` | `158458.3889` | `158726.3889` | `148157.1990` | `351828.0567` | `0.2784` |
| `anchor_only_no_log_a035` | `103871.1475` | `105937.0297` | `103962.1941` | `97099.9113` | `231062.8415` | `0.2784` |
| `anchor_only_no_log_a085` | `158726.3889` | `28141.4149` | `24786.1984` | `22247.2684` | `58058.5828` | `0.2784` |
| `scale_anchor_no_log_a035` | `103899.1658` | `108780.8195` | `104095.0154` | `99654.3212` | `231062.9261` | `0.2784` |
| `scale_anchor_no_log_a085` | `25816.2830` | `28596.4350` | `23538.2988` | `20306.0650` | `58058.5837` | `0.2784` |

Reference incumbent local MAE on the same cells:

- `full / h7`: `463.6938`
- `full / h30`: `463.6847`
- `core_edgar / h7`: `165158.0177`
- `core_edgar / h30`: `463.6847`
- `core_only / h30`: `6238.5882`
- `full / is_funded / h30`: `0.2730`

## Main Conclusions

1. Source scaling alone is not the active rescue mechanism.
   `scale_only_no_log` is effectively identical to `baseline_off` on the tested surface.

2. The dominant positive funding mechanism is the continuous anchor, not source scaling.
   Both anchor-only variants materially improve the tested funding cells.

3. Anchor strength matters sharply.
   `0.85` is much stronger than `0.35` on the current audited surface.
   The clearest examples are:
   - `core_edgar / h30`: `97099.9113 -> 22247.2684`
   - `core_only / h30`: `231062.8415 -> 58058.5828`
   - `full / h30`: `105937.0297 -> 28141.4149`

4. Scaling only becomes useful when paired with the strong anchor.
   The strongest evidence is `full / h7`, where:
   - `anchor_only_no_log_a085`: `158726.3889`
   - `scale_anchor_no_log_a085`: `25816.2830`
   So scaling is currently a stabilizer / amplifier for the hardest fuller-source short-horizon funding cell, not the main mechanism by itself.

5. The best current primary funding regime is `scale_anchor_no_log_a085`.
   The best secondary comparator is `anchor_only_no_log_a085`.
   Those are the only two variants that now justify widening to a broader funding surface.

6. The binary guard is still non-discriminative.
   All variants land the same `0.2784` MAE because funding controls are inactive when `funding_target = false`.

## Operational Reading

The funding mechanism search is now much narrower than it was before the two April 2 gates:

- abandon log-domain funding variants
- stop treating source scaling as a first-order mechanism
- treat strong anchor as the main funding rescue path
- keep scaling only as a conditional companion to the strong anchor branch

This does **not** mean funding is solved.
Even the best variants remain far behind the incumbent on the hardest fuller-source and longer-horizon funding cells.
What changed is that the search space is now honest and small enough to widen without wasting budget on already-failed branches.