# V740 / Comparator Roadmap Status (Updated 2026-04-03)

This note answers one operational question:

> where does the V740 line truly stand after the April 1-3 local evidence wave?

For the forward execution sequence after the 2026-04-04 planning refresh, use:

- `docs/V740_MASTER_EXECUTION_PLAN.md`

This file should now be read as a status snapshot, not as the full execution roadmap.

Strict reading rules:

- executed evidence only counts if the job actually ran and produced artifacts
- queued jobs are not results
- local V740 evidence is not canonical benchmark truth

## 1. Executed V740 State

| Lane | Current status | Best executed evidence today | Honest read |
| --- | --- | --- | --- |
| binary non-routed shared112 | complete | `16/16` complete, aggregate `7/2/7`; full post-audit `10/0/6`; routed h1 `2/0/2` | real binary competitiveness exists, and routed h1 has first landed proof, but it is not yet dominant |
| funding non-routed shared112 | complete | `48/48` complete, aggregate `8/0/40`; widened best-branch duel `5304260` finished `20/28` for both strongest branches | funding can be partially rescued, but this line is not a near-champion funding lane |
| investors non-routed shared112 | complete | `48/48` complete, aggregate `0/0/48`; full post-audit `0/0/48`; h1 post-audit rerun `0/0/12` | investors remains a structural failure lane |
| EDGAR root-cause audit | complete | exact-day vs as-of join mismatch falsified on the freeze-backed daily surface | do not spend another cycle on EDGAR join semantics |
| text root-cause audit | complete | embeddings are present and aligned; the current weakness is representation, not missing assets | do not spend another cycle on "missing text embeddings" stories |
| target-routed implementation | code and smoke complete | routed decoder, count-anchor head, and count-jump path are live; routed smokes stayed non-constant | the implementation path is real code, not a design placeholder |

## 2. Executed Comparator State

| Comparator lane | Current status | Honest use today |
| --- | --- | --- |
| `SAMformer` | completed through a wider local-only lane | comparator evidence exists already; not a current queue blocker |
| `LightGTS` | completed through a wider local-only lane | comparator evidence exists already; remains a local-only side path |
| `OLinear` | funding-side local clear, count-side not promotable | funding-only comparator |
| `ElasTST` | funding-side local clear, count-side not promotable | funding-only comparator |
| `UniTS` | funding-side local clear, count-side not promotable | funding-only comparator |

## 3. What Is Still Pending

| Item | Job(s) | Scope | Landed today | Current ETA |
| --- | --- | --- | --- | --- |
| full routed investors loop | `5305468 v740_112_inv` | shared112 investors routed loop | `0/48` cells | `2026-04-09T22:50:00` |
| full routed binary loop | `5305469 v740_112_bin` | shared112 binary routed loop | `0/16` cells | `2026-04-09T22:50:00` |
| routed investors h1 probe | `5305472 v740_112_invh1` | routed investors `h=1` probe | `0/12` cells | `2026-04-09T22:50:00` |

Already landed during the April 3 wave:

- `5304393 v740_repr_pa`: completed, landing full binary post-audit (`10/0/6`, `32` JSON) and investors post-audit (`0/0/48`, `96` JSON)
- `5305473 v740_112_binh1`: completed, landing the first routed summary doc plus `8` routed JSON outputs for binary h1 (`2/0/2` over `4` cells)

## 4. Progress Dashboard

These bars track execution progress, not outcome quality.

- Code path landing: `[##########]` complete
- Shared112 non-routed diagnosis: `[##########]` `112/112` complete
- Funding branch selection: `[##########]` complete through `g4`
- Formal routed full cells: `[..........]` `0/64` landed
- Formal routed h1 probes: `[##........]` `4/16` landed
- Routed shared112 verdict: `[##........]` partial readout only
- Full160 local compare on routed line: `[..........]` not started
- Canonical promotion decision: `[..........]` not started

## 5. Remaining-Time Reality

- `5304393 v740_repr_pa` has already completed and should now be read as executed evidence, not queued work.
- `5305473 v740_112_binh1` has already completed and provided the first routed readout on binary h1.
- The remaining routed jobs `5305468`, `5305469`, and `5305472` are all currently placed on `2026-04-09T22:50:00`.
- Without a new backfill-friendly resubmission strategy, the next routed expansion beyond binary h1 cannot honestly be expected before the April 9 night window.

## 6. Current Blocker Ranking

1. gpu queue delay on the remaining routed V740 jobs
2. the unresolved investors competitiveness gap
3. funding reintegration after routing proves itself on binary and investors
4. the fact that full160 routed local compare has not started

## 7. Next Honest Sequence

1. Read out `5305468` and `5305469` before making any claim about routed shared112 gains beyond the already-landed binary h1 probe.
2. Read out `5305472` to test whether the routed path helps the hardest investors `h=1` slice at all.
3. If the routed path materially improves binary and investors, re-run a full routed shared112 package that includes funding again.
4. Only then widen to a routed full160 local compare.
5. Only after that should a canonical-promotion discussion reopen.
