# V740 / Comparator Roadmap Status (Updated 2026-04-03)

This note answers one operational question:

> where does the V740 line truly stand after the April 1-3 local evidence wave?

Strict reading rules:

- executed evidence only counts if the job actually ran and produced artifacts
- queued jobs are not results
- local V740 evidence is not canonical benchmark truth

## 1. Executed V740 State

| Lane | Current status | Best executed evidence today | Honest read |
| --- | --- | --- | --- |
| binary non-routed shared112 | complete | `16/16` complete, aggregate `7/2/7`; h1 post-audit rerun `2/0/2` | real binary competitiveness exists, but it is not yet dominant |
| funding non-routed shared112 | complete | `48/48` complete, aggregate `8/0/40`; widened best-branch duel `5304260` finished `20/28` for both strongest branches | funding can be partially rescued, but this line is not a near-champion funding lane |
| investors non-routed shared112 | complete | `48/48` complete, aggregate `0/0/48`; h1 post-audit rerun `0/0/12` | investors remains a structural failure lane |
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

## 3. What Has Not Landed Yet

| Item | Job(s) | Scope | Landed today | Current ETA |
| --- | --- | --- | --- | --- |
| post-audit repr rerun | `5304393 v740_repr_pa` | full binary + investors post-audit rerun | `0` new outputs from this pending job | `2026-04-11T21:20:00` |
| full routed investors loop | `5305468 v740_112_inv` | shared112 investors routed loop | `0/48` cells | `2026-04-12T11:20:00` |
| full routed binary loop | `5305469 v740_112_bin` | shared112 binary routed loop | `0/16` cells | `2026-04-12T12:10:00` |
| routed investors h1 probe | `5305472 v740_112_invh1` | routed investors `h=1` probe | `0/12` cells | `2026-04-12T15:30:00` |
| routed binary h1 probe | `5305473 v740_112_binh1` | routed binary `h=1` probe | `0/4` cells | `2026-04-12T15:40:00` |
| routed summary docs | none landed | `docs/references/V740_SHARED112_*ROUTED*_2026040*.md` | `0` docs | blocked on queue |
| routed JSON outputs | none landed | `runs/benchmarks/v740_localclear_20260402/` and `...20260403/` | `0` outputs | blocked on queue |

## 4. Progress Dashboard

These bars track execution progress, not outcome quality.

- Code path landing: `[##########]` complete
- Shared112 non-routed diagnosis: `[##########]` `112/112` complete
- Funding branch selection: `[##########]` complete through `g4`
- Formal routed full cells: `[..........]` `0/64` landed
- Formal routed h1 probes: `[..........]` `0/16` landed
- Routed shared112 verdict: `[..........]` waiting on queue
- Full160 local compare on routed line: `[..........]` not started
- Canonical promotion decision: `[..........]` not started

## 5. Remaining-Time Reality

- The scheduler currently places `5304393 v740_repr_pa` on `2026-04-11T21:20:00`.
- The first full routed job `5305468 v740_112_inv` is currently placed on `2026-04-12T11:20:00`.
- The first routed binary job `5305469 v740_112_bin` is currently placed on `2026-04-12T12:10:00`.
- The shorter h1 routed probes are not currently winning the queue race; they are also placed on `2026-04-12`.
- Without a new backfill-friendly resubmission strategy, the first formal routed readout cannot honestly be expected before the April 11-12 window.

## 6. Current Blocker Ranking

1. gpu queue delay on the routed/post-audit V740 jobs
2. the unresolved investors competitiveness gap
3. funding reintegration after routing proves itself on binary and investors
4. the fact that full160 routed local compare has not started

## 7. Next Honest Sequence

1. Let `5304393` or one of the routed jobs actually start and land real outputs.
2. Read out `5305468` and `5305469` before making any claim about routed shared112 gains.
3. If the routed path materially improves binary and investors, re-run a full routed shared112 package that includes funding again.
4. Only then widen to a routed full160 local compare.
5. Only after that should a canonical-promotion discussion reopen.
