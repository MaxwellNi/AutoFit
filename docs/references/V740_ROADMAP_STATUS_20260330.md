# V740 / Comparator Roadmap Status (2026-03-30)

This note answers one narrow operational question:

> Of the concrete next-wave routes we said we would push, which are already
> done to a meaningful evidence threshold, and which are still incomplete?

The threshold here is strict:

- "done" means at least **real local-clear / audited evidence exists**
- not just paper reading or wrapper scaffolding
- not canonical benchmark landed unless explicitly stated

## 1. Comparator lane status

| Wave | Item | Current status | Best evidence today | Honest next step |
| --- | --- | --- | --- | --- |
| First | `TabPFNRegressor 2.6` | **narrow-clear evaluated; not promotable yet** | funding clear runs with `fairness_pass=true` but weak quality; investors clear returns `fairness_pass=false` | do not widen yet unless a new targeted count-safe setting emerges |
| First | `SAMformer` | **narrow-clear done** | `MAE=0.3485`, `fairness_pass=true` on `task1_outcome/core_edgar/is_funded/h14` | decide whether to widen into a small local-clear matrix |
| First | `Prophet` | **narrow-clear done; sanity baseline only** | corrected CPU-only narrow clear succeeds, but quality is weak | keep only as a reviewer-friendly sanity baseline |
| Second | `LightGTS` | **narrow-clear done** | repaired narrow clear `MAE=201930.5506`, `fairness_pass=true` | widen local-clear matrix before any canonical promotion |
| Second | `OLinear` | **funding narrow-clear done; count side incomplete** | funding clear `MAE=131288.8062`, `fairness_pass=true`; count-side evidence is still weak | add a count-side audited probe before any promotion decision |
| Third | `ElasTST` | **partially done** | funding clear succeeds; investors clear runs but `fairness_pass=false` | keep in local-clear lane and fix count-side fairness first |
| Third | `UniTS` | **funding narrow-clear done** | `MAE=131725.2212`, `fairness_pass=true` | widen beyond the first funding clear |

## 2. V740 methodology lane status

| Item | Current status | Best evidence today | Honest next step |
| --- | --- | --- | --- |
| `DistDF` | **first and second engineering passes landed** | source-aware multistep alignment stays non-degenerate on funding and binary audits | continue tightening target-specific multistep behavior |
| `Selective Learning` | **first and second engineering passes landed** | weighted objective remains stable; later passes improved audited slices rather than breaking them | continue auditing on harder binary and fuller-source slices |
| `CASA` | **first lightweight code pass landed** | `CASALocalContextBlock` now materially improves audited `core_edgar` funding `h=30` and helps binary `h=14` | keep testing on fuller-source and harder long-h slices |
| `TimeEmb` | **first lightweight code pass landed** | `StaticDynamicTimeFusion` now supports stable fuller-source audits with `text_source_density=1.0` | continue testing where text may become real gain rather than neutral source coverage |

## 3. What is actually done vs not done

### Done to a meaningful evidence threshold

- `TabPFNRegressor 2.6` narrow-clear decision has been made:
  - runnable,
  - but not worth widening right now
- `SAMformer`
- `Prophet`
- `LightGTS`
- `OLinear` funding side
- `ElasTST` funding side
- `UniTS` funding side
- `DistDF`
- `Selective Learning`
- first lightweight `CASA`
- first lightweight `TimeEmb`

### Not done to the extreme yet

- `OLinear` count-side evidence
- `ElasTST` count-side fairness
- `UniTS` beyond its first funding clear
- whether `SAMformer` deserves wider benchmark promotion
- whether text becomes a real gain source under the new `CASA + TimeEmb` pass
- canonical benchmark landing for any of the new local-clear entrants

## 4. Current honest summary

The earlier promised roadmap is **not fully complete to the extreme** yet.

But it is no longer just a roadmap:

- the first wave is effectively settled at narrow-clear level,
- the second wave now has two real entrants (`LightGTS`, `OLinear`) beyond
  docs-only status,
- the third wave now has two real entrants (`ElasTST`, `UniTS`) beyond
  docs-only status,
- and the V740 methodology lane now has the first real code-and-audit evidence
  for `DistDF`, `Selective Learning`, `CASA`, and `TimeEmb`.

That means the project has moved from:

- "these are good ideas"

to:

- "these are now audited engineering lines with real evidence, but several
  still need a second or third expansion step before they can be called
  complete."
