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
| First | `SAMformer` | **binary narrow-clear done; funding probes promotion-positive; first serious multi-horizon lane now queued** | first audited clear: `MAE=0.3485`, `fairness_pass=true` on `task1_outcome/core_edgar/is_funded/h14`; second funding-side tiny smoke is non-fallback with `MAE=265368.0` on `task2_forecast/core_edgar/funding_raised_usd/h30`; first funding probe `5299018` failed only because `quick` preset cannot run `task2 / h=30`; repaired follow-up clear `5299636` lands with `MAE=130514.7325`, `fairness_pass=true`; wider lane `5300057` also completed cleanly on the current harness surface with the same funding-side result; the first more-serious multi-horizon follow-up is now queued as `5300644` on `task1_outcome/core_edgar/is_funded/h={1,7,14,30}` | read out `5300644` and decide whether `SAMformer` is ready to move beyond local-clear |
| First | `Prophet` | **narrow-clear done; sanity baseline only** | corrected CPU-only narrow clear succeeds, but quality is weak | keep only as a reviewer-friendly sanity baseline |
| Second | `LightGTS` | **narrow-clear done; fuller-source funding probes promotion-positive; first serious multi-horizon lane now queued** | repaired core-edgar narrow clear `MAE=201930.5506`, `fairness_pass=true`; fuller-source tiny funding smoke is also non-fallback with `MAE=445368.0`; matching promotion probe `5299637` on `full / funding / h=30` lands cleanly with `MAE=201930.5506`, `fairness_pass=true`; wider lane `5300058` also completed cleanly on the current harness surface with the same fuller-source funding result; the first more-serious multi-horizon follow-up is now queued as `5300645` on `task2_forecast/full/funding_raised_usd/h={1,7,14,30}` | read out `5300645` and decide whether `LightGTS` is ready to move beyond local-clear |
| Second | `OLinear` | **funding narrow-clear done; count-side narrow-clear evaluated and failed fairness** | funding clear `MAE=131288.8062`, `fairness_pass=true`; count-side clear `5298523` ends with `fairness_pass=false` and constant `137.0` predictions on the audited harness slice | keep as a funding-side comparator only until count-side behavior is fixed |
| Third | `ElasTST` | **funding-side clear; count-side now effectively stop** | funding clear succeeds; first investors clear runs but `fairness_pass=false`; the more conservative repair probe `5299641` from `input_size=21 / l_patch_size=3_6` also ends with `fairness_pass=false` | keep `ElasTST` funding-only unless a materially different count-safe design appears |
| Third | `UniTS` | **funding narrow-clear done; count-side narrow-clear evaluated and failed fairness** | funding clear `MAE=131725.2212`, `fairness_pass=true`; count-side clear `5298559` ends with `fairness_pass=false` and constant `136.9999` predictions on the audited harness slice | keep `UniTS` as a funding-side comparator only until a materially different count-safe setting exists |

## 2. V740 methodology lane status

| Item | Current status | Best evidence today | Honest next step |
| --- | --- | --- | --- |
| `DistDF` | **first and second engineering passes landed** | source-aware multistep alignment stays non-degenerate on funding and binary audits | continue tightening target-specific multistep behavior |
| `Selective Learning` | **first and second engineering passes landed** | weighted objective remains stable; later passes improved audited slices rather than breaking them | continue auditing on harder binary and fuller-source slices |
| `CASA` | **first lightweight code pass landed** | `CASALocalContextBlock` now materially improves audited `core_edgar` funding `h=30` and remains stable on fuller-source binary audits through `h=60` | keep testing on fuller-source and harder long-h slices |
| `TimeEmb` | **first lightweight code pass landed** | `StaticDynamicTimeFusion` now supports stable fuller-source audits with `text_source_density=1.0`; the paired binary text-gain audit `5300635` has now completed and shows `full` beating `core_edgar` on both audited `is_funded` slices (`h=14` and `h=60`); funding remains a tie / slight edge rather than a decisive text win | treat binary fuller-source text gain as established enough for now, and keep the remaining open text question focused on funding/count behavior |

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

- `OLinear` count-side recovery beyond the current failed narrow clear
- `UniTS` count-side recovery beyond the current failed narrow clear
- reading out the first more-serious `SAMformer` multi-horizon lane (`5300644`)
- reading out the first more-serious `LightGTS` multi-horizon lane (`5300645`)
- whether the newly established binary text gain also generalizes beyond the current fuller-source audited `is_funded` slices
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

## 5. When This Actually Ends

This line of work is not supposed to be open-ended forever. The practical
finish line is reached when all four conditions below are true at the same
time:

1. the canonical benchmark backlog has converged except for the already-known
   structural exceptions (`XGBoost@159`, `XGBoostPoisson@157`),
2. the current next-wave entrants have all reached a real decision state:
   - worth promoting,
   - worth keeping as a local-clear side lane only,
   - or not worth further spend right now,
3. `V740` shows stable audited gains on:
   - a hard binary slice,
   - a funding slice,
   - and at least one longer-h or fuller-source slice,
   across consecutive mechanism iterations without fairness regression,
4. the paper-facing accounting stops drifting:
   - `55 @ 160`,
   - `58 @ 112`,
   - `62 raw active complete @ 160`,
   - `86 non-retired post-filter models`,
   and the docs no longer need frequent correction.

As of 2026-03-30, we are not at that finish line yet. The remaining named gaps
are still concrete rather than vague:

- deciding whether `SAMformer` is ready to move beyond local-clear after `5300644`,
- deciding whether `LightGTS` is ready to move beyond local-clear after `5300645`,
- deciding whether the now-positive binary text-gain result also extends beyond the current audited `is_funded` fuller-source slices,
- `UniTS` count-side recovery,
- and getting at least one new entrant past local-clear into a more serious
  benchmark lane.
