# V740 EDGAR and Text Root-Cause Audit (2026-04-02)

This note records the first implementation-stage root-cause audit for the two
most disputed V740 source paths:

- `core_edgar`
- `core_text` / `full` text embeddings

It is a local-only engineering diagnosis note. It does not alter the canonical
Phase 9 benchmark line.

## Execution Summary

- EDGAR audit script: `scripts/audit_v740_edgar_alignment.py`
- EDGAR audit output: `docs/references/V740_EDGAR_ALIGNMENT_AUDIT_20260402.json`
- Text audit script: `scripts/audit_v740_text_embeddings.py`
- Text audit output: `docs/references/V740_TEXT_EMBEDDING_AUDIT_20260402.json`
- Runtime: micromamba `insider`
- Pointer: `docs/audits/FULL_SCALE_POINTER.yaml`

The goal was not to train new models. The goal was to answer a narrower and
more important question first:

> are the current `core_edgar` / text problems primarily caused by missing or
> incorrect alignment, or are they mostly representation / mechanism problems?

## EDGAR Audit: What Is Now Confirmed

The strongest new result is negative but decisive.

- `offers_core_daily` rows: `5,774,931`
- rows with non-null `cik`: `1,480,780` (`25.64%`)
- EDGAR rows: `1,325,506`
- unique core CIKs: `7,606`
- unique EDGAR CIKs: `7,605`
- core/EDGAR CIK overlap rate: `99.99%`

Most importantly, on the current frozen daily surface:

- exact-day join match rate on rows with CIK: `99.6916%`
- benchmark-style backward `merge_asof(..., tolerance=90D)` match rate on rows
  with CIK: `99.6916%`
- coverage gain from as-of over exact-day: `0`
- lag distribution among matched rows: mean=`0`, median=`0`, p90=`0`, max=`0`

So the current freeze-backed EDGAR table is already daily aligned strongly
enough that exact-day join and benchmark as-of join are functionally identical
for this audit.

Target-side row coverage on rows with CIK is also already very high:

- `funding_raised_usd`: `99.3983%`
- `investors_count`: `99.7627%`
- `is_funded`: `99.2518%`

### EDGAR Root-Cause Reading

This rules out one previously plausible explanation:

- the current `core_edgar` weakness is **not** primarily caused by a mismatch
  between exact-day local join semantics and benchmark-style backward as-of
  join semantics

What remains as the more plausible explanation is:

1. **limited source reach**
   only `25.64%` of core rows carry a CIK, so EDGAR can only influence a
   minority of the panel at all
2. **target-dependent usefulness**
   EDGAR is not uniformly helpful across `is_funded`, `funding_raised_usd`, and
   `investors_count`
3. **model-path mismatch**
   V740 is still not exploiting the EDGAR signal with the right target-aware
   inductive bias, especially outside the already-live binary/funding niches

### EDGAR Operational Consequence

Do **not** spend the next implementation cycle rewriting V740 around a supposed
exact-vs-asof EDGAR join bug. That hypothesis is now falsified on the frozen
surface.

The next EDGAR-side work should instead focus on:

- source-native EDGAR event memory
- target-aware use of EDGAR signal
- explicit handling of the fact that ~`74%` of rows have no EDGAR path at all

## Text Audit: What Is Now Confirmed

The text path also yields a decisive separation between coverage questions and
representation questions.

Confirmed embedding lineage:

- base model: `Alibaba-NLP/gte-Qwen2-1.5B-instruct`
- raw dim: `1536`
- PCA dim: `64`
- explained variance after PCA: `0.7016`
- row count in embedding artifact: `5,774,931`
- unique texts embedded: `69,697`
- unique text ratio: `1.2069%`

Coverage is not the problem on the active text artifact:

- embedding rows from metadata / core rows parity: `1.0`
- streamed sample exact join rate on `20,000` embedding rows: `1.0`
- sampled target-side exact join rate: also `1.0`

So the active text artifact is present and aligned. The current text problem is
not “embeddings failed to generate” and not “text rows are missing from the
daily panel”.

### What the Sample Says About Text Signal Quality

The audit exposed three structural issues.

1. **extreme repetition / templating**
   only `69,697` unique combined texts map onto `5.77M` rows, so the text path
   is dominated by repeated or slowly changing descriptions rather than rich
   day-level semantic novelty

2. **many candidate text fields are sparse or effectively absent**
   on the streamed raw-text sample:
   - `company_description`: `0.0%` non-empty
   - `financial_forecasts`: `0.0%` non-empty
   - `offering_purpose`: `0.0%` non-empty
   - `financial_condition`: `0.6%` non-empty
   - `use_of_funds_text`: `1.0%` non-empty
   - `financial_risks`: `3.1%` non-empty
   - the denser fields are mainly `description_text` (`76.25%`), `headline`
     (`61.7%`), `title` (`59.9%`), and `reasons_to_invest_text` (`55.6%`)

3. **direct target association is weak for the two hardest practical tasks**
   on the `20,000`-row streamed embedding sample:
   - `investors_count`: top absolute Pearson correlation only `0.0508`, top
     absolute Spearman only `0.1136`
   - `is_funded`: top absolute Pearson only `0.0772`, top absolute Spearman
     only `0.0774`

The `funding_raised_usd` sample shows larger correlations on a few PCs, but
that sample has only `18` eligible rows in the streamed slice, so it is not
stable enough to justify a strong claim.

### Text Root-Cause Reading

The current evidence points away from a simple “Qwen embeddings are broken”
story.

The stronger current explanation is:

1. **representation compression is aggressive**
   `1536 -> 64` PCA keeps only ~`70%` variance
2. **the underlying text surface is highly repetitive**
   day-level rows do not imply day-level semantic novelty
3. **the informative fields are uneven and often sparse**
4. **the current benchmark path still feeds text as dense daily numeric columns**
   rather than as a sparse, source-native event stream

### Text Operational Consequence

Do **not** spend the next cycle claiming that text is absent or that the active
Qwen artifact silently failed. That is no longer true.

Do **not** jump straight to “replace Qwen” either. The current evidence is not
yet strong enough to blame the base encoder first.

The next text-side work should instead prioritize:

- changing the representation path before changing the encoder
- testing source-native sparse text-event memory instead of dense daily columns
- comparing the same base encoder at higher retained dimensionality before any
  costly encoder swap

## Updated Root-Cause Ranking

### `core_edgar`

1. source reach limitation (`cik` only on ~`25.6%` of rows)
2. target-specific semantic mismatch / usefulness
3. V740 mechanism mismatch on EDGAR-conditioned slices
4. exact-vs-asof join semantics

The last item is now explicitly demoted by direct audit evidence.

### text embeddings

1. dense-daily usage of what is actually a sparse semantic event source
2. repeated / low-novelty text surface
3. PCA compression to `64` dims
4. field sparsity / heterogeneous information quality
5. base encoder choice itself

The last item remains unproven and should not be treated as the primary cause
yet.

## Immediate Implementation Implications

1. EDGAR work should move toward source-native event memory and away from join
   debugging.
2. Text work should move toward representation redesign before encoder swap.
3. Funding / binary / investors must continue to be treated as separate target
   families inside one shared trunk; the source audits do not justify going
   back to one undifferentiated continuous recipe.
4. The next text experiment should compare:
   - current `gte-Qwen2-1.5B` at `64` dims
   - same encoder at higher retained dimensionality (`128` or `256`)
   - same encoder but source-native event-memory consumption
   before introducing a new encoder family.

## Bottom Line

The first real root-cause audit changes the burden of proof in two ways.

- `core_edgar` is **not** mainly failing because the join is misaligned.
- text is **not** mainly failing because embeddings are missing.

So the next V740 implementation wave should focus on **how** EDGAR/text are
represented and consumed, not on re-litigating whether the frozen assets exist
or whether exact-day join should secretly be backward as-of.