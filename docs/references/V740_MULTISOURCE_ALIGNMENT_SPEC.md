# V740 Multisource Alignment and Tokenization Specification

> Date: 2026-03-24
> Status: design-ready, pre-implementation
> Scope: `core` / `text` / `edgar` alignment and tokenization for V740
> Depends on: `docs/CURRENT_SOURCE_OF_TRUTH.md`, `docs/references/V740_DESIGN_SPECIFICATION.md`, `docs/references/V740_ALPHA_ENGINEERING_SPEC.md`

## 1. Purpose

This document defines the exact alignment and tokenization strategy for the
three data surfaces used in Block 3:

- `core`
- `text`
- `edgar`

The goal is to replace naive feature concatenation with a **dual-clock,
source-aware event representation** that is faithful to the real availability
structure of the data.

The central principle is:

> `core` is a regular state stream, while `text` and `edgar` are sparse,
> irregular event streams. They should not be represented in the same way.

## 1.1 Current freeze schema reality

The current frozen WIDE2 surface already gives enough structure to implement a
serious multisource path, but it also makes clear that the three sources are
not symmetric.

- `offers_core_daily`: **139 columns**
- `offers_text`: **19 columns**
- `edgar_store_full_daily`: **43 columns**

The actual join rules remain:

- `core` x `text`: `entity_id + crawled_date_day`
- `core` x `edgar`: `cik + crawled_date_day`

The EDGAR store is not an arbitrary feature soup. It is already organized as a
small family of semantically meaningful as-of features:

- `last_*`
- `mean_*`
- `ema_*`
- `*_is_missing`
- `edgar_has_filing`
- `edgar_valid`

This matters for V740 because it means the EDGAR path can start from a real,
auditable source schema rather than a guessed column pattern.

Text is different. The frozen `offers_text` table contains raw text fields such
as `headline`, `company_description`, `offering_purpose`, `updates_text`, and
`questions_text`, while the active benchmark surface uses downstream numeric
text embeddings (`text_emb_*`) for model consumption. In practice, V740-alpha
therefore treats the benchmark-available text embeddings as timestamped text
events, while preserving the broader design principle that text should still be
modeled as an event stream rather than as dense static tabular columns.

## 2. Why the Old Approach Is Not Good Enough

The current benchmark already shows that the existing text pathway is often
harmful and that EDGAR is mixed rather than uniformly helpful. This strongly
suggests that the current representation is too coarse.

The likely root problems are:

1. text and EDGAR are treated too much like ordinary tabular columns
2. event timing is not represented richly enough
3. recency and event density are mixed together without explicit control
4. source-specific availability and latency are not treated as first-class
   modeling variables
5. important sparse-event structure is lost when everything is collapsed into a
   single daily feature vector

## 3. The Dual-Clock Principle

V740 should use two clocks simultaneously.

### 3.1 State clock

The **state clock** is a regular daily grid used by `core`.
Each day contributes one state token.

This preserves:

- panel continuity
- smooth temporal dynamics
- stable decomposition and multi-resolution modeling

### 3.2 Event clock

The **event clock** is irregular and used for `text` and `edgar`.
Each event is represented relative to the prediction timestamp rather than forced
into a single dense daily row.

This preserves:

- exact event order
- true recency
- sparse-event density
- source-specific timing behavior

## 4. Source Roles

### 4.1 Core

`core` is the primary state stream.
It should remain on a regular daily grid.

### 4.2 EDGAR

`edgar` is primarily an event stream with occasional quasi-static carry-forward
state.

Two different things must be distinguished:

1. **filing events**
   - something happened on a date
   - type, amendment, novelty, and timing matter

2. **as-of EDGAR state**
   - latest known filing-derived values available at the current timestamp
   - can be summarized and carried into the state stream if needed

In the current freeze, the EDGAR input available to the model is already
daily-aligned and as-of filtered. That means alpha can treat rows with valid
EDGAR signals as **event proxies** even before a richer filing-event table is
introduced.

### 4.3 Text

`text` is also an event stream.
A text item is not just another numeric column. It is a timestamped semantic
update whose novelty, drift, and source matter.

In the current benchmark path, text reaches models through numeric PCA
embeddings (`text_emb_*`) that are attached to the daily panel after strict
alignment. Alpha should treat the presence of those embedding rows as a sparse
text-event signal, not as proof that text is natively dense.

## 5. Tokenization Strategy

The best representation for `edgar` and `text` is **hybrid tokenization**, not a
single scheme.

### 5.1 Why one tokenization scheme is not enough

- fixed-time windows preserve calendar recency, but smear rare event detail
- fixed-event bundles preserve local detail, but lose absolute time spacing

Our data need both.

### 5.2 Recommended hybrid tokenization

For each event source (`edgar`, `text`), build:

1. **recent raw event tokens**
2. **recency-bucket summary tokens**

This yields a compact but expressive event memory.

For the current alpha implementation, the practical interpretation is:

- use the joined daily panel as the authoritative as-of surface,
- detect rows where source-specific signals are present,
- construct recent-event and bucket-summary memories from those rows only,
- and keep the door open for a later upgrade to richer source-native event
  tables if they are added.

## 6. Core Token Design

Each daily `core` token should contain:

1. target history value(s)
2. selected numeric covariates available under the active ablation
3. missingness mask
4. first-order deltas / pct changes where stable
5. time features
   - day index
   - age since first observation
   - day-of-week or other periodic marker if useful
6. compact static summary interaction

Recommended state sequence lengths:

- alpha default: `60`
- ablation candidates: `96`, `128`

## 7. EDGAR Token Design

### 7.1 Raw event tokens

For the most recent `K_edgar` events, create one token per event.

Recommended initial budget:

- `K_edgar = 4` or `8`

Each EDGAR event token should include:

1. source id = `edgar`
2. filing type / form type embedding
3. event timestamp
4. `time_since_event`
5. `availability_lag`
6. amendment flag
7. event novelty score
8. key numeric payload projection
9. missing / quality mask
10. optional learned embedding of filing metadata bucketized into a compact code

### 7.2 Recency-bucket summary tokens

In addition, build EDGAR summary tokens over fixed recency buckets.

Recommended bucket set:

- `0–1d`
- `2–7d`
- `8–30d`
- `31–90d`
- `>90d`

Each bucket token should contain:

1. number of filings
2. filing-type histogram or learned compressed projection
3. most recent event age in bucket
4. aggregated numeric deltas
5. novelty summary (`mean`, `max`)
6. no-event flag
7. source reliability / coverage flag

## 8. Text Token Design

### 8.1 Raw event tokens

For the most recent `K_text` text updates, create one token per event.

Recommended initial budget:

- `K_text = 4` or `8`

Each text event token should include:

1. source id = `text`
2. text type / source subtype embedding
3. event timestamp
4. `time_since_event`
5. `availability_lag`
6. dense text embedding projection
7. novelty vs previous text for the same entity
8. semantic drift score
9. text length or confidence feature
10. missing / quality mask

### 8.2 Recency-bucket summary tokens

Use the same recency buckets as EDGAR where practical:

- `0–1d`
- `2–7d`
- `8–30d`
- `31–90d`
- `>90d`

Each bucket token should contain:

1. number of text updates
2. mean pooled embedding
3. novelty summary
4. semantic drift summary
5. presence / no-event flag
6. source quality / availability mask

## 9. Availability and Leakage Rules

This is a hard constraint area.

### 9.1 General rule

For every prediction timestamp `t_pred`, only information with
`availability_time <= t_pred` may be used.

### 9.2 Core

Core state is aligned directly on the daily observation grid.

### 9.3 EDGAR

For EDGAR, do **not** use filing time alone when availability is delayed.
Define:

- `event_time`
- `availability_time`
- `latency = availability_time - event_time`

The model should consume the event only if:

- `availability_time <= t_pred`

### 9.4 Text

For text, define the same pair:

- `text_time`
- `availability_time`

This is especially important when text is crawled or observed later than the
original source timestamp.

### 9.5 Same-day duplicates

If multiple events from the same source occur within the same day, do not
collapse them blindly.

Keep:

- exact order for raw event tokens
- aggregated summaries for recency buckets

## 10. Alignment Output Structure

For each prediction instance, construct three inputs:

1. `X_core`: regular daily state sequence
2. `E_edgar`: EDGAR event-memory tokens
3. `E_text`: text event-memory tokens

Optionally also construct:

4. `S_static`: compact static summary vector

This yields a clean separation:

- state stream
- sparse event memory
- static context

## 11. Fusion Strategy inside V740

### 11.1 Core as the trunk

The `core` stream should remain the main trunk.

### 11.2 Event streams as conditional memory

`edgar` and `text` should act as conditional event memory rather than as dense
co-equal tabular features.

### 11.3 Recommended fusion stack

1. encode `X_core` with decomposition + multi-resolution trunk
2. encode `E_edgar` with lightweight event encoder
3. encode `E_text` with lightweight event encoder
4. fuse event memories into the trunk with gated cross-attention or FiLM-style
   modulation
5. expose the fused representation to target-specific heads

This preserves the idea that `core` is the state backbone while `edgar` and
`text` provide sparse but important regime shifts.

## 12. Value-Space and Event-Space Complementarity

For V740, the event-token design should be paired with a value-space auxiliary
branch for the target itself.

Why:

- `Chronos`-like strengths come partly from scale-insensitive value modeling
- event tokens explain *why* the regime changed
- value-space modeling helps the model remain robust to heavy-tailed or shifted
  numeric targets

The two should coexist.

## 13. Audit Checklist

Any implementation of this design must be audited against the following.

### 13.1 Leakage audit

1. no event used before its availability time
2. no bucket summary accidentally includes future events
3. same-day duplicate ordering preserved for raw event tokens

### 13.2 Sparsity audit

1. no-event cases represented explicitly
2. masks survive preprocessing and batching
3. sparse entities are not silently dropped from evaluation unless the benchmark
   protocol explicitly requires it

### 13.3 Token integrity audit

1. raw-event token budget respected (`K_edgar`, `K_text`)
2. recency-bucket summaries correctly computed
3. novelty and drift features are entity-local and time-local

## 14. Initial Hyperparameter Defaults

A good initial implementation target is:

- `L_core = 60`
- `K_edgar = 4`
- `K_text = 4`
- recency buckets = `[0-1d, 2-7d, 8-30d, 31-90d, >90d]`
- event token hidden dim = small (`16-64` range initially)
- source-specific missing masks always present

## 15. Immediate Implementation Plan

### Phase A

1. build freeze-backed feature extraction that produces:
   - `X_core`
   - `E_edgar`
   - `E_text`
   - `S_static`
2. verify strict as-of alignment on a few entities manually

### Phase B

1. plug event encoders into `V740AlphaPrototypeWrapper`
2. keep text branch optional at first
3. benchmark `core` vs `core+edgar-event` before enabling `text`

### Phase C

1. add text event branch
2. add recency-bucket ablations
3. test whether raw-event-only, bucket-only, or hybrid tokenization is best on
   Block 3

## 16. Bottom Line

The right design is not:

- daily-core plus a pile of text columns
- daily-core plus a pile of EDGAR columns

The right design is:

- `core` as a regular state stream
- `edgar` as a sparse event stream with raw-event and recency-bucket tokens
- `text` as a sparse semantic event stream with raw-event and recency-bucket tokens
- strict availability-aware as-of alignment
- event memory fused into a single condition-aware forecasting model

That is the cleanest and most powerful way to let V740 exploit `core`, `text`,
and `edgar` without collapsing their very different temporal semantics.
