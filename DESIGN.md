# Wide Freeze Design

## Data Products
1. **Wide Snapshot**
   - Source: `data/raw/offers` (Delta)
   - Output: `runs/offers_core_full_snapshot_wide_${STAMP}/offers_core_snapshot.parquet`
   - Contains: all wide scalar columns + keys + time columns
   - Nested columns are not dropped; they are routed to `offers_extras_snapshot` as:
     - `<col>__json`, `<col>__len`, `<col>__hash`

2. **Wide Daily**
   - Source: snapshot
   - Output: `runs/offers_core_full_daily_wide_${STAMP}/offers_core_daily.parquet`
   - Aggregation: last-non-null per `entity_id`/day; derived features (e.g., delta, pct_change)
   - Extras: `offers_extras_daily.parquet` with nested-derived fields

3. **Snapshots Index**
   - Output:
     - `snapshots_offer_day.parquet` (offer/day grain)
     - `snapshots_cik_day.parquet` (cik/day grain for EDGAR)

4. **EDGAR Store (Wide)**
   - Output: `runs/edgar_feature_store_full_daily_wide_${STAMP}/edgar_features/**`
   - Aligns EDGAR accessions to snapshots (cik/day)

5. **Multiscale Views**
   - Output: weekly/monthly/stage parquets under `runs/multiscale_full_wide_${STAMP}`

## Wide Contract Logic
- Derived from raw inventory with coverage threshold (`non_null >= 0.001`).
- Unknown stats default to keep.
- Categorical columns with extreme cardinality may be retained as derived-only.
- Nested columns are frozen via JSON/length/hash in extras.

## Manifest Requirements
Every data product directory must include `MANIFEST.json` with:
- raw delta version, active_files
- rows_scanned / rows_emitted
- date range
- command args, git hash, built_at
- grain, partition strategy
- output schema / columns
- dropped_columns with reasons

## Audit Design
1. **Column Manifest**
   - Contract-driven checks (must_keep + derived_structured + derived_nested)
   - EDGAR recompute value-level check (`total_compared >= 200`, `diff_count == 0`)
   - No fallback store allowed for PASS

2. **Raw Cardinality Coverage**
   - Full scan (`raw_scan_limit=0`)
   - Raw entity-day totals and structured-signal totals
   - Structured-signal coverage by `offers_core_daily`
   - Text coverage by key overlap (entity_id + day)
   - Pair-level snapshots→edgar coverage
   - Debug samples in `analysis/**/debug`

## Single Source of Truth
`docs/audits/FULL_SCALE_POINTER.yaml` is the only approved input for Block 3.  
All audits and summaries must reference the pointer paths.

## Modeling Tasks Enabled
- Task-1 Outcome Prediction: wide static + early K snapshots + EDGAR + narrative indices
- Task-2 Trajectory Forecasting: multi-step time series from early K snapshots
- Task-3 EDGAR-conditioned Risk Adjustment: ablation with vs without EDGAR features
- Task-4 Narrative Shift: NBI/NCI indices and post-2023 effect analysis
- Hierarchical AutoFit: multi-scale encoding with cross-scale conditioning
