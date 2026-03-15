# Column Contract Wide (20260203_031917)

Wide-table principle: non_null >= 0.001, std=0 still keep, categorical distinct <= 200k.

## offers_core_snapshot / offers_core_daily
- must_keep: 150 columns
- high_value (std>0): 0
- derived_only (text len/tokens): ['text_len', 'num_tokens_est', 'num_urls', 'num_hashtags']
- nested_columns: 17
- derived_nested: 51

## edgar_store
- must_keep: 45 columns

## vs v3
- v3 coverage_min=0.05; wide uses 0.001
- Arrays/text: nested raw not in core; derived __json/__len/__hash are frozen