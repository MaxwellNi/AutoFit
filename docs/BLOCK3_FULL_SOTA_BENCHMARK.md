# Block 3 Full SOTA Benchmark (Strict Comparable)

> Generated UTC: 2026-02-26T17:26:24.569069+00:00
> Strict comparable completion: `[########################] 104/104 (100.0%)`

## Summary

| Metric | Value | Evidence |
|---|---:|---|
| strict_completed_conditions | 104/104 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| table_rows | 104 | `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv` |
| champion_family_distribution | deep_classical=62, foundation=6, transformer_sota=36 | `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv` |
| top_champion_models | NBEATS=39, PatchTST=24, NHITS=23, KAN=7, Chronos=6, NBEATSx=4, DLinear=1 | `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv` |

## Full 104-condition Champion Table

| task | ablation | target | horizon | champion_model | champion_family | champion_mae | best_non_autofit_model | best_non_autofit_mae | best_autofit_model | best_autofit_mae | autofit_gap_pct |
|---|---|---|---:|---|---|---:|---|---:|---|---:|---:|
| task1_outcome | core_edgar | funding_raised_usd | 1 | NBEATS | deep_classical | 374,514.684 | NBEATS | 374,514.684 | AutoFitV3 | 395,551.536 | +5.62% |
| task1_outcome | core_edgar | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV3 | 395,551.536 | +5.64% |
| task1_outcome | core_edgar | funding_raised_usd | 14 | PatchTST | transformer_sota | 375,055.785 | PatchTST | 375,055.785 | AutoFitV3 | 395,551.536 | +5.46% |
| task1_outcome | core_edgar | funding_raised_usd | 30 | PatchTST | transformer_sota | 375,472.395 | PatchTST | 375,472.395 | AutoFitV3Max | 395,551.536 | +5.35% |
| task1_outcome | core_edgar | investors_count | 1 | NHITS | deep_classical | 44.836898 | NHITS | 44.836898 | AutoFitV1 | 125.572330 | +180.06% |
| task1_outcome | core_edgar | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV1 | 125.468632 | +180.12% |
| task1_outcome | core_edgar | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV1 | 125.468632 | +180.07% |
| task1_outcome | core_edgar | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV1 | 125.468632 | +179.99% |
| task1_outcome | core_edgar | is_funded | 1 | PatchTST | transformer_sota | 0.032367 | PatchTST | 0.032367 | AutoFitV72 | 0.087164 | +169.30% |
| task1_outcome | core_edgar | is_funded | 7 | NHITS | deep_classical | 0.032383 | NHITS | 0.032383 | AutoFitV72 | 0.087164 | +169.17% |
| task1_outcome | core_edgar | is_funded | 14 | PatchTST | transformer_sota | 0.032355 | PatchTST | 0.032355 | AutoFitV72 | 0.087164 | +169.40% |
| task1_outcome | core_edgar | is_funded | 30 | NHITS | deep_classical | 0.032322 | NHITS | 0.032322 | AutoFitV72 | 0.087164 | +169.67% |
| task1_outcome | core_only | funding_raised_usd | 1 | NBEATS | deep_classical | 380,659.460 | NBEATS | 380,659.460 | AutoFitV71 | 398,329.873 | +4.64% |
| task1_outcome | core_only | funding_raised_usd | 7 | NHITS | deep_classical | 380,577.133 | NHITS | 380,577.133 | AutoFitV71 | 398,329.873 | +4.66% |
| task1_outcome | core_only | funding_raised_usd | 14 | PatchTST | transformer_sota | 381,200.561 | PatchTST | 381,200.561 | AutoFitV71 | 398,329.873 | +4.49% |
| task1_outcome | core_only | funding_raised_usd | 30 | PatchTST | transformer_sota | 381,617.171 | PatchTST | 381,617.171 | AutoFitV71 | 398,329.873 | +4.38% |
| task1_outcome | core_only | investors_count | 1 | KAN | transformer_sota | 44.703800 | KAN | 44.703800 | AutoFitV72 | 257.790005 | +476.66% |
| task1_outcome | core_only | investors_count | 7 | KAN | transformer_sota | 44.692755 | KAN | 44.692755 | AutoFitV72 | 257.790005 | +476.80% |
| task1_outcome | core_only | investors_count | 14 | NBEATS | deep_classical | 44.734036 | NBEATS | 44.734036 | AutoFitV72 | 257.790005 | +476.27% |
| task1_outcome | core_only | investors_count | 30 | NBEATS | deep_classical | 44.746757 | NBEATS | 44.746757 | AutoFitV72 | 257.790005 | +476.11% |
| task1_outcome | core_only | is_funded | 1 | PatchTST | transformer_sota | 0.033084 | PatchTST | 0.033084 | AutoFitV71 | 0.091689 | +177.14% |
| task1_outcome | core_only | is_funded | 7 | NHITS | deep_classical | 0.033100 | NHITS | 0.033100 | AutoFitV71 | 0.091690 | +177.01% |
| task1_outcome | core_only | is_funded | 14 | PatchTST | transformer_sota | 0.033072 | PatchTST | 0.033072 | AutoFitV71 | 0.092128 | +178.57% |
| task1_outcome | core_only | is_funded | 30 | NBEATSx | transformer_sota | 0.033033 | NBEATSx | 0.033033 | AutoFitV71 | 0.092128 | +178.90% |
| task1_outcome | core_text | funding_raised_usd | 1 | NBEATS | deep_classical | 380,659.460 | NBEATS | 380,659.460 | AutoFitV7 | 399,671.899 | +4.99% |
| task1_outcome | core_text | funding_raised_usd | 7 | NHITS | deep_classical | 380,577.133 | NHITS | 380,577.133 | AutoFitV7 | 399,671.899 | +5.02% |
| task1_outcome | core_text | funding_raised_usd | 14 | PatchTST | transformer_sota | 381,200.561 | PatchTST | 381,200.561 | AutoFitV7 | 399,671.899 | +4.85% |
| task1_outcome | core_text | funding_raised_usd | 30 | PatchTST | transformer_sota | 381,617.171 | PatchTST | 381,617.171 | AutoFitV7 | 399,671.899 | +4.73% |
| task1_outcome | core_text | investors_count | 1 | KAN | transformer_sota | 44.703800 | KAN | 44.703800 | AutoFitV7 | 279.931714 | +526.19% |
| task1_outcome | core_text | investors_count | 7 | KAN | transformer_sota | 44.692755 | KAN | 44.692755 | AutoFitV7 | 279.931656 | +526.35% |
| task1_outcome | core_text | investors_count | 14 | NBEATS | deep_classical | 44.734036 | NBEATS | 44.734036 | AutoFitV7 | 279.931643 | +525.77% |
| task1_outcome | core_text | investors_count | 30 | NBEATS | deep_classical | 44.746757 | NBEATS | 44.746757 | AutoFitV7 | 280.703712 | +527.32% |
| task1_outcome | core_text | is_funded | 1 | PatchTST | transformer_sota | 0.033084 | PatchTST | 0.033084 | AutoFitV71 | 0.086412 | +161.19% |
| task1_outcome | core_text | is_funded | 7 | NHITS | deep_classical | 0.033094 | NHITS | 0.033094 | AutoFitV71 | 0.086412 | +161.11% |
| task1_outcome | core_text | is_funded | 14 | PatchTST | transformer_sota | 0.033072 | PatchTST | 0.033072 | AutoFitV71 | 0.086412 | +161.29% |
| task1_outcome | core_text | is_funded | 30 | NHITS | deep_classical | 0.033034 | NHITS | 0.033034 | AutoFitV71 | 0.086412 | +161.59% |
| task1_outcome | full | funding_raised_usd | 1 | NBEATSx | transformer_sota | 374,514.684 | NBEATSx | 374,514.684 | AutoFitV1 | 396,360.349 | +5.83% |
| task1_outcome | full | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV1 | 396,360.349 | +5.86% |
| task1_outcome | full | funding_raised_usd | 14 | Chronos | foundation | 374,687.533 | Chronos | 374,687.533 | AutoFitV1 | 396,360.349 | +5.78% |
| task1_outcome | full | funding_raised_usd | 30 | Chronos | foundation | 374,610.314 | Chronos | 374,610.314 | AutoFitV1 | 396,360.349 | +5.81% |
| task1_outcome | full | investors_count | 1 | KAN | transformer_sota | 44.809991 | KAN | 44.809991 | AutoFitV1 | 125.468632 | +180.00% |
| task1_outcome | full | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV1 | 125.468632 | +180.12% |
| task1_outcome | full | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV1 | 125.468632 | +180.07% |
| task1_outcome | full | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV1 | 125.468632 | +179.99% |
| task1_outcome | full | is_funded | 1 | PatchTST | transformer_sota | 0.032294 | PatchTST | 0.032294 | AutoFitV71 | 0.087164 | +169.91% |
| task1_outcome | full | is_funded | 7 | DLinear | transformer_sota | 0.032379 | DLinear | 0.032379 | AutoFitV71 | 0.087164 | +169.20% |
| task1_outcome | full | is_funded | 14 | PatchTST | transformer_sota | 0.032281 | PatchTST | 0.032281 | AutoFitV71 | 0.087164 | +170.02% |
| task1_outcome | full | is_funded | 30 | NHITS | deep_classical | 0.032322 | NHITS | 0.032322 | AutoFitV71 | 0.087164 | +169.67% |
| task2_forecast | core_edgar | funding_raised_usd | 1 | NBEATS | deep_classical | 374,514.684 | NBEATS | 374,514.684 | AutoFitV3 | 395,551.536 | +5.62% |
| task2_forecast | core_edgar | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV3 | 395,551.536 | +5.64% |
| task2_forecast | core_edgar | funding_raised_usd | 14 | PatchTST | transformer_sota | 375,055.785 | PatchTST | 375,055.785 | AutoFitV3 | 395,551.536 | +5.46% |
| task2_forecast | core_edgar | funding_raised_usd | 30 | PatchTST | transformer_sota | 375,472.395 | PatchTST | 375,472.395 | AutoFitV3 | 395,551.536 | +5.35% |
| task2_forecast | core_edgar | investors_count | 1 | NHITS | deep_classical | 44.836898 | NHITS | 44.836898 | AutoFitV3Max | 137.261891 | +206.14% |
| task2_forecast | core_edgar | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV3Max | 137.261891 | +206.45% |
| task2_forecast | core_edgar | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV3Max | 137.261891 | +206.40% |
| task2_forecast | core_edgar | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV3Max | 137.261891 | +206.31% |
| task2_forecast | core_only | funding_raised_usd | 1 | NBEATS | deep_classical | 380,659.460 | NBEATS | 380,659.460 | AutoFitV7 | 399,671.899 | +4.99% |
| task2_forecast | core_only | funding_raised_usd | 7 | NHITS | deep_classical | 380,577.133 | NHITS | 380,577.133 | AutoFitV7 | 399,671.899 | +5.02% |
| task2_forecast | core_only | funding_raised_usd | 14 | PatchTST | transformer_sota | 381,200.561 | PatchTST | 381,200.561 | AutoFitV7 | 399,671.899 | +4.85% |
| task2_forecast | core_only | funding_raised_usd | 30 | PatchTST | transformer_sota | 381,617.171 | PatchTST | 381,617.171 | AutoFitV7 | 399,671.899 | +4.73% |
| task2_forecast | core_only | investors_count | 1 | NHITS | deep_classical | 44.771955 | NHITS | 44.771955 | AutoFitV7 | 279.931794 | +525.24% |
| task2_forecast | core_only | investors_count | 7 | NBEATS | deep_classical | 44.726689 | NBEATS | 44.726689 | AutoFitV7 | 279.931622 | +525.87% |
| task2_forecast | core_only | investors_count | 14 | NBEATS | deep_classical | 44.734036 | NBEATS | 44.734036 | AutoFitV7 | 279.931805 | +525.77% |
| task2_forecast | core_only | investors_count | 30 | NBEATS | deep_classical | 44.746757 | NBEATS | 44.746757 | AutoFitV7 | 279.931644 | +525.59% |
| task2_forecast | core_text | funding_raised_usd | 1 | NBEATS | deep_classical | 380,659.460 | NBEATS | 380,659.460 | AutoFitV7 | 399,671.899 | +4.99% |
| task2_forecast | core_text | funding_raised_usd | 7 | NHITS | deep_classical | 380,577.133 | NHITS | 380,577.133 | AutoFitV7 | 399,671.899 | +5.02% |
| task2_forecast | core_text | funding_raised_usd | 14 | PatchTST | transformer_sota | 381,200.561 | PatchTST | 381,200.561 | AutoFitV7 | 399,671.899 | +4.85% |
| task2_forecast | core_text | funding_raised_usd | 30 | PatchTST | transformer_sota | 381,617.171 | PatchTST | 381,617.171 | AutoFitV7 | 399,671.899 | +4.73% |
| task2_forecast | core_text | investors_count | 1 | NHITS | deep_classical | 44.771955 | NHITS | 44.771955 | AutoFitV6 | 114.116541 | +154.88% |
| task2_forecast | core_text | investors_count | 7 | NBEATS | deep_classical | 44.726689 | NBEATS | 44.726689 | AutoFitV6 | 114.116541 | +155.14% |
| task2_forecast | core_text | investors_count | 14 | NBEATS | deep_classical | 44.734036 | NBEATS | 44.734036 | AutoFitV6 | 114.116541 | +155.10% |
| task2_forecast | core_text | investors_count | 30 | NBEATS | deep_classical | 44.746757 | NBEATS | 44.746757 | AutoFitV6 | 114.116541 | +155.03% |
| task2_forecast | full | funding_raised_usd | 1 | NBEATSx | transformer_sota | 374,514.684 | NBEATSx | 374,514.684 | AutoFitV3 | 396,360.349 | +5.83% |
| task2_forecast | full | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV1 | 396,360.349 | +5.86% |
| task2_forecast | full | funding_raised_usd | 14 | Chronos | foundation | 374,687.533 | Chronos | 374,687.533 | AutoFitV3 | 396,360.349 | +5.78% |
| task2_forecast | full | funding_raised_usd | 30 | Chronos | foundation | 374,610.314 | Chronos | 374,610.314 | AutoFitV1 | 396,360.349 | +5.81% |
| task2_forecast | full | investors_count | 1 | KAN | transformer_sota | 44.809991 | KAN | 44.809991 | AutoFitV1 | 125.468632 | +180.00% |
| task2_forecast | full | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV1 | 125.468632 | +180.12% |
| task2_forecast | full | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV1 | 125.468632 | +180.07% |
| task2_forecast | full | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV1 | 125.468632 | +179.99% |
| task3_risk_adjust | core_edgar | funding_raised_usd | 1 | NBEATS | deep_classical | 374,514.684 | NBEATS | 374,514.684 | AutoFitV3 | 395,551.536 | +5.62% |
| task3_risk_adjust | core_edgar | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV3 | 395,551.536 | +5.64% |
| task3_risk_adjust | core_edgar | funding_raised_usd | 14 | PatchTST | transformer_sota | 375,055.785 | PatchTST | 375,055.785 | AutoFitV3 | 395,551.536 | +5.46% |
| task3_risk_adjust | core_edgar | funding_raised_usd | 30 | PatchTST | transformer_sota | 375,472.395 | PatchTST | 375,472.395 | AutoFitV3 | 395,551.536 | +5.35% |
| task3_risk_adjust | core_edgar | investors_count | 1 | NHITS | deep_classical | 44.836898 | NHITS | 44.836898 | AutoFitV1 | 125.572330 | +180.06% |
| task3_risk_adjust | core_edgar | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV1 | 125.572330 | +180.35% |
| task3_risk_adjust | core_edgar | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV1 | 125.572330 | +180.30% |
| task3_risk_adjust | core_edgar | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV1 | 125.572330 | +180.22% |
| task3_risk_adjust | core_only | funding_raised_usd | 1 | NBEATS | deep_classical | 380,659.460 | NBEATS | 380,659.460 | AutoFitV7 | 399,671.899 | +4.99% |
| task3_risk_adjust | core_only | funding_raised_usd | 7 | NHITS | deep_classical | 380,577.133 | NHITS | 380,577.133 | AutoFitV7 | 399,671.899 | +5.02% |
| task3_risk_adjust | core_only | funding_raised_usd | 14 | PatchTST | transformer_sota | 381,200.561 | PatchTST | 381,200.561 | AutoFitV7 | 399,671.899 | +4.85% |
| task3_risk_adjust | core_only | funding_raised_usd | 30 | PatchTST | transformer_sota | 381,616.832 | PatchTST | 381,616.832 | AutoFitV7 | 399,671.899 | +4.73% |
| task3_risk_adjust | core_only | investors_count | 1 | NHITS | deep_classical | 44.771955 | NHITS | 44.771955 | AutoFitV7 | 279.931659 | +525.24% |
| task3_risk_adjust | core_only | investors_count | 7 | NBEATS | deep_classical | 44.726689 | NBEATS | 44.726689 | AutoFitV7 | 279.931673 | +525.87% |
| task3_risk_adjust | core_only | investors_count | 14 | NBEATS | deep_classical | 44.734036 | NBEATS | 44.734036 | AutoFitV7 | 280.703021 | +527.49% |
| task3_risk_adjust | core_only | investors_count | 30 | NBEATS | deep_classical | 44.746757 | NBEATS | 44.746757 | AutoFitV7 | 279.931659 | +525.59% |
| task3_risk_adjust | full | funding_raised_usd | 1 | NBEATSx | transformer_sota | 374,514.684 | NBEATSx | 374,514.684 | AutoFitV1 | 396,360.349 | +5.83% |
| task3_risk_adjust | full | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV1 | 396,360.349 | +5.86% |
| task3_risk_adjust | full | funding_raised_usd | 14 | Chronos | foundation | 374,687.533 | Chronos | 374,687.533 | AutoFitV1 | 396,360.349 | +5.78% |
| task3_risk_adjust | full | funding_raised_usd | 30 | Chronos | foundation | 374,610.314 | Chronos | 374,610.314 | AutoFitV1 | 396,360.349 | +5.81% |
| task3_risk_adjust | full | investors_count | 1 | KAN | transformer_sota | 44.809991 | KAN | 44.809991 | AutoFitV1 | 125.468632 | +180.00% |
| task3_risk_adjust | full | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV1 | 125.468632 | +180.12% |
| task3_risk_adjust | full | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV1 | 125.468632 | +180.07% |
| task3_risk_adjust | full | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV1 | 125.468632 | +179.99% |

Source table: `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv`
