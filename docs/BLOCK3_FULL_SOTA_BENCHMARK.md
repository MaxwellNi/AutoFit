# Block 3 Full SOTA Benchmark (Strict Comparable)

> Generated UTC: 2026-03-02T23:11:37.007139+00:00
> Strict comparable completion: `[########################] 104/104 (100.0%)`

## Summary

| Metric | Value | Evidence |
|---|---:|---|
| strict_completed_conditions | 104/104 | `docs/benchmarks/block3_truth_pack/truth_pack_summary.json` |
| table_rows | 104 | `docs/benchmarks/block3_truth_pack/full_sota_104_table.csv` |
| champion_family_distribution | deep_classical=56, transformer_sota=26, foundation=22 | `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv` |
| top_champion_models | NBEATS=41, Chronos=22, NHITS=15, KAN=10, DeepNPTS=8, PatchTST=4, NBEATSx=3, DLinear=1 | `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv` |

## Full 104-condition Champion Table

| task | ablation | target | horizon | champion_model | champion_family | champion_mae | best_non_autofit_model | best_non_autofit_mae | best_autofit_model | best_autofit_mae | autofit_gap_pct |
|---|---|---|---:|---|---|---:|---|---:|---|---:|---:|
| task1_outcome | core_edgar | funding_raised_usd | 1 | NBEATS | deep_classical | 374,514.684 | NBEATS | 374,514.684 | AutoFitV3 | 395,551.536 | +5.62% |
| task1_outcome | core_edgar | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV3 | 395,551.536 | +5.64% |
| task1_outcome | core_edgar | funding_raised_usd | 14 | Chronos | foundation | 374,687.533 | Chronos | 374,687.533 | AutoFitV3 | 395,551.536 | +5.57% |
| task1_outcome | core_edgar | funding_raised_usd | 30 | Chronos | foundation | 374,610.314 | Chronos | 374,610.314 | AutoFitV3Max | 395,551.536 | +5.59% |
| task1_outcome | core_edgar | investors_count | 1 | NHITS | deep_classical | 44.836898 | NHITS | 44.836898 | AutoFitV1 | 125.572330 | +180.06% |
| task1_outcome | core_edgar | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV1 | 125.468632 | +180.12% |
| task1_outcome | core_edgar | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV1 | 125.468632 | +180.07% |
| task1_outcome | core_edgar | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV1 | 125.468632 | +179.99% |
| task1_outcome | core_edgar | is_funded | 1 | PatchTST | transformer_sota | 0.032367 | PatchTST | 0.032367 | AutoFitV2 | 0.089394 | +176.19% |
| task1_outcome | core_edgar | is_funded | 7 | NHITS | deep_classical | 0.032383 | NHITS | 0.032383 | AutoFitV2 | 0.089394 | +176.06% |
| task1_outcome | core_edgar | is_funded | 14 | PatchTST | transformer_sota | 0.032355 | PatchTST | 0.032355 | AutoFitV2 | 0.089394 | +176.30% |
| task1_outcome | core_edgar | is_funded | 30 | NHITS | deep_classical | 0.032322 | NHITS | 0.032322 | AutoFitV2 | 0.089394 | +176.57% |
| task1_outcome | core_only | funding_raised_usd | 1 | NBEATS | deep_classical | 380,659.460 | NBEATS | 380,659.460 | AutoFitV7 | 397,244.478 | +4.36% |
| task1_outcome | core_only | funding_raised_usd | 7 | NHITS | deep_classical | 380,577.133 | NHITS | 380,577.133 | AutoFitV7 | 397,244.478 | +4.38% |
| task1_outcome | core_only | funding_raised_usd | 14 | Chronos | foundation | 380,832.309 | Chronos | 380,832.309 | AutoFitV7 | 397,244.478 | +4.31% |
| task1_outcome | core_only | funding_raised_usd | 30 | Chronos | foundation | 380,755.090 | Chronos | 380,755.090 | AutoFitV7 | 397,244.478 | +4.33% |
| task1_outcome | core_only | investors_count | 1 | KAN | transformer_sota | 44.745049 | KAN | 44.745049 | AutoFitV3Max | 113.911783 | +154.58% |
| task1_outcome | core_only | investors_count | 7 | NBEATS | deep_classical | 44.726689 | NBEATS | 44.726689 | AutoFitV3Max | 113.911783 | +154.68% |
| task1_outcome | core_only | investors_count | 14 | NBEATS | deep_classical | 44.734036 | NBEATS | 44.734036 | AutoFitV3Max | 113.911783 | +154.64% |
| task1_outcome | core_only | investors_count | 30 | NBEATS | deep_classical | 44.746757 | NBEATS | 44.746757 | AutoFitV6 | 113.911783 | +154.57% |
| task1_outcome | core_only | is_funded | 1 | DeepNPTS | transformer_sota | 0.032956 | DeepNPTS | 0.032956 | AutoFitV4 | 0.088906 | +169.77% |
| task1_outcome | core_only | is_funded | 7 | DeepNPTS | transformer_sota | 0.032957 | DeepNPTS | 0.032957 | AutoFitV4 | 0.088906 | +169.76% |
| task1_outcome | core_only | is_funded | 14 | DeepNPTS | transformer_sota | 0.032954 | DeepNPTS | 0.032954 | AutoFitV4 | 0.088906 | +169.79% |
| task1_outcome | core_only | is_funded | 30 | DeepNPTS | transformer_sota | 0.032958 | DeepNPTS | 0.032958 | AutoFitV4 | 0.088906 | +169.75% |
| task1_outcome | core_text | funding_raised_usd | 1 | NBEATS | deep_classical | 380,659.121 | NBEATS | 380,659.121 | AutoFitV7 | 397,244.478 | +4.36% |
| task1_outcome | core_text | funding_raised_usd | 7 | NHITS | deep_classical | 380,577.133 | NHITS | 380,577.133 | AutoFitV7 | 397,244.478 | +4.38% |
| task1_outcome | core_text | funding_raised_usd | 14 | Chronos | foundation | 380,832.309 | Chronos | 380,832.309 | AutoFitV7 | 397,244.478 | +4.31% |
| task1_outcome | core_text | funding_raised_usd | 30 | Chronos | foundation | 380,755.090 | Chronos | 380,755.090 | AutoFitV7 | 397,244.478 | +4.33% |
| task1_outcome | core_text | investors_count | 1 | KAN | transformer_sota | 44.745049 | KAN | 44.745049 | AutoFitV6 | 113.911783 | +154.58% |
| task1_outcome | core_text | investors_count | 7 | NBEATS | deep_classical | 44.726689 | NBEATS | 44.726689 | AutoFitV3Max | 113.911783 | +154.68% |
| task1_outcome | core_text | investors_count | 14 | NBEATS | deep_classical | 44.734036 | NBEATS | 44.734036 | AutoFitV3Max | 113.911783 | +154.64% |
| task1_outcome | core_text | investors_count | 30 | NBEATS | deep_classical | 44.746757 | NBEATS | 44.746757 | AutoFitV3Max | 113.911783 | +154.57% |
| task1_outcome | core_text | is_funded | 1 | DeepNPTS | transformer_sota | 0.032956 | DeepNPTS | 0.032956 | AutoFitV4 | 0.088906 | +169.77% |
| task1_outcome | core_text | is_funded | 7 | DeepNPTS | transformer_sota | 0.032957 | DeepNPTS | 0.032957 | AutoFitV4 | 0.088906 | +169.76% |
| task1_outcome | core_text | is_funded | 14 | DeepNPTS | transformer_sota | 0.032954 | DeepNPTS | 0.032954 | AutoFitV4 | 0.088906 | +169.79% |
| task1_outcome | core_text | is_funded | 30 | DeepNPTS | transformer_sota | 0.032958 | DeepNPTS | 0.032958 | AutoFitV4 | 0.088906 | +169.75% |
| task1_outcome | full | funding_raised_usd | 1 | NBEATSx | transformer_sota | 374,514.684 | NBEATSx | 374,514.684 | AutoFitV1 | 396,360.349 | +5.83% |
| task1_outcome | full | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV5 | 396,360.349 | +5.86% |
| task1_outcome | full | funding_raised_usd | 14 | Chronos | foundation | 374,687.533 | Chronos | 374,687.533 | AutoFitV3Max | 396,360.349 | +5.78% |
| task1_outcome | full | funding_raised_usd | 30 | Chronos | foundation | 374,610.314 | Chronos | 374,610.314 | AutoFitV3Max | 396,360.349 | +5.81% |
| task1_outcome | full | investors_count | 1 | KAN | transformer_sota | 44.809991 | KAN | 44.809991 | AutoFitV1 | 125.468632 | +180.00% |
| task1_outcome | full | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV1 | 125.468632 | +180.12% |
| task1_outcome | full | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV1 | 125.468632 | +180.07% |
| task1_outcome | full | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV1 | 125.468632 | +179.99% |
| task1_outcome | full | is_funded | 1 | PatchTST | transformer_sota | 0.032294 | PatchTST | 0.032294 | AutoFitV2 | 0.089394 | +176.82% |
| task1_outcome | full | is_funded | 7 | DLinear | transformer_sota | 0.032379 | DLinear | 0.032379 | AutoFitV2 | 0.089394 | +176.09% |
| task1_outcome | full | is_funded | 14 | PatchTST | transformer_sota | 0.032281 | PatchTST | 0.032281 | AutoFitV2 | 0.089394 | +176.93% |
| task1_outcome | full | is_funded | 30 | NHITS | deep_classical | 0.032322 | NHITS | 0.032322 | AutoFitV2 | 0.089394 | +176.57% |
| task2_forecast | core_edgar | funding_raised_usd | 1 | NBEATS | deep_classical | 374,514.684 | NBEATS | 374,514.684 | AutoFitV3 | 395,551.536 | +5.62% |
| task2_forecast | core_edgar | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV3 | 395,551.536 | +5.64% |
| task2_forecast | core_edgar | funding_raised_usd | 14 | Chronos | foundation | 374,687.533 | Chronos | 374,687.533 | AutoFitV3 | 395,551.536 | +5.57% |
| task2_forecast | core_edgar | funding_raised_usd | 30 | Chronos | foundation | 374,610.314 | Chronos | 374,610.314 | AutoFitV3 | 395,551.536 | +5.59% |
| task2_forecast | core_edgar | investors_count | 1 | KAN | transformer_sota | 44.809991 | KAN | 44.809991 | AutoFitV3Max | 137.261891 | +206.32% |
| task2_forecast | core_edgar | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV3Max | 137.261891 | +206.45% |
| task2_forecast | core_edgar | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV3Max | 137.261891 | +206.40% |
| task2_forecast | core_edgar | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV3Max | 137.261891 | +206.31% |
| task2_forecast | core_only | funding_raised_usd | 1 | NBEATS | deep_classical | 380,659.460 | NBEATS | 380,659.460 | AutoFitV7 | 397,244.478 | +4.36% |
| task2_forecast | core_only | funding_raised_usd | 7 | NHITS | deep_classical | 380,577.133 | NHITS | 380,577.133 | AutoFitV7 | 397,244.478 | +4.38% |
| task2_forecast | core_only | funding_raised_usd | 14 | Chronos | foundation | 380,832.309 | Chronos | 380,832.309 | AutoFitV7 | 397,244.478 | +4.31% |
| task2_forecast | core_only | funding_raised_usd | 30 | Chronos | foundation | 380,755.090 | Chronos | 380,755.090 | AutoFitV7 | 397,244.478 | +4.33% |
| task2_forecast | core_only | investors_count | 1 | KAN | transformer_sota | 44.745049 | KAN | 44.745049 | AutoFitV3Max | 113.911783 | +154.58% |
| task2_forecast | core_only | investors_count | 7 | NBEATS | deep_classical | 44.726689 | NBEATS | 44.726689 | AutoFitV3Max | 113.911783 | +154.68% |
| task2_forecast | core_only | investors_count | 14 | NBEATS | deep_classical | 44.734036 | NBEATS | 44.734036 | AutoFitV3Max | 113.911783 | +154.64% |
| task2_forecast | core_only | investors_count | 30 | NBEATS | deep_classical | 44.746757 | NBEATS | 44.746757 | AutoFitV3Max | 113.911783 | +154.57% |
| task2_forecast | core_text | funding_raised_usd | 1 | NBEATS | deep_classical | 380,659.460 | NBEATS | 380,659.460 | AutoFitV7 | 397,244.478 | +4.36% |
| task2_forecast | core_text | funding_raised_usd | 7 | NHITS | deep_classical | 380,577.133 | NHITS | 380,577.133 | AutoFitV7 | 397,244.478 | +4.38% |
| task2_forecast | core_text | funding_raised_usd | 14 | Chronos | foundation | 380,832.309 | Chronos | 380,832.309 | AutoFitV7 | 397,244.478 | +4.31% |
| task2_forecast | core_text | funding_raised_usd | 30 | Chronos | foundation | 380,755.090 | Chronos | 380,755.090 | AutoFitV7 | 397,244.478 | +4.33% |
| task2_forecast | core_text | investors_count | 1 | KAN | transformer_sota | 44.745049 | KAN | 44.745049 | AutoFitV1 | 113.911783 | +154.58% |
| task2_forecast | core_text | investors_count | 7 | NBEATS | deep_classical | 44.726689 | NBEATS | 44.726689 | AutoFitV1 | 113.911783 | +154.68% |
| task2_forecast | core_text | investors_count | 14 | NBEATS | deep_classical | 44.734036 | NBEATS | 44.734036 | AutoFitV1 | 113.911783 | +154.64% |
| task2_forecast | core_text | investors_count | 30 | NBEATS | deep_classical | 44.746757 | NBEATS | 44.746757 | AutoFitV3 | 113.911783 | +154.57% |
| task2_forecast | full | funding_raised_usd | 1 | NBEATSx | transformer_sota | 374,514.684 | NBEATSx | 374,514.684 | AutoFitV3 | 396,360.349 | +5.83% |
| task2_forecast | full | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV1 | 396,360.349 | +5.86% |
| task2_forecast | full | funding_raised_usd | 14 | Chronos | foundation | 374,687.533 | Chronos | 374,687.533 | AutoFitV3Max | 396,360.349 | +5.78% |
| task2_forecast | full | funding_raised_usd | 30 | Chronos | foundation | 374,610.314 | Chronos | 374,610.314 | AutoFitV3Max | 396,360.349 | +5.81% |
| task2_forecast | full | investors_count | 1 | KAN | transformer_sota | 44.809991 | KAN | 44.809991 | AutoFitV1 | 125.468632 | +180.00% |
| task2_forecast | full | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV1 | 125.468632 | +180.12% |
| task2_forecast | full | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV1 | 125.468632 | +180.07% |
| task2_forecast | full | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV1 | 125.468632 | +179.99% |
| task3_risk_adjust | core_edgar | funding_raised_usd | 1 | NBEATS | deep_classical | 374,514.684 | NBEATS | 374,514.684 | AutoFitV3 | 395,551.536 | +5.62% |
| task3_risk_adjust | core_edgar | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV3 | 395,551.536 | +5.64% |
| task3_risk_adjust | core_edgar | funding_raised_usd | 14 | Chronos | foundation | 374,687.533 | Chronos | 374,687.533 | AutoFitV3 | 395,551.536 | +5.57% |
| task3_risk_adjust | core_edgar | funding_raised_usd | 30 | Chronos | foundation | 374,610.314 | Chronos | 374,610.314 | AutoFitV3 | 395,551.536 | +5.59% |
| task3_risk_adjust | core_edgar | investors_count | 1 | KAN | transformer_sota | 44.809991 | KAN | 44.809991 | AutoFitV1 | 125.572330 | +180.23% |
| task3_risk_adjust | core_edgar | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV1 | 125.572330 | +180.35% |
| task3_risk_adjust | core_edgar | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV1 | 125.572330 | +180.30% |
| task3_risk_adjust | core_edgar | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV1 | 125.572330 | +180.22% |
| task3_risk_adjust | core_only | funding_raised_usd | 1 | NBEATS | deep_classical | 380,659.460 | NBEATS | 380,659.460 | AutoFitV7 | 397,244.478 | +4.36% |
| task3_risk_adjust | core_only | funding_raised_usd | 7 | NHITS | deep_classical | 380,577.133 | NHITS | 380,577.133 | AutoFitV7 | 397,244.478 | +4.38% |
| task3_risk_adjust | core_only | funding_raised_usd | 14 | Chronos | foundation | 380,832.309 | Chronos | 380,832.309 | AutoFitV7 | 397,244.478 | +4.31% |
| task3_risk_adjust | core_only | funding_raised_usd | 30 | Chronos | foundation | 380,755.090 | Chronos | 380,755.090 | AutoFitV7 | 397,244.478 | +4.33% |
| task3_risk_adjust | core_only | investors_count | 1 | KAN | transformer_sota | 44.745049 | KAN | 44.745049 | AutoFitV3 | 113.911783 | +154.58% |
| task3_risk_adjust | core_only | investors_count | 7 | NBEATS | deep_classical | 44.726689 | NBEATS | 44.726689 | AutoFitV1 | 113.911783 | +154.68% |
| task3_risk_adjust | core_only | investors_count | 14 | NBEATS | deep_classical | 44.734036 | NBEATS | 44.734036 | AutoFitV1 | 113.911783 | +154.64% |
| task3_risk_adjust | core_only | investors_count | 30 | NBEATS | deep_classical | 44.746757 | NBEATS | 44.746757 | AutoFitV1 | 113.911783 | +154.57% |
| task3_risk_adjust | full | funding_raised_usd | 1 | NBEATSx | transformer_sota | 374,514.684 | NBEATSx | 374,514.684 | AutoFitV1 | 396,360.349 | +5.83% |
| task3_risk_adjust | full | funding_raised_usd | 7 | NHITS | deep_classical | 374,432.357 | NHITS | 374,432.357 | AutoFitV1 | 396,360.349 | +5.86% |
| task3_risk_adjust | full | funding_raised_usd | 14 | Chronos | foundation | 374,687.533 | Chronos | 374,687.533 | AutoFitV3Max | 396,360.349 | +5.78% |
| task3_risk_adjust | full | funding_raised_usd | 30 | Chronos | foundation | 374,610.314 | Chronos | 374,610.314 | AutoFitV3Max | 396,360.349 | +5.81% |
| task3_risk_adjust | full | investors_count | 1 | KAN | transformer_sota | 44.809991 | KAN | 44.809991 | AutoFitV1 | 125.468632 | +180.00% |
| task3_risk_adjust | full | investors_count | 7 | NBEATS | deep_classical | 44.791632 | NBEATS | 44.791632 | AutoFitV1 | 125.468632 | +180.12% |
| task3_risk_adjust | full | investors_count | 14 | NBEATS | deep_classical | 44.798978 | NBEATS | 44.798978 | AutoFitV1 | 125.468632 | +180.07% |
| task3_risk_adjust | full | investors_count | 30 | NBEATS | deep_classical | 44.811699 | NBEATS | 44.811699 | AutoFitV1 | 125.468632 | +179.99% |

Source table: `docs/benchmarks/block3_truth_pack/condition_leaderboard.csv`
