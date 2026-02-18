# AutoFit V7.2 Evidence Master (2026-02-17)

Scope: Block 3 full-scale benchmark evidence consolidation in one document.

Method: strict comparability filter (`fairness_pass=true` and `prediction_coverage_ratio>=0.98`) with explicit legacy-unverified separation.

Refresh command:

```bash
python scripts/build_block3_truth_pack.py --include-freeze-history --update-master-doc
```

## Evidence Snapshot

<!-- BEGIN AUTO:EVIDENCE_SNAPSHOT -->
Generated from strict/legacy layered truth pack outputs under `docs/benchmarks/block3_truth_pack/`.

| metric | value | evidence_path |
|---|---|---|
| bench_dirs | runs/benchmarks/block3_20260203_225620_iris;runs/benchmarks/block3_20260203_225620_iris_phase3;runs/benchmarks/block3_20260203_225620_iris_full;runs/benchmarks/block3_20260203_225620_iris_phase7;runs/benchmarks/block3_20260203_225620_phase7;runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205;runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| raw_records | 13261 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| strict_records | 3879 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| legacy_unverified_records | 9382 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| strict_condition_completion | 104/104 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| legacy_condition_completion | 104/104 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| v71_win_rate_vs_v7 | 0.709677 | docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv |
| v71_median_relative_gain_vs_v7_pct | 0.131639 | docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv |
| critical_failures | 4 | docs/benchmarks/block3_truth_pack/failure_taxonomy.csv |
<!-- END AUTO:EVIDENCE_SNAPSHOT -->

## Audit Gates Snapshot

<!-- BEGIN AUTO:AUDIT_GATES -->
Gate snapshot is sourced from latest read-only audit artifacts.

| audit | generated_at_utc | overall_pass | key_signal | evidence_path |
|---|---|---|---|---|
| data_integrity | 2026-02-18T15:20:24.723890+00:00 | true | {"embargo_non_negative": true, "train_before_val": true, "val_before_test": true} | docs/benchmarks/block3_truth_pack/data_integrity_audit_latest.json |
| data_integrity.freeze_gate | 2026-02-18T15:20:24.723890+00:00 | true | {"exit_code": 1, "fallback_mode": "pointer_internal_checks", "n_checks": 11} | docs/benchmarks/block3_truth_pack/data_integrity_audit_latest.json |
| investors_count_stability | 2026-02-18T15:20:54.540707+00:00 | false | {"catastrophic_spikes": 4, "guard_telemetry": {"anchor_models_used_topk": [], "inverse_transform_guard_hits_total": 0, "lane_clip_rate_median": 0.0, "n_rows": 88, "oof_guard_triggered_count": 0, "policy_action_id_topk": []}, "strict_record_count": 1595} | docs/benchmarks/block3_truth_pack/investors_count_stability_audit_latest.json |
<!-- END AUTO:AUDIT_GATES -->

## Task And Subtask Universe

<!-- BEGIN AUTO:TASK_AND_SUBTASK_UNIVERSE -->
### Task/Subtask Catalog

| subtask_id | subtask_family | definition_rule | key_count | key_coverage_strict | evidence_path |
|---|---|---|---|---|---|
| all_condition_keys | condition_universe | All expected keys from task x ablation x target x horizon lattice. | 104 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task:task1_outcome | task | All condition keys where task == task1_outcome. | 48 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task:task2_forecast | task | All condition keys where task == task2_forecast. | 32 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task:task3_risk_adjust | task | All condition keys where task == task3_risk_adjust. | 24 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task_target:task1_outcome|funding_raised_usd | task_target | All keys where task == task1_outcome and target == funding_raised_usd. | 16 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task_target:task1_outcome|investors_count | task_target | All keys where task == task1_outcome and target == investors_count. | 16 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task_target:task1_outcome|is_funded | task_target | All keys where task == task1_outcome and target == is_funded. | 16 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task_target:task2_forecast|funding_raised_usd | task_target | All keys where task == task2_forecast and target == funding_raised_usd. | 16 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task_target:task2_forecast|investors_count | task_target | All keys where task == task2_forecast and target == investors_count. | 16 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task_target:task3_risk_adjust|funding_raised_usd | task_target | All keys where task == task3_risk_adjust and target == funding_raised_usd. | 12 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task_target:task3_risk_adjust|investors_count | task_target | All keys where task == task3_risk_adjust and target == investors_count. | 12 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| target_family:binary | target_family | Lane family inferred from target semantics. | 16 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| target_family:count | target_family | Lane family inferred from target semantics. | 44 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| target_family:heavy_tail | target_family | Lane family inferred from target semantics. | 44 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| modality:core_edgar | modality | All keys where ablation == core_edgar. | 28 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| modality:core_only | modality | All keys where ablation == core_only. | 28 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| modality:core_text | modality | All keys where ablation == core_text. | 20 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| modality:full | modality | All keys where ablation == full. | 28 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| horizon:1 | horizon | All keys where horizon == 1. | 26 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| horizon:7 | horizon | All keys where horizon == 7. | 26 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| horizon:14 | horizon | All keys where horizon == 14. | 26 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| horizon:30 | horizon | All keys where horizon == 30. | 26 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| horizon_band:short | horizon_band | Band grouping on horizon values ([1, 7]). | 52 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| horizon_band:mid | horizon_band | Band grouping on horizon values ([14]). | 26 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| horizon_band:long | horizon_band | Band grouping on horizon values ([30]). | 26 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| robustness:task3_risk_adjust | robustness | OOD robustness proxy keys from task3_risk_adjust. | 24 | 1.000000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| data_feature:funding_raised_usd | data_characteristic | Missingness and cardinality profile for funding_raised_usd. missing_rate=0.78503, n_unique=14554. | 44 | 1.000000 | runs/orchestrator/20260129_073037/analysis/wide_20260203_225620/column_manifest.json |
| data_feature:investors_count | data_characteristic | Missingness and cardinality profile for investors_count. missing_rate=0.42975, n_unique=817. | 44 | 1.000000 | runs/orchestrator/20260129_073037/analysis/wide_20260203_225620/column_manifest.json |
| data_feature:is_funded | data_characteristic | Missingness and cardinality profile for is_funded. missing_rate=0.53212, n_unique=2. | 16 | 1.000000 | runs/orchestrator/20260129_073037/analysis/wide_20260203_225620/column_manifest.json |

### Full Condition Inventory

| task | ablation | target | horizon | expected | strict_completed | legacy_completed | best_model_strict | best_category_strict | best_mae_strict | evidence_path |
|---|---|---|---|---|---|---|---|---|---|---|
| task1_outcome | core_edgar | funding_raised_usd | 1 | true | true | true | NBEATS | deep_classical | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_edgar | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_edgar | funding_raised_usd | 14 | true | true | true | PatchTST | transformer_sota | 375055.785 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_edgar | funding_raised_usd | 30 | true | true | true | PatchTST | transformer_sota | 375472.395 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_edgar | investors_count | 1 | true | true | true | NHITS | deep_classical | 44.836898 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_edgar | investors_count | 7 | true | true | true | NBEATS | deep_classical | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_edgar | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_edgar | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_edgar | is_funded | 1 | true | true | true | PatchTST | transformer_sota | 0.032294 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_edgar | is_funded | 7 | true | true | true | NHITS | deep_classical | 0.032309 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_edgar | is_funded | 14 | true | true | true | PatchTST | transformer_sota | 0.032281 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_edgar | is_funded | 30 | true | true | true | NHITS | deep_classical | 0.032248 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | funding_raised_usd | 1 | true | true | true | NBEATS | deep_classical | 380659.460 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 380577.133 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | funding_raised_usd | 14 | true | true | true | PatchTST | transformer_sota | 381200.561 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | funding_raised_usd | 30 | true | true | true | PatchTST | transformer_sota | 381617.171 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | investors_count | 1 | true | true | true | NHITS | deep_classical | 44.771955 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | investors_count | 7 | true | true | true | NBEATS | deep_classical | 44.726689 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | is_funded | 1 | true | true | true | PatchTST | transformer_sota | 0.033009 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | is_funded | 7 | true | true | true | NHITS | deep_classical | 0.033030 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | is_funded | 14 | true | true | true | PatchTST | transformer_sota | 0.033002 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | is_funded | 30 | true | true | true | NHITS | deep_classical | 0.032970 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | funding_raised_usd | 1 | true | true | true | NBEATS | deep_classical | 380659.460 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 380577.133 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | funding_raised_usd | 14 | true | true | true | PatchTST | transformer_sota | 381200.561 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | funding_raised_usd | 30 | true | true | true | PatchTST | transformer_sota | 381617.171 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | investors_count | 1 | true | true | true | NHITS | deep_classical | 44.771955 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | investors_count | 7 | true | true | true | NBEATS | deep_classical | 44.726689 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | is_funded | 1 | true | true | true | PatchTST | transformer_sota | 0.033015 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | is_funded | 7 | true | true | true | NHITS | deep_classical | 0.033030 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | is_funded | 14 | true | true | true | PatchTST | transformer_sota | 0.033002 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | is_funded | 30 | true | true | true | NHITS | deep_classical | 0.032970 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | funding_raised_usd | 1 | true | true | true | NBEATSx | transformer_sota | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | funding_raised_usd | 14 | true | true | true | Chronos | foundation | 374687.533 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | funding_raised_usd | 30 | true | true | true | Chronos | foundation | 374610.314 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | investors_count | 1 | true | true | true | KAN | transformer_sota | 44.809991 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | investors_count | 7 | true | true | true | NBEATSx | transformer_sota | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | investors_count | 14 | true | true | true | NBEATSx | transformer_sota | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | investors_count | 30 | true | true | true | NBEATSx | transformer_sota | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | is_funded | 1 | true | true | true | PatchTST | transformer_sota | 0.032294 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | is_funded | 7 | true | true | true | NHITS | deep_classical | 0.032309 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | is_funded | 14 | true | true | true | PatchTST | transformer_sota | 0.032281 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | full | is_funded | 30 | true | true | true | NHITS | deep_classical | 0.032248 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_edgar | funding_raised_usd | 1 | true | true | true | NBEATS | deep_classical | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_edgar | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_edgar | funding_raised_usd | 14 | true | true | true | PatchTST | transformer_sota | 375055.785 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_edgar | funding_raised_usd | 30 | true | true | true | PatchTST | transformer_sota | 375472.395 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_edgar | investors_count | 1 | true | true | true | NHITS | deep_classical | 44.836898 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_edgar | investors_count | 7 | true | true | true | NBEATS | deep_classical | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_edgar | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_edgar | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_only | funding_raised_usd | 1 | true | true | true | NBEATS | deep_classical | 380659.121 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_only | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 380577.133 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_only | funding_raised_usd | 14 | true | true | true | PatchTST | transformer_sota | 381200.561 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_only | funding_raised_usd | 30 | true | true | true | PatchTST | transformer_sota | 381617.171 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_only | investors_count | 1 | true | true | true | NHITS | deep_classical | 44.771955 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_only | investors_count | 7 | true | true | true | NBEATS | deep_classical | 44.726689 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_only | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_only | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_text | funding_raised_usd | 1 | true | true | true | NBEATS | deep_classical | 380659.460 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_text | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 380576.794 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_text | funding_raised_usd | 14 | true | true | true | PatchTST | transformer_sota | 381200.561 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_text | funding_raised_usd | 30 | true | true | true | PatchTST | transformer_sota | 381616.832 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_text | investors_count | 1 | true | true | true | NHITS | deep_classical | 44.771955 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_text | investors_count | 7 | true | true | true | NBEATS | deep_classical | 44.726689 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_text | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | core_text | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | full | funding_raised_usd | 1 | true | true | true | NBEATSx | transformer_sota | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | full | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | full | funding_raised_usd | 14 | true | true | true | Chronos | foundation | 374687.533 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | full | funding_raised_usd | 30 | true | true | true | Chronos | foundation | 374610.314 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | full | investors_count | 1 | true | true | true | KAN | transformer_sota | 44.809991 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | full | investors_count | 7 | true | true | true | NBEATS | deep_classical | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | full | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast | full | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_edgar | funding_raised_usd | 1 | true | true | true | NBEATS | deep_classical | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_edgar | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_edgar | funding_raised_usd | 14 | true | true | true | PatchTST | transformer_sota | 375055.785 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_edgar | funding_raised_usd | 30 | true | true | true | PatchTST | transformer_sota | 375472.395 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_edgar | investors_count | 1 | true | true | true | NHITS | deep_classical | 44.836898 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_edgar | investors_count | 7 | true | true | true | NBEATS | deep_classical | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_edgar | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_edgar | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_only | funding_raised_usd | 1 | true | true | true | NBEATS | deep_classical | 380659.460 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_only | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 380577.133 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_only | funding_raised_usd | 14 | true | true | true | PatchTST | transformer_sota | 381200.561 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_only | funding_raised_usd | 30 | true | true | true | PatchTST | transformer_sota | 381616.832 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_only | investors_count | 1 | true | true | true | NHITS | deep_classical | 44.771955 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_only | investors_count | 7 | true | true | true | NBEATS | deep_classical | 44.726689 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_only | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | core_only | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | full | funding_raised_usd | 1 | true | true | true | NBEATS | deep_classical | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | full | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | full | funding_raised_usd | 14 | true | true | true | Chronos | foundation | 374687.533 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | full | funding_raised_usd | 30 | true | true | true | Chronos | foundation | 374610.314 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | full | investors_count | 1 | true | true | true | KAN | transformer_sota | 44.809991 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | full | investors_count | 7 | true | true | true | NBEATS | deep_classical | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | full | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust | full | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
<!-- END AUTO:TASK_AND_SUBTASK_UNIVERSE -->

## Data-Characteristic-Derived Subtasks

<!-- BEGIN AUTO:DATA_CHARACTERISTIC_DERIVED_SUBTASKS -->
Derived subtasks are created from target lane semantics and target missingness/cardinality profile.

| subtask_id | subtask_family | definition_rule | key_count | key_coverage_strict | evidence_path |
|---|---|---|---|---|---|
| data_feature:funding_raised_usd | data_characteristic | Missingness and cardinality profile for funding_raised_usd. missing_rate=0.78503, n_unique=14554. | 44 | 1.000000 | runs/orchestrator/20260129_073037/analysis/wide_20260203_225620/column_manifest.json |
| data_feature:investors_count | data_characteristic | Missingness and cardinality profile for investors_count. missing_rate=0.42975, n_unique=817. | 44 | 1.000000 | runs/orchestrator/20260129_073037/analysis/wide_20260203_225620/column_manifest.json |
| data_feature:is_funded | data_characteristic | Missingness and cardinality profile for is_funded. missing_rate=0.53212, n_unique=2. | 16 | 1.000000 | runs/orchestrator/20260129_073037/analysis/wide_20260203_225620/column_manifest.json |
<!-- END AUTO:DATA_CHARACTERISTIC_DERIVED_SUBTASKS -->

## Model Family Coverage Audit

<!-- BEGIN AUTO:MODEL_FAMILY_COVERAGE_AUDIT -->
| category | n_models_registered | n_models_observed_strict | strict_model_coverage_ratio | models_observed_strict | missing_models_strict | condition_wins | condition_win_share | evidence_path |
|---|---|---|---|---|---|---|---|---|
| autofit | 12 | 11 | 0.916667 | AutoFitV1;AutoFitV2;AutoFitV2E;AutoFitV3;AutoFitV3E;AutoFitV3Max;AutoFitV4;AutoFitV5;AutoFitV6;AutoFitV7;AutoFitV71 | AutoFitV72 | 0 | 0.000000 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| deep_classical | 4 | 4 | 1.000000 | DeepAR;NBEATS;NHITS;TFT |  | 66 | 0.634615 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| foundation | 11 | 10 | 0.909091 | Chronos;ChronosBolt;LagLlama;MOMENT;Moirai;Moirai2;MoiraiLarge;TimeMoE;Timer;TimesFM | Chronos2 | 6 | 0.057692 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| irregular | 2 | 0 | 0.000000 |  | GRU-D;SAITS | 0 | 0.000000 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| ml_tabular | 19 | 17 | 0.894737 | CatBoost;ElasticNet;ExtraTrees;HistGradientBoosting;KNN;Lasso;LightGBM;LightGBMTweedie;LogisticRegression;MeanPredictor;QuantileRegressor;RandomForest;Ridge;SVR;SeasonalNaive;XGBoost;XGBoostPoisson | TabPFNClassifier;TabPFNRegressor | 0 | 0.000000 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| statistical | 5 | 5 | 1.000000 | AutoARIMA;AutoETS;AutoTheta;MSTL;SF_SeasonalNaive |  | 0 | 0.000000 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| transformer_sota | 20 | 20 | 1.000000 | Autoformer;BiTCN;DLinear;FEDformer;Informer;KAN;NBEATSx;NLinear;PatchTST;RMoK;SOFTS;StemGNN;TSMixer;TSMixerx;TiDE;TimeMixer;TimeXer;TimesNet;VanillaTransformer;iTransformer |  | 32 | 0.307692 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
<!-- END AUTO:MODEL_FAMILY_COVERAGE_AUDIT -->

## Target Subtasks (is_funded / funding_raised_usd / investors_count)

<!-- BEGIN AUTO:TARGET_SUBTASKS -->
| subtask_id | task | ablation | target | target_family | horizon | strict_completed | legacy_completed | best_model_strict | best_category_strict | best_mae_strict | evidence_path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| task1_outcome__core_edgar__funding_raised_usd__h1 | task1_outcome | core_edgar | funding_raised_usd | heavy_tail | 1 | true | true | NBEATS | deep_classical | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_edgar__funding_raised_usd__h7 | task1_outcome | core_edgar | funding_raised_usd | heavy_tail | 7 | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_edgar__funding_raised_usd__h14 | task1_outcome | core_edgar | funding_raised_usd | heavy_tail | 14 | true | true | PatchTST | transformer_sota | 375055.785 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_edgar__funding_raised_usd__h30 | task1_outcome | core_edgar | funding_raised_usd | heavy_tail | 30 | true | true | PatchTST | transformer_sota | 375472.395 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__funding_raised_usd__h1 | task1_outcome | core_only | funding_raised_usd | heavy_tail | 1 | true | true | NBEATS | deep_classical | 380659.460 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__funding_raised_usd__h7 | task1_outcome | core_only | funding_raised_usd | heavy_tail | 7 | true | true | NHITS | deep_classical | 380577.133 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__funding_raised_usd__h14 | task1_outcome | core_only | funding_raised_usd | heavy_tail | 14 | true | true | PatchTST | transformer_sota | 381200.561 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__funding_raised_usd__h30 | task1_outcome | core_only | funding_raised_usd | heavy_tail | 30 | true | true | PatchTST | transformer_sota | 381617.171 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__funding_raised_usd__h1 | task1_outcome | core_text | funding_raised_usd | heavy_tail | 1 | true | true | NBEATS | deep_classical | 380659.460 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__funding_raised_usd__h7 | task1_outcome | core_text | funding_raised_usd | heavy_tail | 7 | true | true | NHITS | deep_classical | 380577.133 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__funding_raised_usd__h14 | task1_outcome | core_text | funding_raised_usd | heavy_tail | 14 | true | true | PatchTST | transformer_sota | 381200.561 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__funding_raised_usd__h30 | task1_outcome | core_text | funding_raised_usd | heavy_tail | 30 | true | true | PatchTST | transformer_sota | 381617.171 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__funding_raised_usd__h1 | task1_outcome | full | funding_raised_usd | heavy_tail | 1 | true | true | NBEATSx | transformer_sota | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__funding_raised_usd__h7 | task1_outcome | full | funding_raised_usd | heavy_tail | 7 | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__funding_raised_usd__h14 | task1_outcome | full | funding_raised_usd | heavy_tail | 14 | true | true | Chronos | foundation | 374687.533 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__funding_raised_usd__h30 | task1_outcome | full | funding_raised_usd | heavy_tail | 30 | true | true | Chronos | foundation | 374610.314 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_edgar__funding_raised_usd__h1 | task2_forecast | core_edgar | funding_raised_usd | heavy_tail | 1 | true | true | NBEATS | deep_classical | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_edgar__funding_raised_usd__h7 | task2_forecast | core_edgar | funding_raised_usd | heavy_tail | 7 | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_edgar__funding_raised_usd__h14 | task2_forecast | core_edgar | funding_raised_usd | heavy_tail | 14 | true | true | PatchTST | transformer_sota | 375055.785 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_edgar__funding_raised_usd__h30 | task2_forecast | core_edgar | funding_raised_usd | heavy_tail | 30 | true | true | PatchTST | transformer_sota | 375472.395 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_only__funding_raised_usd__h1 | task2_forecast | core_only | funding_raised_usd | heavy_tail | 1 | true | true | NBEATS | deep_classical | 380659.121 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_only__funding_raised_usd__h7 | task2_forecast | core_only | funding_raised_usd | heavy_tail | 7 | true | true | NHITS | deep_classical | 380577.133 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_only__funding_raised_usd__h14 | task2_forecast | core_only | funding_raised_usd | heavy_tail | 14 | true | true | PatchTST | transformer_sota | 381200.561 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_only__funding_raised_usd__h30 | task2_forecast | core_only | funding_raised_usd | heavy_tail | 30 | true | true | PatchTST | transformer_sota | 381617.171 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_text__funding_raised_usd__h1 | task2_forecast | core_text | funding_raised_usd | heavy_tail | 1 | true | true | NBEATS | deep_classical | 380659.460 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_text__funding_raised_usd__h7 | task2_forecast | core_text | funding_raised_usd | heavy_tail | 7 | true | true | NHITS | deep_classical | 380576.794 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_text__funding_raised_usd__h14 | task2_forecast | core_text | funding_raised_usd | heavy_tail | 14 | true | true | PatchTST | transformer_sota | 381200.561 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_text__funding_raised_usd__h30 | task2_forecast | core_text | funding_raised_usd | heavy_tail | 30 | true | true | PatchTST | transformer_sota | 381616.832 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__full__funding_raised_usd__h1 | task2_forecast | full | funding_raised_usd | heavy_tail | 1 | true | true | NBEATSx | transformer_sota | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__full__funding_raised_usd__h7 | task2_forecast | full | funding_raised_usd | heavy_tail | 7 | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__full__funding_raised_usd__h14 | task2_forecast | full | funding_raised_usd | heavy_tail | 14 | true | true | Chronos | foundation | 374687.533 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__full__funding_raised_usd__h30 | task2_forecast | full | funding_raised_usd | heavy_tail | 30 | true | true | Chronos | foundation | 374610.314 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_edgar__funding_raised_usd__h1 | task3_risk_adjust | core_edgar | funding_raised_usd | heavy_tail | 1 | true | true | NBEATS | deep_classical | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_edgar__funding_raised_usd__h7 | task3_risk_adjust | core_edgar | funding_raised_usd | heavy_tail | 7 | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_edgar__funding_raised_usd__h14 | task3_risk_adjust | core_edgar | funding_raised_usd | heavy_tail | 14 | true | true | PatchTST | transformer_sota | 375055.785 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_edgar__funding_raised_usd__h30 | task3_risk_adjust | core_edgar | funding_raised_usd | heavy_tail | 30 | true | true | PatchTST | transformer_sota | 375472.395 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_only__funding_raised_usd__h1 | task3_risk_adjust | core_only | funding_raised_usd | heavy_tail | 1 | true | true | NBEATS | deep_classical | 380659.460 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_only__funding_raised_usd__h7 | task3_risk_adjust | core_only | funding_raised_usd | heavy_tail | 7 | true | true | NHITS | deep_classical | 380577.133 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_only__funding_raised_usd__h14 | task3_risk_adjust | core_only | funding_raised_usd | heavy_tail | 14 | true | true | PatchTST | transformer_sota | 381200.561 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_only__funding_raised_usd__h30 | task3_risk_adjust | core_only | funding_raised_usd | heavy_tail | 30 | true | true | PatchTST | transformer_sota | 381616.832 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__full__funding_raised_usd__h1 | task3_risk_adjust | full | funding_raised_usd | heavy_tail | 1 | true | true | NBEATS | deep_classical | 374514.684 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__full__funding_raised_usd__h7 | task3_risk_adjust | full | funding_raised_usd | heavy_tail | 7 | true | true | NHITS | deep_classical | 374432.357 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__full__funding_raised_usd__h14 | task3_risk_adjust | full | funding_raised_usd | heavy_tail | 14 | true | true | Chronos | foundation | 374687.533 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__full__funding_raised_usd__h30 | task3_risk_adjust | full | funding_raised_usd | heavy_tail | 30 | true | true | Chronos | foundation | 374610.314 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_edgar__investors_count__h1 | task1_outcome | core_edgar | investors_count | count | 1 | true | true | NHITS | deep_classical | 44.836898 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_edgar__investors_count__h7 | task1_outcome | core_edgar | investors_count | count | 7 | true | true | NBEATS | deep_classical | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_edgar__investors_count__h14 | task1_outcome | core_edgar | investors_count | count | 14 | true | true | NBEATS | deep_classical | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_edgar__investors_count__h30 | task1_outcome | core_edgar | investors_count | count | 30 | true | true | NBEATS | deep_classical | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__investors_count__h1 | task1_outcome | core_only | investors_count | count | 1 | true | true | NHITS | deep_classical | 44.771955 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__investors_count__h7 | task1_outcome | core_only | investors_count | count | 7 | true | true | NBEATS | deep_classical | 44.726689 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__investors_count__h14 | task1_outcome | core_only | investors_count | count | 14 | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__investors_count__h30 | task1_outcome | core_only | investors_count | count | 30 | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__investors_count__h1 | task1_outcome | core_text | investors_count | count | 1 | true | true | NHITS | deep_classical | 44.771955 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__investors_count__h7 | task1_outcome | core_text | investors_count | count | 7 | true | true | NBEATS | deep_classical | 44.726689 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__investors_count__h14 | task1_outcome | core_text | investors_count | count | 14 | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__investors_count__h30 | task1_outcome | core_text | investors_count | count | 30 | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__investors_count__h1 | task1_outcome | full | investors_count | count | 1 | true | true | KAN | transformer_sota | 44.809991 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__investors_count__h7 | task1_outcome | full | investors_count | count | 7 | true | true | NBEATSx | transformer_sota | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__investors_count__h14 | task1_outcome | full | investors_count | count | 14 | true | true | NBEATSx | transformer_sota | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__investors_count__h30 | task1_outcome | full | investors_count | count | 30 | true | true | NBEATSx | transformer_sota | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_edgar__investors_count__h1 | task2_forecast | core_edgar | investors_count | count | 1 | true | true | NHITS | deep_classical | 44.836898 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_edgar__investors_count__h7 | task2_forecast | core_edgar | investors_count | count | 7 | true | true | NBEATS | deep_classical | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_edgar__investors_count__h14 | task2_forecast | core_edgar | investors_count | count | 14 | true | true | NBEATS | deep_classical | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_edgar__investors_count__h30 | task2_forecast | core_edgar | investors_count | count | 30 | true | true | NBEATS | deep_classical | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_only__investors_count__h1 | task2_forecast | core_only | investors_count | count | 1 | true | true | NHITS | deep_classical | 44.771955 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_only__investors_count__h7 | task2_forecast | core_only | investors_count | count | 7 | true | true | NBEATS | deep_classical | 44.726689 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_only__investors_count__h14 | task2_forecast | core_only | investors_count | count | 14 | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_only__investors_count__h30 | task2_forecast | core_only | investors_count | count | 30 | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_text__investors_count__h1 | task2_forecast | core_text | investors_count | count | 1 | true | true | NHITS | deep_classical | 44.771955 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_text__investors_count__h7 | task2_forecast | core_text | investors_count | count | 7 | true | true | NBEATS | deep_classical | 44.726689 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_text__investors_count__h14 | task2_forecast | core_text | investors_count | count | 14 | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__core_text__investors_count__h30 | task2_forecast | core_text | investors_count | count | 30 | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__full__investors_count__h1 | task2_forecast | full | investors_count | count | 1 | true | true | KAN | transformer_sota | 44.809991 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__full__investors_count__h7 | task2_forecast | full | investors_count | count | 7 | true | true | NBEATS | deep_classical | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__full__investors_count__h14 | task2_forecast | full | investors_count | count | 14 | true | true | NBEATS | deep_classical | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task2_forecast__full__investors_count__h30 | task2_forecast | full | investors_count | count | 30 | true | true | NBEATS | deep_classical | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_edgar__investors_count__h1 | task3_risk_adjust | core_edgar | investors_count | count | 1 | true | true | NHITS | deep_classical | 44.836898 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_edgar__investors_count__h7 | task3_risk_adjust | core_edgar | investors_count | count | 7 | true | true | NBEATS | deep_classical | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_edgar__investors_count__h14 | task3_risk_adjust | core_edgar | investors_count | count | 14 | true | true | NBEATS | deep_classical | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_edgar__investors_count__h30 | task3_risk_adjust | core_edgar | investors_count | count | 30 | true | true | NBEATS | deep_classical | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_only__investors_count__h1 | task3_risk_adjust | core_only | investors_count | count | 1 | true | true | NHITS | deep_classical | 44.771955 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_only__investors_count__h7 | task3_risk_adjust | core_only | investors_count | count | 7 | true | true | NBEATS | deep_classical | 44.726689 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_only__investors_count__h14 | task3_risk_adjust | core_only | investors_count | count | 14 | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__core_only__investors_count__h30 | task3_risk_adjust | core_only | investors_count | count | 30 | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__full__investors_count__h1 | task3_risk_adjust | full | investors_count | count | 1 | true | true | KAN | transformer_sota | 44.809991 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__full__investors_count__h7 | task3_risk_adjust | full | investors_count | count | 7 | true | true | NBEATS | deep_classical | 44.791632 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__full__investors_count__h14 | task3_risk_adjust | full | investors_count | count | 14 | true | true | NBEATS | deep_classical | 44.798978 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task3_risk_adjust__full__investors_count__h30 | task3_risk_adjust | full | investors_count | count | 30 | true | true | NBEATS | deep_classical | 44.811699 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_edgar__is_funded__h1 | task1_outcome | core_edgar | is_funded | binary | 1 | true | true | PatchTST | transformer_sota | 0.032294 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_edgar__is_funded__h7 | task1_outcome | core_edgar | is_funded | binary | 7 | true | true | NHITS | deep_classical | 0.032309 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_edgar__is_funded__h14 | task1_outcome | core_edgar | is_funded | binary | 14 | true | true | PatchTST | transformer_sota | 0.032281 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_edgar__is_funded__h30 | task1_outcome | core_edgar | is_funded | binary | 30 | true | true | NHITS | deep_classical | 0.032248 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__is_funded__h1 | task1_outcome | core_only | is_funded | binary | 1 | true | true | PatchTST | transformer_sota | 0.033009 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__is_funded__h7 | task1_outcome | core_only | is_funded | binary | 7 | true | true | NHITS | deep_classical | 0.033030 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__is_funded__h14 | task1_outcome | core_only | is_funded | binary | 14 | true | true | PatchTST | transformer_sota | 0.033002 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__is_funded__h30 | task1_outcome | core_only | is_funded | binary | 30 | true | true | NHITS | deep_classical | 0.032970 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__is_funded__h1 | task1_outcome | core_text | is_funded | binary | 1 | true | true | PatchTST | transformer_sota | 0.033015 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__is_funded__h7 | task1_outcome | core_text | is_funded | binary | 7 | true | true | NHITS | deep_classical | 0.033030 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__is_funded__h14 | task1_outcome | core_text | is_funded | binary | 14 | true | true | PatchTST | transformer_sota | 0.033002 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__is_funded__h30 | task1_outcome | core_text | is_funded | binary | 30 | true | true | NHITS | deep_classical | 0.032970 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__is_funded__h1 | task1_outcome | full | is_funded | binary | 1 | true | true | PatchTST | transformer_sota | 0.032294 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__is_funded__h7 | task1_outcome | full | is_funded | binary | 7 | true | true | NHITS | deep_classical | 0.032309 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__is_funded__h14 | task1_outcome | full | is_funded | binary | 14 | true | true | PatchTST | transformer_sota | 0.032281 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__full__is_funded__h30 | task1_outcome | full | is_funded | binary | 30 | true | true | NHITS | deep_classical | 0.032248 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
<!-- END AUTO:TARGET_SUBTASKS -->

## Top-3 Representative Models

<!-- BEGIN AUTO:TOP3_REPRESENTATIVE_MODELS -->
| target | target_family | rank | model_name | category | win_count | win_rate | total_conditions | evidence_path |
|---|---|---|---|---|---|---|---|---|
| funding_raised_usd | heavy_tail | 1 | PatchTST | transformer_sota | 16 | 0.363636 | 44 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| funding_raised_usd | heavy_tail | 2 | NHITS | deep_classical | 11 | 0.250000 | 44 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| funding_raised_usd | heavy_tail | 3 | NBEATS | deep_classical | 9 | 0.204545 | 44 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| investors_count | count | 1 | NBEATS | deep_classical | 30 | 0.681818 | 44 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| investors_count | count | 2 | NHITS | deep_classical | 8 | 0.181818 | 44 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| investors_count | count | 3 | KAN | transformer_sota | 3 | 0.068182 | 44 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| is_funded | binary | 1 | PatchTST | transformer_sota | 8 | 0.500000 | 16 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| is_funded | binary | 2 | NHITS | deep_classical | 8 | 0.500000 | 16 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
<!-- END AUTO:TOP3_REPRESENTATIVE_MODELS -->

## Family Gap Matrix

<!-- BEGIN AUTO:FAMILY_GAP_MATRIX -->
| target | target_family | category | category_best_model | category_best_mae | global_best_model | global_best_category | global_best_mae | gap_vs_global_best_pct | evidence_path |
|---|---|---|---|---|---|---|---|---|---|
| funding_raised_usd | heavy_tail | autofit | AutoFitV3 | 395551.536 | NHITS | deep_classical | 374432.357 | 5.640319 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| funding_raised_usd | heavy_tail | deep_classical | NHITS | 374432.357 | NHITS | deep_classical | 374432.357 | 0.000000 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737/full/deep_refs/task1_outcome/deep_classical/core_edgar/metrics.json |
| funding_raised_usd | heavy_tail | foundation | Chronos | 374610.314 | NHITS | deep_classical | 374432.357 | 0.047527 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| funding_raised_usd | heavy_tail | ml_tabular | RandomForest | 396360.349 | NHITS | deep_classical | 374432.357 | 5.856329 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| funding_raised_usd | heavy_tail | statistical | SF_SeasonalNaive | 4064930.583 | NHITS | deep_classical | 374432.357 | 985.624814 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| funding_raised_usd | heavy_tail | transformer_sota | NBEATSx | 374514.684 | NHITS | deep_classical | 374432.357 | 0.021987 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| investors_count | count | autofit | AutoFitV1 | 125.572330 | NBEATS | deep_classical | 44.726689 | 180.754807 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| investors_count | count | deep_classical | NBEATS | 44.726689 | NBEATS | deep_classical | 44.726689 | 0.000000 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/deep_refs/task1_outcome/deep_classical/core_only/metrics.json |
| investors_count | count | foundation | ChronosBolt | 44.992359 | NBEATS | deep_classical | 44.726689 | 0.593984 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| investors_count | count | ml_tabular | RandomForest | 97.901537 | NBEATS | deep_classical | 44.726689 | 118.888406 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | count | statistical | MSTL | 48.222248 | NBEATS | deep_classical | 44.726689 | 7.815376 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| investors_count | count | transformer_sota | PatchTST | 44.771809 | NBEATS | deep_classical | 44.726689 | 0.100878 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/ts_refs/task1_outcome/transformer_sota/core_only/metrics.json |
| is_funded | binary | autofit | AutoFitV71 | 0.086412 | NHITS | deep_classical | 0.032248 | 167.958234 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/v71_g02/task1_outcome/autofit/core_text/metrics.json |
| is_funded | binary | deep_classical | NHITS | 0.032248 | NHITS | deep_classical | 0.032248 | 0.000000 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737/full/deep_refs/task1_outcome/deep_classical/core_edgar/metrics.json |
| is_funded | binary | foundation | ChronosBolt | 0.033394 | NHITS | deep_classical | 0.032248 | 3.551434 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| is_funded | binary | ml_tabular | ExtraTrees | 0.065715 | NHITS | deep_classical | 0.032248 | 103.777737 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| is_funded | binary | statistical | SF_SeasonalNaive | 0.046009 | NHITS | deep_classical | 0.032248 | 42.670935 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| is_funded | binary | transformer_sota | PatchTST | 0.032281 | NHITS | deep_classical | 0.032248 | 0.100360 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
<!-- END AUTO:FAMILY_GAP_MATRIX -->

## Historical Full-Scale Experiment Ledger

<!-- BEGIN AUTO:HISTORICAL_FULL_SCALE_EXPERIMENT_LEDGER -->
### Run Ledger

| run_name | run_stage | raw_records | strict_records | legacy_records | strict_ratio | models | categories | condition_coverage_strict | condition_coverage_legacy | best_model_by_target_json | key_failures | evidence_path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block3_20260203_225620_iris | iris_initial | 631 | 0 | 631 | 0.000000 | 28 | 5 | 0.000000 | 0.403846 | {} | {} | runs/benchmarks/block3_20260203_225620_iris |
| block3_20260203_225620_iris_phase3 | iris_phase3 | 1750 | 0 | 1750 | 0.000000 | 49 | 7 | 0.000000 | 0.538462 | {} | {} | runs/benchmarks/block3_20260203_225620_iris_phase3 |
| block3_20260203_225620_iris_full | iris_full_baseline | 2646 | 0 | 2646 | 0.000000 | 49 | 7 | 0.000000 | 0.538462 | {} | {} | runs/benchmarks/block3_20260203_225620_iris_full |
| block3_20260203_225620_iris_phase7 | iris_phase7_partial | 1060 | 0 | 1060 | 0.000000 | 56 | 6 | 0.000000 | 0.230769 | {} | {} | runs/benchmarks/block3_20260203_225620_iris_phase7 |
| block3_20260203_225620_phase7 | phase7_canonical | 4814 | 1519 | 3295 | 0.315538 | 68 | 7 | 0.586538 | 1.000000 | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.35741397936, "model": "NHITS"}, "investors_count": {"category": "transformer_sota", "mae": 44.79163160827051, "model": "NBEATSx"}, "is_funded": {"category": "transformer_sota", "mae": 0.032280800947848576, "model": "PatchTST"}} | {"autofit_gap_gt_100pct": 16} | runs/benchmarks/block3_20260203_225620_phase7 |
| block3_20260203_225620_phase7_v71extreme_20260214_032205 | v71_pilot | 1248 | 1248 | 0 | 1.000000 | 12 | 4 | 1.000000 | 0.000000 | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.35741397936, "model": "NHITS"}, "investors_count": {"category": "deep_classical", "mae": 44.726689230844215, "model": "NBEATS"}, "is_funded": {"category": "deep_classical", "mae": 0.03232223233914305, "model": "NHITS"}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 8} | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205 |
| block3_20260203_225620_phase7_v71extreme_20260214_130737 | v71_full | 1112 | 1112 | 0 | 1.000000 | 12 | 4 | 1.000000 | 0.000000 | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.3574093943, "model": "NHITS"}, "investors_count": {"category": "deep_classical", "mae": 44.726689230844215, "model": "NBEATS"}, "is_funded": {"category": "deep_classical", "mae": 0.03224843655460842, "model": "NHITS"}} | {} | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737 |

### Observations

| run_name | observation_type | observation | supporting_metric | evidence_path |
|---|---|---|---|---|
| block3_20260203_225620_iris | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_iris | strict_ratio | Share of strict-comparable records among all valid records. | 0.0000 | runs/benchmarks/block3_20260203_225620_iris |
| block3_20260203_225620_iris_phase3 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_iris_phase3 | strict_ratio | Share of strict-comparable records among all valid records. | 0.0000 | runs/benchmarks/block3_20260203_225620_iris_phase3 |
| block3_20260203_225620_iris_full | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_iris_full | strict_ratio | Share of strict-comparable records among all valid records. | 0.0000 | runs/benchmarks/block3_20260203_225620_iris_full |
| block3_20260203_225620_iris_phase7 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_iris_phase7 | strict_ratio | Share of strict-comparable records among all valid records. | 0.0000 | runs/benchmarks/block3_20260203_225620_iris_phase7 |
| block3_20260203_225620_phase7 | failure_tags | Failure taxonomy tags found in this run. | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/failure_taxonomy.csv |
| block3_20260203_225620_phase7 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.5865 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_phase7 | strict_ratio | Share of strict-comparable records among all valid records. | 0.3155 | runs/benchmarks/block3_20260203_225620_phase7 |
| block3_20260203_225620_phase7 | target_winners | Best strict model per target for this run. | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.35741397936, "model": "NHITS"}, "investors_count": {"category": "transformer_sota", "mae": 44.79163160827051, "model": "NBEATSx"}, "is_funded": {"category": "transformer_sota", "mae": 0.032280800947848576, "model": "PatchTST"}} | runs/benchmarks/block3_20260203_225620_phase7 |
| block3_20260203_225620_phase7_v71extreme_20260214_032205 | failure_tags | Failure taxonomy tags found in this run. | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 8} | docs/benchmarks/block3_truth_pack/failure_taxonomy.csv |
| block3_20260203_225620_phase7_v71extreme_20260214_032205 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 1.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_phase7_v71extreme_20260214_032205 | strict_ratio | Share of strict-comparable records among all valid records. | 1.0000 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205 |
| block3_20260203_225620_phase7_v71extreme_20260214_032205 | target_winners | Best strict model per target for this run. | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.35741397936, "model": "NHITS"}, "investors_count": {"category": "deep_classical", "mae": 44.726689230844215, "model": "NBEATS"}, "is_funded": {"category": "deep_classical", "mae": 0.03232223233914305, "model": "NHITS"}} | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205 |
| block3_20260203_225620_phase7_v71extreme_20260214_130737 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 1.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_phase7_v71extreme_20260214_130737 | strict_ratio | Share of strict-comparable records among all valid records. | 1.0000 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737 |
| block3_20260203_225620_phase7_v71extreme_20260214_130737 | target_winners | Best strict model per target for this run. | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.3574093943, "model": "NHITS"}, "investors_count": {"category": "deep_classical", "mae": 44.726689230844215, "model": "NBEATS"}, "is_funded": {"category": "deep_classical", "mae": 0.03224843655460842, "model": "NHITS"}} | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737 |
<!-- END AUTO:HISTORICAL_FULL_SCALE_EXPERIMENT_LEDGER -->

## AutoFit Version Ladder (V1->V7.2)

<!-- BEGIN AUTO:AUTOFIT_VERSION_LADDER -->
### Version Ladder

| version | commit_hint | core_changes | inspiration_source | measured_targets | median_mae_by_target_json | median_gap_vs_best_non_autofit_json | primary_failure_mode | evidence_path |
|---|---|---|---|---|---|---|---|---|
| AutoFitV1 | baseline_pre_phase2 | Data-driven best-single selection with residual correction. | Pragmatic stacked residual correction from tabular ensemble practice. | funding_raised_usd,investors_count | {"funding_raised_usd": 437133.2568725792, "investors_count": 125.57232995289112} | {"funding_raised_usd": 16.551530356678402, "investors_count": 180.22220034525606} | high_gap_vs_best_non_autofit | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV2 | 87baa13 | Top-K weighted ensemble by inverse validation MAE. | Classical weighted model averaging. | funding_raised_usd,investors_count | {"funding_raised_usd": 428570.3098318192, "investors_count": 223.17549131231723} | {"funding_raised_usd": 14.43341440788155, "investors_count": 397.74962629363387} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV2E | 87baa13 | Top-K stacking with LightGBM meta-learner. | Stacked generalization. | funding_raised_usd,investors_count | {"funding_raised_usd": 428569.9402728804, "investors_count": 223.17549131231723} | {"funding_raised_usd": 14.435231334913734, "investors_count": 397.74962629363387} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV3 | 320c314 | Temporal CV + greedy/exhaustive subset selection with diversity preference. | Temporal OOF stacking with subset search. | funding_raised_usd,investors_count | {"funding_raised_usd": 395551.53603914875, "investors_count": 166.50003118685348} | {"funding_raised_usd": 5.617096712484337, "investors_count": 271.3460103249283} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV3E | 320c314 | Top-K stacking variant under V3 temporal CV framework. | OOF-based stacking simplification. | funding_raised_usd,investors_count | {"funding_raised_usd": 429854.0511788266, "investors_count": 204.14996378872286} | {"funding_raised_usd": 14.779908898765193, "investors_count": 355.5729122390402} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV3Max | 320c314 | Exhaustive V3 search with bounded candidate budget. | Combinatorial ensemble search. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 396360.34859371965, "investors_count": 137.26189066319031, "is_funded": 0.10890193145407928} | {"funding_raised_usd": 5.784237218695543, "investors_count": 206.30815753453794, "is_funded": 237.14540893207106} | high_gap_vs_best_non_autofit | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV4 | c53abf6 | Target transforms + full OOF stacking + diversity-aware selection. | NCL and transform-aware robust stacking. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 428265.8942024873, "investors_count": 196.73861156988488, "is_funded": 0.09019224705267709} | {"funding_raised_usd": 14.30077913562794, "investors_count": 339.23059762196976, "is_funded": 179.22279811298966} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV5 | dce0ff9 | Regime-aware tiered evaluation and collapse detection guard. | Cost-aware routing with monotonic fallback guard. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 396360.34859371965, "investors_count": 252.99398560055954, "is_funded": 0.09947219709664773} | {"funding_raised_usd": 5.784237218695543, "investors_count": 464.5928030300439, "is_funded": 207.95224767581152} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV6 | ad07032 | Caruana-style greedy ensembling + two-layer stacking + conformal weighting. | AutoGluon-style weighted ensemble with robust transform. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 396360.34859371965, "investors_count": 300.71177668475053, "is_funded": 0.09818964911714113} | {"funding_raised_usd": 5.784237218695543, "investors_count": 571.0819804651214, "is_funded": 203.98165544432214} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV7 | faafdcf | Data-adapted robust ensemble with missingness/ratio features and repeated temporal CV. | SOTA tabular feature engineering + robust ensemble selection. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 397006.3727232514, "investors_count": 267.726448471727, "is_funded": 0.09068313538611147} | {"funding_raised_usd": 5.97849492584821, "investors_count": 497.47043246547696, "is_funded": 180.74252086047545} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV71 | phase7_v71_extreme_branch | Lane-adaptive routing, dynamic thresholds, count-safe mode, anchor and policy logging. | Target-family specialization with fairness-first guards. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 396399.79881869024, "investors_count": 274.4207316624478, "is_funded": 0.08716407382070858} | {"funding_raised_usd": 5.568690813313937, "investors_count": 512.658335534649, "is_funded": 169.91107236916724} | v71_count_explosion | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV72 | planned_not_materialized | Count-safe hardening + champion-anchor + offline policy layer (planned). | Evidence-driven v7.2 design gates. |  | {} | {} | planned_not_materialized | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |

### Step Deltas

| from_version | to_version | target | overlap_keys | median_mae_delta_pct | median_gap_delta_pct | evidence_path |
|---|---|---|---|---|---|---|
| AutoFitV1 | AutoFitV2 | funding_raised_usd | 17 | -1.961041 | -2.289330 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV1 | AutoFitV2 | investors_count | 5 | 77.726647 | 217.684912 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV1 | AutoFitV2 | is_funded | 0 |  |  | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2 | AutoFitV2E | funding_raised_usd | 17 | -0.000000 | -0.000000 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2 | AutoFitV2E | investors_count | 5 | 0.000000 | 0.000000 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2 | AutoFitV2E | is_funded | 0 |  |  | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2E | AutoFitV3 | funding_raised_usd | 17 | -7.704321 | -8.814025 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2E | AutoFitV3 | investors_count | 5 | -25.357559 | -126.323991 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2E | AutoFitV3 | is_funded | 0 |  |  | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3 | AutoFitV3E | funding_raised_usd | 17 | 8.672072 | 9.158879 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3 | AutoFitV3E | investors_count | 5 | 22.551033 | 83.869081 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3 | AutoFitV3E | is_funded | 0 |  |  | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3E | AutoFitV3Max | funding_raised_usd | 13 | -7.979787 | -9.158879 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3E | AutoFitV3Max | investors_count | 1 | -32.495409 | -147.367977 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3E | AutoFitV3Max | is_funded | 0 |  |  | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3Max | AutoFitV4 | funding_raised_usd | 20 | 8.049631 | 8.516542 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3Max | AutoFitV4 | investors_count | 9 | 43.330833 | 132.785207 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3Max | AutoFitV4 | is_funded | 2 | -17.180305 | -57.922611 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV4 | AutoFitV5 | funding_raised_usd | 20 | -7.449938 | -8.516542 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV4 | AutoFitV5 | investors_count | 9 | 28.593967 | 125.466695 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV4 | AutoFitV5 | is_funded | 2 | 10.289077 | 28.729450 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV5 | AutoFitV6 | funding_raised_usd | 20 | 0.000000 | 0.000000 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV5 | AutoFitV6 | investors_count | 9 | 18.861235 | 106.489177 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV5 | AutoFitV6 | is_funded | 2 | -1.289353 | -3.970592 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV6 | AutoFitV7 | funding_raised_usd | 20 | 0.162989 | 0.172534 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV6 | AutoFitV7 | investors_count | 9 | -10.969084 | -73.611548 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV6 | AutoFitV7 | is_funded | 2 | -7.644913 | -23.239135 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV7 | AutoFitV71 | funding_raised_usd | 20 | -0.157449 | -0.166827 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV7 | AutoFitV71 | investors_count | 9 | 2.500023 | 14.938281 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV7 | AutoFitV71 | is_funded | 2 | -3.880613 | -10.894531 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV71 | AutoFitV72 | funding_raised_usd | 0 |  |  | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV71 | AutoFitV72 | investors_count | 0 |  |  | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV71 | AutoFitV72 | is_funded | 0 |  |  | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
<!-- END AUTO:AUTOFIT_VERSION_LADDER -->

## High-Value SOTA Components For This Freeze

<!-- BEGIN AUTO:HIGH_VALUE_SOTA_COMPONENTS -->
| feature_component | winner_evidence | affected_subtasks | why_effective | integration_priority | risk | verification_test | evidence_path |
|---|---|---|---|---|---|---|---|
| Deep sequence inductive bias | investors_count winners by category: deep_classical=38, transformer_sota=6 | target_family:count; task2_forecast investors_count | Deep temporal models preserve count trajectory structure better than current AutoFit count lane. | high | Over-parameterized deep models may become unstable on sparse slices. | Track median gap reduction on investors_count under strict comparability. | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| Foundation-model global priors | funding_raised_usd winners by category: deep_classical=20, transformer_sota=18, foundation=6 | target_family:heavy_tail | Foundation/deep models are robust to heavy-tail dynamics and long-range temporal drift. | high | Inference cost and dependency drift across clusters. | Monitor heavy-tail lane median MAE and tail quantile errors. | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| Transformer representation for binary outcome | is_funded winners by model: PatchTST=8, NHITS=8 | target_family:binary; task1_outcome is_funded | Temporal representation improves ranking and calibration for binary outcome targets. | medium | Calibration drift when class balance shifts over time. | Track binary MAE/logloss/AUC drift by horizon under strict guard. | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| Count-safe postprocess hard guard | failure tags: autofit_gap_gt_100pct=60, v71_count_explosion=4 | target_family:count; core_edgar investors_count | Count explosions indicate inverse-transform and clipping chain needs hard rejection logic. | critical | Over-clipping can suppress true extreme events. | Count lane guard-hit accounting and no catastrophic MAE spikes. | docs/benchmarks/block3_truth_pack/failure_taxonomy.csv |
| Champion-anchor retention | AutoFit high-gap tags: 60 | all target families | Anchor retention prevents ensemble collapse to weak homogeneous candidate sets. | high | Anchor may overfit historical winner if guard is too permissive. | OOF guard and bounded degradation constraint on anchor injection. | docs/benchmarks/block3_truth_pack/failure_taxonomy.csv |
| Fairness and coverage comparability gate | Strict vs legacy layering enforced at record level. | all | Separating strict-comparable from legacy-unverified prevents invalid leaderboard conclusions. | critical | Historical results with missing guards become non-comparable. | Summary fields strict_records/legacy_records and strict condition coverage. | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
<!-- END AUTO:HIGH_VALUE_SOTA_COMPONENTS -->

## Primary Literature Matrix

<!-- BEGIN AUTO:PRIMARY_LITERATURE_MATRIX -->
Strict-primary sources only (arXiv originals, official release notes, official docs).

| topic | source | problem | core_mechanism | what_it_fixes | risk | integration_point | expected_gain | verification_test | primary_link | status |
|---|---|---|---|---|---|---|---|---|---|---|
| time_series_foundation | Chronos (arXiv:2403.07815) | Universal forecasting across heterogeneous time series with limited task-specific tuning. | Language-model style tokenization over scaled time series and seq2seq pretraining. | Weak cross-domain priors in sparse or noisy slices. | Tokenization mismatch on non-stationary count bursts. | Global prior features and champion-anchor candidates for heavy-tail/count lanes. | Improved robustness on low-signal and long-horizon regimes. | Strict comparable median MAE and tail quantile error by horizon. | https://arxiv.org/abs/2403.07815 | verified_primary |
| time_series_foundation | Chronos-2 (arXiv:2510.15821) | Unified large-scale pretrained forecasting for zero-shot and fine-tuned settings. | Chronos family second-generation architecture and training pipeline improvements. | Transfer degradation under distribution shift versus earlier Chronos variants. | Operational dependency drift and heavier inference footprint. | Reference foundation baseline and lane-aware anchor pool updates. | Better OOD resilience on full ablation matrix. | Condition-level win share in strict_comparable subset. | https://arxiv.org/abs/2510.15821 | verified_primary |
| time_series_foundation | Moirai-MoE (arXiv:2410.10469) | Handling diverse time-series regimes with scalable specialist capacity. | Sparse mixture-of-experts routing for temporal patterns. | Single-backbone underfitting for regime-diverse signals. | Route instability and training complexity. | Inspiration for offline routing policy and lane-specialist candidate pools. | Lower variance across task/ablation slices. | Variance of MAE across repeated strict runs and per-lane gap. | https://arxiv.org/abs/2410.10469 | verified_primary |
| time_series_foundation | TiRex (arXiv:2505.23719) | General-purpose time-series foundation transfer across tasks. | Unified representation learning and transfer-oriented objective design. | Task-specific over-specialization and weak reuse of temporal representations. | Potential mismatch between benchmark tasks and pretraining assumptions. | Representation-oriented features for lane routing and anchor diversity. | Stronger cross-task consistency on strict condition keys. | WinRate@Condition across tasks under fairness gate. | https://arxiv.org/abs/2505.23719 | verified_primary |
| time_series_foundation | Time-MoE (arXiv:2409.16040) | Scaling temporal models without fully dense compute cost. | Sparse expert activation and efficient MoE architecture for time series. | Throughput bottlenecks at large model scale. | Expert imbalance and route collapse. | Offline policy design and budgeted candidate routing in AutoFitV7.2. | Better performance-compute Pareto under fixed QOS constraints. | Compute_cost_report vs GlobalNormalizedMAE under strict filters. | https://arxiv.org/abs/2409.16040 | verified_primary |
| time_series_llm | Time-LLM (arXiv:2310.01728) | Injecting semantic priors from LLMs into temporal forecasting. | Reprogramming time series into LLM-compatible prompt/token spaces. | Weak contextual semantics in pure numerical models. | Prompt sensitivity and unstable gains across slices. | Auxiliary retrieval/regime descriptors and policy features (offline only). | Improved binary/count subtasks with richer context encoding. | Ablation against baseline retrieval-disabled setting. | https://arxiv.org/abs/2310.01728 | verified_primary |
| time_series_retrieval | Retrieval-Augmented Forecasting (arXiv:2505.04163) | Model adaptation to recurring regimes and non-stationary contexts. | Retrieve analogous historical windows and condition predictions on retrieved context. | Context-loss in long-history or regime-switch settings. | Leakage if retrieval index uses future slices. | Train-only prototype bank for regime retrieval features in AutoFit lanes. | Better robustness in task3_risk_adjust and long-horizon slices. | Leakage audit + OOD slice degradation_pct delta. | https://arxiv.org/abs/2505.04163 | verified_primary |
| time_series_multimodal | TimeOmni-1 (arXiv:2509.24803) | Unified multimodal time-series representation with long context. | Cross-modal alignment with long-context modeling. | Fragmented handling of text/edgar/core modalities. | High training/inference complexity for production benchmark loops. | Guidance for modality-aware feature fusion and lane gating. | More stable full-ablation gains over core-only baselines. | full vs core_only uplift consistency under strict keys. | https://arxiv.org/abs/2509.24803 | verified_primary |
| rl_policy | LangTime (arXiv:2506.10630) | Improving forecast decisions via policy optimization rather than static heuristics. | PPO-based policy optimization for time-series decision flow. | Rigid static routing/weighting in changing regimes. | Test leakage if policy is updated on held-out feedback. | Offline-only policy action selection from train/OOF trajectories. | Higher condition-level win rate with bounded risk. | Policy leak audit + reproducible policy_action_id telemetry. | https://arxiv.org/abs/2506.10630 | verified_primary |
| tree_sota | XGBoost 3.2.0 Release Notes | Scalable tabular learning with improved infrastructure/runtime behavior. | Core library and ecosystem updates in official release pipeline. | Version-drift and stale baseline reproducibility. | Behavior shifts across minor/major upgrades. | Pinned baseline refresh for ml_tabular comparison lanes. | Cleaner baseline, fewer silent incompatibilities. | Version-pinned smoke + strict benchmark delta log. | https://xgboost.readthedocs.io/en/stable/changes/v3.2.0.html | verified_primary |
| tree_sota | LightGBM Releases (official GitHub) | Maintaining strong tabular baseline quality and compatibility. | Official incremental improvements through released versions. | Outdated baseline behavior and dependency mismatch. | Regression risk if upgrading without gate checks. | Version-audited baseline model in ml_tabular family. | Stable and reproducible tabular baseline comparisons. | Gate S smoke + strict comparable delta tracking. | https://github.com/microsoft/LightGBM/releases | verified_primary |
| tabular_non_tree | TabPFN docs (official) | Fast strong tabular generalization without heavy per-task tuning. | Prior-data fitted network inference for tabular tasks. | Weak performance of homogeneous tree-only candidate sets on some slices. | Dependency and hardware compatibility limits. | Optional candidate injection in binary/heavy-tail lanes with graceful fallback. | Diversity uplift and anti-collapse benefits. | Admission gate + fallback_fraction telemetry. | https://priorlabs.ai/docs/getting-started | verified_primary |
<!-- END AUTO:PRIMARY_LITERATURE_MATRIX -->

## Citation Correction Log

<!-- BEGIN AUTO:CITATION_CORRECTION_LOG -->
Unverified or mismatched legacy references are explicitly corrected or marked as hypothesis.

| reference_item | previous_claim | verification_result | action | primary_link | status |
|---|---|---|---|---|---|
| Chronos-2 | arXiv:2503.06548 | Mismatch: arXiv:2503.06548 is unrelated to Chronos-2. | Use verified Chronos-2 link. | https://arxiv.org/abs/2510.15821 | corrected |
| TiRex | arXiv:2502.13995 | Mismatch: arXiv:2502.13995 is unrelated. | Use verified TiRex link. | https://arxiv.org/abs/2505.23719 | corrected |
| TimeOmni-1 | arXiv:2502.15638 | Mismatch: arXiv:2502.15638 is unrelated. | Use verified TimeOmni-1 link. | https://arxiv.org/abs/2509.24803 | corrected |
| DRLTSF | arXiv:2508.07481 | Unverified title/ID mapping from primary source. | Keep as hypothesis only; exclude from design-critical evidence. | https://arxiv.org/search/?query=DRLTSF&searchtype=all | hypothesis |
| Moirai2 naming | arXiv:2410.10469 referenced as Moirai2 | Primary title identifies Moirai-MoE; mapping to 'Moirai2' label should be treated as alias. | Use primary title in evidence table and keep alias explicitly marked. | https://arxiv.org/abs/2410.10469 | clarified |
<!-- END AUTO:CITATION_CORRECTION_LOG -->

## Live Slurm Snapshot

<!-- BEGIN AUTO:LIVE_SLURM_SNAPSHOT -->
Snapshot timestamp is absolute and all queue conclusions are time-bounded.

| metric | value | evidence_path |
|---|---|---|
| snapshot_ts | 2026-02-18T15:20:58.887178+00:00 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_total | 7 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_total | 103 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_by_partition | {"batch": 7} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_by_partition | {"batch": 95, "gpu": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

### Pending Reasons

| reason | count | evidence_path |
|---|---|---|
| (Priority) | 95 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| (QOSGrpNodeLimit) | 8 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

### Prefix Status (p7/p7r/p7x/p7xF)

| prefix | source | state_counts | evidence_path |
|---|---|---|---|
| p7 | squeue | {"PENDING": 5, "RUNNING": 7} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | squeue | {"PENDING": 2} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | squeue | {"PENDING": 66} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | squeue | {"PENDING": 30} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7 | sacct | {"CANCELLED": 66, "COMPLETED": 170, "FAILED": 8, "OUT_OF_ME": 4, "PENDING": 5, "RUNNING": 7} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | sacct | {"PENDING": 2} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | sacct | {"COMPLETED": 55, "PENDING": 66} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | sacct | {"COMPLETED": 47, "PENDING": 30} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

### Collection Commands

```bash
squeue -u $USER -h -o "%T %j %P %R"
sacct -u $USER -S 2026-02-12 -n -X -o JobName,State
sacctmgr show qos iris-batch-long,iris-gpu-long format=Name,MaxJobsPU,MaxWall,Priority -P -n
```
<!-- END AUTO:LIVE_SLURM_SNAPSHOT -->

## Validation Gates / Risk & Rollback

### Stage S (Smoke/Audit)

1. Preflight gate must pass.
2. Leakage audit must pass.
3. Coverage/fairness audit must pass.
4. Count-safe and policy logging tests must pass.

### Stage P (Pilot)

1. `fairness_pass = 100%`
2. `investors_count` median gap vs V7 reduced by at least 50%
3. `GlobalNormalizedMAE` improved by at least 8% vs V7

### Stage F (Full)

1. Condition-level win rate vs V7 at least 70%
2. No target with median degradation worse than 3%
3. No new fairness/leakage anomalies

### Risks

1. Count-safe clipping can over-constrain true extremes; keep lane-level toggle.
2. Anchor injection can overfit historical winners; keep bounded OOF degradation guard.
3. Policy layer can create hidden leakage paths; keep offline-only policy and explicit logging.
4. Dependency drift can break reproducibility; keep release-note and seed audits.
