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
| bench_dirs | runs/benchmarks/block3_20260203_225620_iris;runs/benchmarks/block3_20260203_225620_iris_phase3;runs/benchmarks/block3_20260203_225620_iris_full;runs/benchmarks/block3_20260203_225620_iris_phase7;runs/benchmarks/block3_20260203_225620_phase7;runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205;runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737;runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_172651;runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_173453;runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_182303;runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_185618;runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_191536;runs/benchmarks/block3_20260203_225620_phase7_v72_4090_20260219_173137;runs/benchmarks/block3_20260203_225620_phase7_v72_failure_pool_rerun_heavy_20260219_v72fix_sched | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| raw_records | 14618 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| strict_records | 5236 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| legacy_unverified_records | 9382 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| strict_condition_completion | 104/104 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| legacy_condition_completion | 104/104 | docs/benchmarks/block3_truth_pack/truth_pack_summary.json |
| v71_win_rate_vs_v7 | 0.423077 | docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv |
| v71_median_relative_gain_vs_v7_pct | -0.333876 | docs/benchmarks/block3_truth_pack/v71_vs_v7_overlap.csv |
| critical_failures | 4 | docs/benchmarks/block3_truth_pack/failure_taxonomy.csv |
| v72_pilot_overall_pass | false | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| v72_pilot_overlap_keys | 24 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
<!-- END AUTO:EVIDENCE_SNAPSHOT -->

## Audit Gates Snapshot

<!-- BEGIN AUTO:AUDIT_GATES -->
Gate snapshot is sourced from latest read-only audit artifacts.

| audit | generated_at_utc | overall_pass | key_signal | evidence_path |
|---|---|---|---|---|
| data_integrity | 2026-02-18T23:03:03.433891+00:00 | true | {"embargo_non_negative": true, "train_before_val": true, "val_before_test": true} | docs/benchmarks/block3_truth_pack/data_integrity_audit_latest.json |
| data_integrity.freeze_gate | 2026-02-18T23:03:03.433891+00:00 | true | {"exit_code": 0, "fallback_mode": null, "n_checks": 5} | docs/benchmarks/block3_truth_pack/data_integrity_audit_latest.json |
| investors_count_stability | 2026-02-18T23:03:21.202374+00:00 | false | {"catastrophic_spikes": 4, "guard_telemetry": {"anchor_models_used_topk": [], "inverse_transform_guard_hits_total": 0, "lane_clip_rate_median": 0.0, "n_rows": 88, "oof_guard_triggered_count": 0, "policy_action_id_topk": []}, "strict_record_count": 1675} | docs/benchmarks/block3_truth_pack/investors_count_stability_audit_latest.json |
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
| task1_outcome | core_only | investors_count | 1 | true | true | true | KAN | transformer_sota | 44.703800 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | investors_count | 7 | true | true | true | KAN | transformer_sota | 44.692755 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | is_funded | 1 | true | true | true | NHITS | deep_classical | 0.032965 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | is_funded | 7 | true | true | true | PatchTST | transformer_sota | 0.033024 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | is_funded | 14 | true | true | true | PatchTST | transformer_sota | 0.032960 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_only | is_funded | 30 | true | true | true | NHITS | deep_classical | 0.032970 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | funding_raised_usd | 1 | true | true | true | NBEATS | deep_classical | 380659.460 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | funding_raised_usd | 7 | true | true | true | NHITS | deep_classical | 380577.133 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | funding_raised_usd | 14 | true | true | true | PatchTST | transformer_sota | 381200.561 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | funding_raised_usd | 30 | true | true | true | PatchTST | transformer_sota | 381617.171 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | investors_count | 1 | true | true | true | KAN | transformer_sota | 44.703800 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | investors_count | 7 | true | true | true | KAN | transformer_sota | 44.692755 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | investors_count | 14 | true | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | investors_count | 30 | true | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | is_funded | 1 | true | true | true | NHITS | deep_classical | 0.032959 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | is_funded | 7 | true | true | true | PatchTST | transformer_sota | 0.033024 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome | core_text | is_funded | 14 | true | true | true | PatchTST | transformer_sota | 0.032960 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
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
| autofit | 12 | 12 | 1.000000 | AutoFitV1;AutoFitV2;AutoFitV2E;AutoFitV3;AutoFitV3E;AutoFitV3Max;AutoFitV4;AutoFitV5;AutoFitV6;AutoFitV7;AutoFitV71;AutoFitV72 |  | 0 | 0.000000 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| deep_classical | 4 | 4 | 1.000000 | DeepAR;NBEATS;NHITS;TFT |  | 62 | 0.596154 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| foundation | 11 | 10 | 0.909091 | Chronos;ChronosBolt;LagLlama;MOMENT;Moirai;Moirai2;MoiraiLarge;TimeMoE;Timer;TimesFM | Chronos2 | 6 | 0.057692 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| irregular | 2 | 0 | 0.000000 |  | GRU-D;SAITS | 0 | 0.000000 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| ml_tabular | 20 | 17 | 0.850000 | CatBoost;ElasticNet;ExtraTrees;HistGradientBoosting;KNN;Lasso;LightGBM;LightGBMTweedie;LogisticRegression;MeanPredictor;QuantileRegressor;RandomForest;Ridge;SVR;SeasonalNaive;XGBoost;XGBoostPoisson | NegativeBinomialGLM;TabPFNClassifier;TabPFNRegressor | 0 | 0.000000 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| statistical | 5 | 5 | 1.000000 | AutoARIMA;AutoETS;AutoTheta;MSTL;SF_SeasonalNaive |  | 0 | 0.000000 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| transformer_sota | 20 | 20 | 1.000000 | Autoformer;BiTCN;DLinear;FEDformer;Informer;KAN;NBEATSx;NLinear;PatchTST;RMoK;SOFTS;StemGNN;TSMixer;TSMixerx;TiDE;TimeMixer;TimeXer;TimesNet;VanillaTransformer;iTransformer |  | 36 | 0.346154 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
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
| task1_outcome__core_only__investors_count__h1 | task1_outcome | core_only | investors_count | count | 1 | true | true | KAN | transformer_sota | 44.703800 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__investors_count__h7 | task1_outcome | core_only | investors_count | count | 7 | true | true | KAN | transformer_sota | 44.692755 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__investors_count__h14 | task1_outcome | core_only | investors_count | count | 14 | true | true | NBEATS | deep_classical | 44.734036 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__investors_count__h30 | task1_outcome | core_only | investors_count | count | 30 | true | true | NBEATS | deep_classical | 44.746757 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__investors_count__h1 | task1_outcome | core_text | investors_count | count | 1 | true | true | KAN | transformer_sota | 44.703800 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__investors_count__h7 | task1_outcome | core_text | investors_count | count | 7 | true | true | KAN | transformer_sota | 44.692755 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
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
| task1_outcome__core_only__is_funded__h1 | task1_outcome | core_only | is_funded | binary | 1 | true | true | NHITS | deep_classical | 0.032965 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__is_funded__h7 | task1_outcome | core_only | is_funded | binary | 7 | true | true | PatchTST | transformer_sota | 0.033024 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__is_funded__h14 | task1_outcome | core_only | is_funded | binary | 14 | true | true | PatchTST | transformer_sota | 0.032960 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_only__is_funded__h30 | task1_outcome | core_only | is_funded | binary | 30 | true | true | NHITS | deep_classical | 0.032970 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__is_funded__h1 | task1_outcome | core_text | is_funded | binary | 1 | true | true | NHITS | deep_classical | 0.032959 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__is_funded__h7 | task1_outcome | core_text | is_funded | binary | 7 | true | true | PatchTST | transformer_sota | 0.033024 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| task1_outcome__core_text__is_funded__h14 | task1_outcome | core_text | is_funded | binary | 14 | true | true | PatchTST | transformer_sota | 0.032960 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
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
| investors_count | count | 1 | NBEATS | deep_classical | 28 | 0.636364 | 44 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| investors_count | count | 2 | KAN | transformer_sota | 7 | 0.159091 | 44 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| investors_count | count | 3 | NHITS | deep_classical | 6 | 0.136364 | 44 | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
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
| investors_count | count | autofit | AutoFitV6 | 114.116541 | KAN | transformer_sota | 44.692755 | 155.335657 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/autofit_baseline/task2_forecast/autofit/core_text/metrics.json |
| investors_count | count | deep_classical | NBEATS | 44.726689 | KAN | transformer_sota | 44.692755 | 0.075927 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/deep_refs/task1_outcome/deep_classical/core_only/metrics.json |
| investors_count | count | foundation | ChronosBolt | 44.992359 | KAN | transformer_sota | 44.692755 | 0.670361 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| investors_count | count | ml_tabular | RandomForest | 95.599464 | KAN | transformer_sota | 44.692755 | 113.903713 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/ml_refs/task2_forecast/ml_tabular/core_only/metrics.json |
| investors_count | count | statistical | MSTL | 48.222248 | KAN | transformer_sota | 44.692755 | 7.897236 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| investors_count | count | transformer_sota | KAN | 44.692755 | KAN | transformer_sota | 44.692755 | 0.000000 | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_191536/task1_outcome/transformer_sota_B/core_only/metrics.json |
| is_funded | binary | autofit | AutoFitV71 | 0.086412 | NHITS | deep_classical | 0.032248 | 167.958234 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/v71_g02/task1_outcome/autofit/core_text/metrics.json |
| is_funded | binary | deep_classical | NHITS | 0.032248 | NHITS | deep_classical | 0.032248 | 0.000000 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737/full/deep_refs/task1_outcome/deep_classical/core_edgar/metrics.json |
| is_funded | binary | foundation | ChronosBolt | 0.033394 | NHITS | deep_classical | 0.032248 | 3.551434 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| is_funded | binary | ml_tabular | ExtraTrees | 0.065454 | NHITS | deep_classical | 0.032248 | 102.966498 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/ml_refs/task1_outcome/ml_tabular/core_edgar/metrics.json |
| is_funded | binary | statistical | SF_SeasonalNaive | 0.046009 | NHITS | deep_classical | 0.032248 | 42.670935 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| is_funded | binary | transformer_sota | PatchTST | 0.032281 | NHITS | deep_classical | 0.032248 | 0.100360 | runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
<!-- END AUTO:FAMILY_GAP_MATRIX -->

## Champion Template Library

<!-- BEGIN AUTO:CHAMPION_TEMPLATE_LIBRARY -->
| template_id | target_family | horizon_band | ablation | primary_anchor | backup_anchors | n_conditions | winner_distribution_json | failure_signals_json | evidence_path |
|---|---|---|---|---|---|---|---|---|---|
| binary__long__core_edgar | binary | long | core_edgar | NHITS |  | 1 | {"NHITS": {"win_rate": 1.0, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| binary__long__core_only | binary | long | core_only | NHITS |  | 1 | {"NHITS": {"win_rate": 1.0, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| binary__long__core_text | binary | long | core_text | NHITS |  | 1 | {"NHITS": {"win_rate": 1.0, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| binary__long__full | binary | long | full | NHITS |  | 1 | {"NHITS": {"win_rate": 1.0, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| binary__mid__core_edgar | binary | mid | core_edgar | PatchTST |  | 1 | {"PatchTST": {"win_rate": 1.0, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| binary__mid__core_only | binary | mid | core_only | PatchTST |  | 1 | {"PatchTST": {"win_rate": 1.0, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| binary__mid__core_text | binary | mid | core_text | PatchTST |  | 1 | {"PatchTST": {"win_rate": 1.0, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| binary__mid__full | binary | mid | full | PatchTST |  | 1 | {"PatchTST": {"win_rate": 1.0, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| binary__short__core_edgar | binary | short | core_edgar | PatchTST | NHITS | 2 | {"NHITS": {"win_rate": 0.5, "wins": 1}, "PatchTST": {"win_rate": 0.5, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| binary__short__core_only | binary | short | core_only | NHITS | PatchTST | 2 | {"NHITS": {"win_rate": 0.5, "wins": 1}, "PatchTST": {"win_rate": 0.5, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| binary__short__core_text | binary | short | core_text | NHITS | PatchTST | 2 | {"NHITS": {"win_rate": 0.5, "wins": 1}, "PatchTST": {"win_rate": 0.5, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| binary__short__full | binary | short | full | PatchTST | NHITS | 2 | {"NHITS": {"win_rate": 0.5, "wins": 1}, "PatchTST": {"win_rate": 0.5, "wins": 1}} | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__long__core_edgar | count | long | core_edgar | NBEATS |  | 3 | {"NBEATS": {"win_rate": 1.0, "wins": 3}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__long__core_only | count | long | core_only | NBEATS |  | 3 | {"NBEATS": {"win_rate": 1.0, "wins": 3}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__long__core_text | count | long | core_text | NBEATS |  | 2 | {"NBEATS": {"win_rate": 1.0, "wins": 2}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__long__full | count | long | full | NBEATS | NBEATSx | 3 | {"NBEATS": {"win_rate": 0.6666666666666666, "wins": 2}, "NBEATSx": {"win_rate": 0.3333333333333333, "wins": 1}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__mid__core_edgar | count | mid | core_edgar | NBEATS |  | 3 | {"NBEATS": {"win_rate": 1.0, "wins": 3}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__mid__core_only | count | mid | core_only | NBEATS |  | 3 | {"NBEATS": {"win_rate": 1.0, "wins": 3}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__mid__core_text | count | mid | core_text | NBEATS |  | 2 | {"NBEATS": {"win_rate": 1.0, "wins": 2}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__mid__full | count | mid | full | NBEATS | NBEATSx | 3 | {"NBEATS": {"win_rate": 0.6666666666666666, "wins": 2}, "NBEATSx": {"win_rate": 0.3333333333333333, "wins": 1}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__short__core_edgar | count | short | core_edgar | NHITS | NBEATS | 6 | {"NBEATS": {"win_rate": 0.5, "wins": 3}, "NHITS": {"win_rate": 0.5, "wins": 3}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__short__core_only | count | short | core_only | KAN | NHITS,NBEATS | 6 | {"KAN": {"win_rate": 0.3333333333333333, "wins": 2}, "NBEATS": {"win_rate": 0.3333333333333333, "wins": 2}, "NHITS": {"win_rate": 0.3333333333333333, "wins": 2}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__short__core_text | count | short | core_text | KAN | NHITS,NBEATS | 4 | {"KAN": {"win_rate": 0.5, "wins": 2}, "NBEATS": {"win_rate": 0.25, "wins": 1}, "NHITS": {"win_rate": 0.25, "wins": 1}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| count__short__full | count | short | full | KAN | NBEATS,NBEATSx | 6 | {"KAN": {"win_rate": 0.5, "wins": 3}, "NBEATS": {"win_rate": 0.3333333333333333, "wins": 2}, "NBEATSx": {"win_rate": 0.16666666666666666, "wins": 1}} | {"autofit_gap_gt_100pct": 44, "v71_count_explosion": 4} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__long__core_edgar | heavy_tail | long | core_edgar | PatchTST |  | 3 | {"PatchTST": {"win_rate": 1.0, "wins": 3}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__long__core_only | heavy_tail | long | core_only | PatchTST |  | 3 | {"PatchTST": {"win_rate": 1.0, "wins": 3}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__long__core_text | heavy_tail | long | core_text | PatchTST |  | 2 | {"PatchTST": {"win_rate": 1.0, "wins": 2}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__long__full | heavy_tail | long | full | Chronos |  | 3 | {"Chronos": {"win_rate": 1.0, "wins": 3}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__mid__core_edgar | heavy_tail | mid | core_edgar | PatchTST |  | 3 | {"PatchTST": {"win_rate": 1.0, "wins": 3}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__mid__core_only | heavy_tail | mid | core_only | PatchTST |  | 3 | {"PatchTST": {"win_rate": 1.0, "wins": 3}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__mid__core_text | heavy_tail | mid | core_text | PatchTST |  | 2 | {"PatchTST": {"win_rate": 1.0, "wins": 2}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__mid__full | heavy_tail | mid | full | Chronos |  | 3 | {"Chronos": {"win_rate": 1.0, "wins": 3}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__short__core_edgar | heavy_tail | short | core_edgar | NBEATS | NHITS | 6 | {"NBEATS": {"win_rate": 0.5, "wins": 3}, "NHITS": {"win_rate": 0.5, "wins": 3}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__short__core_only | heavy_tail | short | core_only | NBEATS | NHITS | 6 | {"NBEATS": {"win_rate": 0.5, "wins": 3}, "NHITS": {"win_rate": 0.5, "wins": 3}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__short__core_text | heavy_tail | short | core_text | NBEATS | NHITS | 4 | {"NBEATS": {"win_rate": 0.5, "wins": 2}, "NHITS": {"win_rate": 0.5, "wins": 2}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| heavy_tail__short__full | heavy_tail | short | full | NHITS | NBEATSx,NBEATS | 6 | {"NBEATS": {"win_rate": 0.16666666666666666, "wins": 1}, "NBEATSx": {"win_rate": 0.3333333333333333, "wins": 2}, "NHITS": {"win_rate": 0.5, "wins": 3}} | {} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
<!-- END AUTO:CHAMPION_TEMPLATE_LIBRARY -->

## Hyperparameter Search Ledger

<!-- BEGIN AUTO:HYPERPARAMETER_SEARCH_LEDGER -->
| target | target_family | priority_rank | model_name | category | search_budget | trials_executed | status | best_mae_observed_strict | best_config_json | search_space_json | selection_scope | evidence_path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| investors_count | count | 1 | AutoFitV72 | unknown | 96 | 0 | planned |  | {"champion_anchor": true, "count_safe_mode": true, "dynamic_weighting": true, "search_budget": 96, "top_k": 8} | {"champion_anchor": [true], "count_safe_mode": [true], "dynamic_weighting": [true], "search_budget": [96], "top_k": [6, 8, 10]} | train_val_oof_only | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| investors_count | count | 2 | NHITS | deep_classical | 96 | 0 | planned_with_evidence | 44.738015213673336 | {"dropout_prob_theta": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout_prob_theta": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/deep_refs/task1_outcome/deep_classical/core_only/metrics.json |
| investors_count | count | 3 | NBEATS | deep_classical | 96 | 0 | planned_with_evidence | 44.726689230844215 | {"dropout_prob_theta": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout_prob_theta": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/deep_refs/task1_outcome/deep_classical/core_only/metrics.json |
| investors_count | count | 4 | PatchTST | transformer_sota | 96 | 0 | planned_with_evidence | 44.77180869366277 | {"dropout": 0.1, "hidden_size": 128, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout": [0.0, 0.1, 0.2], "hidden_size": [64, 128, 256], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/ts_refs/task1_outcome/transformer_sota/core_only/metrics.json |
| investors_count | count | 5 | iTransformer | transformer_sota | 96 | 0 | planned_with_evidence | 119.25932666022587 | {"d_model": 128, "dropout": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"d_model": [64, 128, 256], "dropout": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/ts_refs/task2_forecast/transformer_sota/core_edgar/metrics.json |
| investors_count | count | 6 | XGBoostPoisson | ml_tabular | 96 | 0 | planned_with_evidence | 175.6918382596547 | {"learning_rate": 0.03, "max_depth": 6, "min_child_weight": 2.0, "subsample": 0.9} | {"learning_rate": [0.01, 0.03, 0.05], "max_depth": [4, 6, 8], "min_child_weight": [1.0, 2.0, 4.0], "subsample": [0.7, 0.9, 1.0]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | count | 7 | LightGBMTweedie | ml_tabular | 96 | 0 | planned_with_evidence | 199.17120257060444 | {"learning_rate": 0.03, "min_child_samples": 40, "num_leaves": 127, "tweedie_variance_power": 1.3} | {"learning_rate": [0.01, 0.03, 0.05], "min_child_samples": [20, 40, 80], "num_leaves": [63, 127, 255], "tweedie_variance_power": [1.1, 1.3, 1.5]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | count | 8 | LightGBM | ml_tabular | 96 | 0 | planned_with_evidence | 199.17120257060444 | {"bagging_fraction": 0.9, "feature_fraction": 0.85, "learning_rate": 0.03, "min_child_samples": 40, "num_leaves": 127} | {"bagging_fraction": [0.7, 0.9, 1.0], "feature_fraction": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "min_child_samples": [20, 40, 80], "num_leaves": [63, 127, 255]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | count | 9 | XGBoost | ml_tabular | 96 | 0 | planned_with_evidence | 175.6918382596547 | {"colsample_bytree": 0.85, "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 2.0, "subsample": 0.85} | {"colsample_bytree": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "max_depth": [6, 8, 10], "min_child_weight": [1.0, 2.0, 4.0], "subsample": [0.7, 0.85, 1.0]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | count | 10 | CatBoost | ml_tabular | 96 | 0 | planned_with_evidence | 505.3142265932834 | {"bagging_temperature": 1.0, "depth": 8, "l2_leaf_reg": 6.0, "learning_rate": 0.03} | {"bagging_temperature": [0.0, 1.0, 2.0], "depth": [6, 8, 10], "l2_leaf_reg": [3.0, 6.0, 12.0], "learning_rate": [0.01, 0.03, 0.05]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | count | 11 | Moirai2 | foundation | 96 | 0 | planned_with_evidence | 228.7646966347054 | {"num_samples": 50, "temperature": 0.9} | {"num_samples": [20, 50, 100], "prediction_length": [1, 7, 14, 30], "temperature": [0.7, 0.9, 1.0]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| investors_count | count | 12 | Chronos | foundation | 96 | 0 | planned_with_evidence | 45.554431746837324 | {"num_samples": 50, "temperature": 0.9} | {"num_samples": [20, 50, 100], "prediction_length": [1, 7, 14, 30], "temperature": [0.7, 0.9, 1.0]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| funding_raised_usd | heavy_tail | 1 | AutoFitV72 | unknown | 96 | 0 | planned |  | {"champion_anchor": true, "count_safe_mode": true, "dynamic_weighting": true, "search_budget": 96, "top_k": 8} | {"champion_anchor": [true], "count_safe_mode": [true], "dynamic_weighting": [true], "search_budget": [96], "top_k": [6, 8, 10]} | train_val_oof_only | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| funding_raised_usd | heavy_tail | 2 | PatchTST | transformer_sota | 96 | 0 | planned_with_evidence | 374564.44013368135 | {"dropout": 0.1, "hidden_size": 128, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout": [0.0, 0.1, 0.2], "hidden_size": [64, 128, 256], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| funding_raised_usd | heavy_tail | 3 | NHITS | deep_classical | 96 | 0 | planned_with_evidence | 374432.3574093943 | {"dropout_prob_theta": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout_prob_theta": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737/full/deep_refs/task1_outcome/deep_classical/core_edgar/metrics.json |
| funding_raised_usd | heavy_tail | 4 | NBEATS | deep_classical | 96 | 0 | planned_with_evidence | 374514.6840344768 | {"dropout_prob_theta": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout_prob_theta": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737/full/deep_refs/task3_risk_adjust/deep_classical/full/metrics.json |
| funding_raised_usd | heavy_tail | 5 | Chronos | foundation | 96 | 0 | planned_with_evidence | 374610.31410290516 | {"num_samples": 50, "temperature": 0.9} | {"num_samples": [20, 50, 100], "prediction_length": [1, 7, 14, 30], "temperature": [0.7, 0.9, 1.0]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| funding_raised_usd | heavy_tail | 6 | Moirai2 | foundation | 96 | 0 | planned_with_evidence | 446570.6359553115 | {"num_samples": 50, "temperature": 0.9} | {"num_samples": [20, 50, 100], "prediction_length": [1, 7, 14, 30], "temperature": [0.7, 0.9, 1.0]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/foundation_refs/task1_outcome/foundation/core_only/metrics.json |
| funding_raised_usd | heavy_tail | 7 | LightGBM | ml_tabular | 96 | 0 | planned_with_evidence | 510180.29051876476 | {"bagging_fraction": 0.9, "feature_fraction": 0.85, "learning_rate": 0.03, "min_child_samples": 40, "num_leaves": 127} | {"bagging_fraction": [0.7, 0.9, 1.0], "feature_fraction": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "min_child_samples": [20, 40, 80], "num_leaves": [63, 127, 255]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task3_risk_adjust/ml_tabular/core_edgar/metrics.json |
| funding_raised_usd | heavy_tail | 8 | XGBoost | ml_tabular | 96 | 0 | planned_with_evidence | 477212.6570808882 | {"colsample_bytree": 0.85, "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 2.0, "subsample": 0.85} | {"colsample_bytree": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "max_depth": [6, 8, 10], "min_child_weight": [1.0, 2.0, 4.0], "subsample": [0.7, 0.85, 1.0]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| funding_raised_usd | heavy_tail | 9 | CatBoost | ml_tabular | 96 | 0 | planned_with_evidence | 586123.943694678 | {"bagging_temperature": 1.0, "depth": 8, "l2_leaf_reg": 6.0, "learning_rate": 0.03} | {"bagging_temperature": [0.0, 1.0, 2.0], "depth": [6, 8, 10], "l2_leaf_reg": [3.0, 6.0, 12.0], "learning_rate": [0.01, 0.03, 0.05]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| funding_raised_usd | heavy_tail | 10 | TabPFNRegressor | unknown | 96 | 0 | planned |  | {"ensemble_size": 16, "subsample": 10000} | {"ensemble_size": [8, 16, 32], "n_estimators": [1], "subsample": [5000, 10000, 20000]} | train_val_oof_only | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| is_funded | binary | 1 | AutoFitV72 | unknown | 96 | 0 | planned |  | {"champion_anchor": true, "count_safe_mode": true, "dynamic_weighting": true, "search_budget": 96, "top_k": 8} | {"champion_anchor": [true], "count_safe_mode": [true], "dynamic_weighting": [true], "search_budget": [96], "top_k": [6, 8, 10]} | train_val_oof_only | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| is_funded | binary | 2 | PatchTST | transformer_sota | 96 | 0 | planned_with_evidence | 0.032280800947848576 | {"dropout": 0.1, "hidden_size": 128, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout": [0.0, 0.1, 0.2], "hidden_size": [64, 128, 256], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| is_funded | binary | 3 | NHITS | deep_classical | 96 | 0 | planned_with_evidence | 0.03224843655460842 | {"dropout_prob_theta": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout_prob_theta": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737/full/deep_refs/task1_outcome/deep_classical/core_edgar/metrics.json |
| is_funded | binary | 4 | LightGBM | ml_tabular | 96 | 0 | planned_with_evidence | 0.09678662744688599 | {"bagging_fraction": 0.9, "feature_fraction": 0.85, "learning_rate": 0.03, "min_child_samples": 40, "num_leaves": 127} | {"bagging_fraction": [0.7, 0.9, 1.0], "feature_fraction": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "min_child_samples": [20, 40, 80], "num_leaves": [63, 127, 255]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| is_funded | binary | 5 | XGBoost | ml_tabular | 96 | 0 | planned_with_evidence | 0.09905975714461265 | {"colsample_bytree": 0.85, "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 2.0, "subsample": 0.85} | {"colsample_bytree": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "max_depth": [6, 8, 10], "min_child_weight": [1.0, 2.0, 4.0], "subsample": [0.7, 0.85, 1.0]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| is_funded | binary | 6 | CatBoost | ml_tabular | 96 | 0 | planned_with_evidence | 0.10531176937117738 | {"bagging_temperature": 1.0, "depth": 8, "l2_leaf_reg": 6.0, "learning_rate": 0.03} | {"bagging_temperature": [0.0, 1.0, 2.0], "depth": [6, 8, 10], "l2_leaf_reg": [3.0, 6.0, 12.0], "learning_rate": [0.01, 0.03, 0.05]} | train_val_oof_only | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| is_funded | binary | 7 | TabPFNClassifier | unknown | 96 | 0 | planned |  | {"ensemble_size": 16, "subsample": 10000} | {"ensemble_size": [8, 16, 32], "n_estimators": [1], "subsample": [5000, 10000, 20000]} | train_val_oof_only | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
<!-- END AUTO:HYPERPARAMETER_SEARCH_LEDGER -->

## Best Config By Model/Target

<!-- BEGIN AUTO:BEST_CONFIG_BY_MODEL_TARGET -->
| target | model_name | target_family | category | status | search_budget | trials_executed | best_mae_observed_strict | best_config_json | search_space_json | evidence_path |
|---|---|---|---|---|---|---|---|---|---|---|
| funding_raised_usd | AutoFitV72 | heavy_tail | unknown | planned | 96 | 0 |  | {"champion_anchor": true, "count_safe_mode": true, "dynamic_weighting": true, "search_budget": 96, "top_k": 8} | {"champion_anchor": [true], "count_safe_mode": [true], "dynamic_weighting": [true], "search_budget": [96], "top_k": [6, 8, 10]} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| funding_raised_usd | CatBoost | heavy_tail | ml_tabular | planned_with_evidence | 96 | 0 | 586123.944 | {"bagging_temperature": 1.0, "depth": 8, "l2_leaf_reg": 6.0, "learning_rate": 0.03} | {"bagging_temperature": [0.0, 1.0, 2.0], "depth": [6, 8, 10], "l2_leaf_reg": [3.0, 6.0, 12.0], "learning_rate": [0.01, 0.03, 0.05]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| funding_raised_usd | Chronos | heavy_tail | foundation | planned_with_evidence | 96 | 0 | 374610.314 | {"num_samples": 50, "temperature": 0.9} | {"num_samples": [20, 50, 100], "prediction_length": [1, 7, 14, 30], "temperature": [0.7, 0.9, 1.0]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| funding_raised_usd | LightGBM | heavy_tail | ml_tabular | planned_with_evidence | 96 | 0 | 510180.291 | {"bagging_fraction": 0.9, "feature_fraction": 0.85, "learning_rate": 0.03, "min_child_samples": 40, "num_leaves": 127} | {"bagging_fraction": [0.7, 0.9, 1.0], "feature_fraction": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "min_child_samples": [20, 40, 80], "num_leaves": [63, 127, 255]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task3_risk_adjust/ml_tabular/core_edgar/metrics.json |
| funding_raised_usd | Moirai2 | heavy_tail | foundation | planned_with_evidence | 96 | 0 | 446570.636 | {"num_samples": 50, "temperature": 0.9} | {"num_samples": [20, 50, 100], "prediction_length": [1, 7, 14, 30], "temperature": [0.7, 0.9, 1.0]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/foundation_refs/task1_outcome/foundation/core_only/metrics.json |
| funding_raised_usd | NBEATS | heavy_tail | deep_classical | planned_with_evidence | 96 | 0 | 374514.684 | {"dropout_prob_theta": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout_prob_theta": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737/full/deep_refs/task3_risk_adjust/deep_classical/full/metrics.json |
| funding_raised_usd | NHITS | heavy_tail | deep_classical | planned_with_evidence | 96 | 0 | 374432.357 | {"dropout_prob_theta": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout_prob_theta": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737/full/deep_refs/task1_outcome/deep_classical/core_edgar/metrics.json |
| funding_raised_usd | PatchTST | heavy_tail | transformer_sota | planned_with_evidence | 96 | 0 | 374564.440 | {"dropout": 0.1, "hidden_size": 128, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout": [0.0, 0.1, 0.2], "hidden_size": [64, 128, 256], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| funding_raised_usd | TabPFNRegressor | heavy_tail | unknown | planned | 96 | 0 |  | {"ensemble_size": 16, "subsample": 10000} | {"ensemble_size": [8, 16, 32], "n_estimators": [1], "subsample": [5000, 10000, 20000]} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| funding_raised_usd | XGBoost | heavy_tail | ml_tabular | planned_with_evidence | 96 | 0 | 477212.657 | {"colsample_bytree": 0.85, "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 2.0, "subsample": 0.85} | {"colsample_bytree": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "max_depth": [6, 8, 10], "min_child_weight": [1.0, 2.0, 4.0], "subsample": [0.7, 0.85, 1.0]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | AutoFitV72 | count | unknown | planned | 96 | 0 |  | {"champion_anchor": true, "count_safe_mode": true, "dynamic_weighting": true, "search_budget": 96, "top_k": 8} | {"champion_anchor": [true], "count_safe_mode": [true], "dynamic_weighting": [true], "search_budget": [96], "top_k": [6, 8, 10]} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| investors_count | CatBoost | count | ml_tabular | planned_with_evidence | 96 | 0 | 505.314227 | {"bagging_temperature": 1.0, "depth": 8, "l2_leaf_reg": 6.0, "learning_rate": 0.03} | {"bagging_temperature": [0.0, 1.0, 2.0], "depth": [6, 8, 10], "l2_leaf_reg": [3.0, 6.0, 12.0], "learning_rate": [0.01, 0.03, 0.05]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | Chronos | count | foundation | planned_with_evidence | 96 | 0 | 45.554432 | {"num_samples": 50, "temperature": 0.9} | {"num_samples": [20, 50, 100], "prediction_length": [1, 7, 14, 30], "temperature": [0.7, 0.9, 1.0]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| investors_count | LightGBM | count | ml_tabular | planned_with_evidence | 96 | 0 | 199.171203 | {"bagging_fraction": 0.9, "feature_fraction": 0.85, "learning_rate": 0.03, "min_child_samples": 40, "num_leaves": 127} | {"bagging_fraction": [0.7, 0.9, 1.0], "feature_fraction": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "min_child_samples": [20, 40, 80], "num_leaves": [63, 127, 255]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | LightGBMTweedie | count | ml_tabular | planned_with_evidence | 96 | 0 | 199.171203 | {"learning_rate": 0.03, "min_child_samples": 40, "num_leaves": 127, "tweedie_variance_power": 1.3} | {"learning_rate": [0.01, 0.03, 0.05], "min_child_samples": [20, 40, 80], "num_leaves": [63, 127, 255], "tweedie_variance_power": [1.1, 1.3, 1.5]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | Moirai2 | count | foundation | planned_with_evidence | 96 | 0 | 228.764697 | {"num_samples": 50, "temperature": 0.9} | {"num_samples": [20, 50, 100], "prediction_length": [1, 7, 14, 30], "temperature": [0.7, 0.9, 1.0]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| investors_count | NBEATS | count | deep_classical | planned_with_evidence | 96 | 0 | 44.726689 | {"dropout_prob_theta": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout_prob_theta": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/deep_refs/task1_outcome/deep_classical/core_only/metrics.json |
| investors_count | NHITS | count | deep_classical | planned_with_evidence | 96 | 0 | 44.738015 | {"dropout_prob_theta": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout_prob_theta": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/deep_refs/task1_outcome/deep_classical/core_only/metrics.json |
| investors_count | PatchTST | count | transformer_sota | planned_with_evidence | 96 | 0 | 44.771809 | {"dropout": 0.1, "hidden_size": 128, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout": [0.0, 0.1, 0.2], "hidden_size": [64, 128, 256], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/ts_refs/task1_outcome/transformer_sota/core_only/metrics.json |
| investors_count | XGBoost | count | ml_tabular | planned_with_evidence | 96 | 0 | 175.691838 | {"colsample_bytree": 0.85, "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 2.0, "subsample": 0.85} | {"colsample_bytree": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "max_depth": [6, 8, 10], "min_child_weight": [1.0, 2.0, 4.0], "subsample": [0.7, 0.85, 1.0]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | XGBoostPoisson | count | ml_tabular | planned_with_evidence | 96 | 0 | 175.691838 | {"learning_rate": 0.03, "max_depth": 6, "min_child_weight": 2.0, "subsample": 0.9} | {"learning_rate": [0.01, 0.03, 0.05], "max_depth": [4, 6, 8], "min_child_weight": [1.0, 2.0, 4.0], "subsample": [0.7, 0.9, 1.0]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| investors_count | iTransformer | count | transformer_sota | planned_with_evidence | 96 | 0 | 119.259327 | {"d_model": 128, "dropout": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"d_model": [64, 128, 256], "dropout": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/ts_refs/task2_forecast/transformer_sota/core_edgar/metrics.json |
| is_funded | AutoFitV72 | binary | unknown | planned | 96 | 0 |  | {"champion_anchor": true, "count_safe_mode": true, "dynamic_weighting": true, "search_budget": 96, "top_k": 8} | {"champion_anchor": [true], "count_safe_mode": [true], "dynamic_weighting": [true], "search_budget": [96], "top_k": [6, 8, 10]} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| is_funded | CatBoost | binary | ml_tabular | planned_with_evidence | 96 | 0 | 0.105312 | {"bagging_temperature": 1.0, "depth": 8, "l2_leaf_reg": 6.0, "learning_rate": 0.03} | {"bagging_temperature": [0.0, 1.0, 2.0], "depth": [6, 8, 10], "l2_leaf_reg": [3.0, 6.0, 12.0], "learning_rate": [0.01, 0.03, 0.05]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| is_funded | LightGBM | binary | ml_tabular | planned_with_evidence | 96 | 0 | 0.096787 | {"bagging_fraction": 0.9, "feature_fraction": 0.85, "learning_rate": 0.03, "min_child_samples": 40, "num_leaves": 127} | {"bagging_fraction": [0.7, 0.9, 1.0], "feature_fraction": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "min_child_samples": [20, 40, 80], "num_leaves": [63, 127, 255]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| is_funded | NHITS | binary | deep_classical | planned_with_evidence | 96 | 0 | 0.032248 | {"dropout_prob_theta": 0.1, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout_prob_theta": [0.0, 0.1, 0.2], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737/full/deep_refs/task1_outcome/deep_classical/core_edgar/metrics.json |
| is_funded | PatchTST | binary | transformer_sota | planned_with_evidence | 96 | 0 | 0.032281 | {"dropout": 0.1, "hidden_size": 128, "learning_rate": 0.0003, "max_steps": 2000} | {"dropout": [0.0, 0.1, 0.2], "hidden_size": [64, 128, 256], "learning_rate": [0.0001, 0.0003, 0.001], "max_steps": [1000, 2000, 3000]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| is_funded | TabPFNClassifier | binary | unknown | planned | 96 | 0 |  | {"ensemble_size": 16, "subsample": 10000} | {"ensemble_size": [8, 16, 32], "n_estimators": [1], "subsample": [5000, 10000, 20000]} | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
| is_funded | XGBoost | binary | ml_tabular | planned_with_evidence | 96 | 0 | 0.099060 | {"colsample_bytree": 0.85, "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 2.0, "subsample": 0.85} | {"colsample_bytree": [0.7, 0.85, 1.0], "learning_rate": [0.01, 0.03, 0.05], "max_depth": [6, 8, 10], "min_child_weight": [1.0, 2.0, 4.0], "subsample": [0.7, 0.85, 1.0]} | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
<!-- END AUTO:BEST_CONFIG_BY_MODEL_TARGET -->

## Compute Cost Report

<!-- BEGIN AUTO:COMPUTE_COST_REPORT -->
| model_name | category | target | strict_records | train_time_median_seconds | inference_time_median_seconds | evidence_path |
|---|---|---|---|---|---|---|
| AutoARIMA | statistical | funding_raised_usd | 28 | 553.2150305509567 | 0.7055814266204834 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| AutoARIMA | statistical | investors_count | 23 | 1014.9123437404633 | 0.9476971626281738 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| AutoARIMA | statistical | is_funded | 8 | 301.12455213069916 | 0.5880396366119385 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| AutoETS | statistical | funding_raised_usd | 28 | 33.10526895523071 | 0.17939281463623047 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| AutoETS | statistical | investors_count | 23 | 155.75881958007812 | 0.24746012687683105 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| AutoETS | statistical | is_funded | 8 | 10.434719681739807 | 0.293277382850647 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| AutoFitV1 | autofit | funding_raised_usd | 22 | 1167.4601204395294 | 2.4951658248901367 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV1 | autofit | investors_count | 9 | 6930.470084190369 | 1.040363073348999 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV2 | autofit | funding_raised_usd | 22 | 1356.84569978714 | 10.176039576530457 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV2 | autofit | investors_count | 9 | 7117.237256526947 | 3.628199815750122 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV2E | autofit | funding_raised_usd | 22 | 1359.9466242790222 | 10.25850772857666 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV2E | autofit | investors_count | 9 | 7139.110965013504 | 3.559293508529663 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV3 | autofit | funding_raised_usd | 22 | 1183.6510068178177 | 2.0391933917999268 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV3 | autofit | investors_count | 9 | 7936.809168577194 | 4.585355281829834 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV3E | autofit | funding_raised_usd | 22 | 1374.4111700057983 | 11.282092213630676 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV3E | autofit | investors_count | 9 | 7904.231534957886 | 4.352551221847534 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV3Max | autofit | funding_raised_usd | 12 | 1132.2702051401138 | 1.9186148643493652 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV3Max | autofit | investors_count | 9 | 6760.532226324081 | 1.4591636657714844 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV3Max | autofit | is_funded | 2 | 4614.139898657799 | 0.9064251184463501 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV4 | autofit | funding_raised_usd | 12 | 1289.3750022649765 | 10.487847924232483 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV4 | autofit | investors_count | 9 | 7832.492486715317 | 4.803562164306641 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV4 | autofit | is_funded | 2 | 6109.186114192009 | 4.607777237892151 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV5 | autofit | funding_raised_usd | 12 | 1261.4057006835938 | 1.9522020816802979 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV5 | autofit | investors_count | 9 | 2413.259698867798 | 3.108950138092041 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV5 | autofit | is_funded | 2 | 1831.2754119634628 | 2.7484978437423706 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV6 | autofit | funding_raised_usd | 12 | 1237.2137160301208 | 1.919173240661621 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV6 | autofit | investors_count | 9 | 2204.721166610718 | 0.5368576049804688 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV6 | autofit | is_funded | 2 | 1627.9588223695755 | 0.5752198696136475 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV7 | autofit | funding_raised_usd | 12 | 2588.203194141388 | 2.719128370285034 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV7 | autofit | investors_count | 9 | 3887.965553045273 | 4.688865661621094 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV7 | autofit | is_funded | 2 | 2495.2639400959015 | 1.0005707740783691 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV71 | autofit | funding_raised_usd | 88 | 3997.625604867935 | 3.6089017391204834 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/v71_g02/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV71 | autofit | investors_count | 88 | 6378.813129901886 | 6.551030516624451 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/v71_g02/task1_outcome/autofit/core_edgar/metrics.json |
| AutoFitV71 | autofit | is_funded | 32 | 3061.943458199501 | 1.5362005233764648 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/v71_g02/task1_outcome/autofit/core_edgar/metrics.json |
| AutoTheta | statistical | funding_raised_usd | 28 | 65.28136610984802 | 0.15905547142028809 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| AutoTheta | statistical | investors_count | 23 | 197.52597570419312 | 0.2270655632019043 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| AutoTheta | statistical | is_funded | 8 | 60.799091815948486 | 0.2421102523803711 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| Autoformer | transformer_sota | funding_raised_usd | 12 | 281.35190057754517 | 2.939230442047119 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| Autoformer | transformer_sota | investors_count | 12 | 185.52908074855804 | 4.291038632392883 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| Autoformer | transformer_sota | is_funded | 4 | 387.4042491912842 | 5.334161877632141 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| BiTCN | transformer_sota | funding_raised_usd | 12 | 22.87924897670746 | 2.0530790090560913 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| BiTCN | transformer_sota | investors_count | 12 | 32.23628628253937 | 2.565114378929138 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| BiTCN | transformer_sota | is_funded | 4 | 37.605379939079285 | 3.1485458612442017 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| CatBoost | ml_tabular | funding_raised_usd | 6 | 29.86212718486786 | 0.06857562065124512 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| CatBoost | ml_tabular | investors_count | 6 | 4.128932356834412 | 0.012008070945739746 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| CatBoost | ml_tabular | is_funded | 2 | 98.84714674949646 | 0.06163918972015381 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| Chronos | foundation | funding_raised_usd | 12 | 3.4632540941238403 | 17.21852731704712 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| Chronos | foundation | investors_count | 12 | 12.057347416877747 | 37.35305190086365 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| Chronos | foundation | is_funded | 4 | 9.904327988624573 | 43.341087222099304 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| ChronosBolt | foundation | funding_raised_usd | 12 | 3.3239378929138184 | 3.075244903564453 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| ChronosBolt | foundation | investors_count | 12 | 11.864245176315308 | 4.6291728019714355 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| ChronosBolt | foundation | is_funded | 4 | 9.737047553062439 | 5.924058794975281 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| DLinear | transformer_sota | funding_raised_usd | 12 | 11.099937796592712 | 2.0113441944122314 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| DLinear | transformer_sota | investors_count | 12 | 26.503034234046936 | 2.4864670038223267 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| DLinear | transformer_sota | is_funded | 4 | 26.91865837574005 | 3.0300559997558594 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| DeepAR | deep_classical | funding_raised_usd | 96 | 35.679803013801575 | 2.9062693119049072 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/deep_classical/full/metrics.json |
| DeepAR | deep_classical | investors_count | 96 | 69.16873621940613 | 4.432110905647278 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/deep_classical/full/metrics.json |
| DeepAR | deep_classical | is_funded | 32 | 80.01411974430084 | 5.381286382675171 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/deep_refs/task1_outcome/deep_classical/core_edgar/metrics.json |
| ElasticNet | ml_tabular | funding_raised_usd | 6 | 148.14075005054474 | 0.18293893337249756 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| ElasticNet | ml_tabular | investors_count | 6 | 937.2939561605453 | 0.11700010299682617 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| ExtraTrees | ml_tabular | funding_raised_usd | 6 | 91.74917685985565 | 1.9758028984069824 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| ExtraTrees | ml_tabular | investors_count | 6 | 956.419330239296 | 1.0630468130111694 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| ExtraTrees | ml_tabular | is_funded | 2 | 774.0270978212357 | 0.9254631996154785 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| FEDformer | transformer_sota | funding_raised_usd | 12 | 413.4934651851654 | 3.029645562171936 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| FEDformer | transformer_sota | investors_count | 12 | 197.27097427845 | 4.4105130434036255 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| FEDformer | transformer_sota | is_funded | 4 | 401.79165840148926 | 5.4206424951553345 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| HistGradientBoosting | ml_tabular | funding_raised_usd | 6 | 55.968268275260925 | 1.0012376308441162 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| HistGradientBoosting | ml_tabular | investors_count | 6 | 71.88907265663147 | 0.7212960720062256 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| HistGradientBoosting | ml_tabular | is_funded | 2 | 131.97428596019745 | 0.7268706560134888 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| Informer | transformer_sota | funding_raised_usd | 100 | 248.69995021820068 | 2.5738736391067505 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| Informer | transformer_sota | investors_count | 100 | 277.44143772125244 | 3.5693697929382324 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| Informer | transformer_sota | is_funded | 36 | 293.40517950057983 | 4.45264732837677 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| KAN | transformer_sota | funding_raised_usd | 12 | 18.805071711540222 | 2.15146541595459 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| KAN | transformer_sota | investors_count | 12 | 32.83353400230408 | 2.731676697731018 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| KAN | transformer_sota | is_funded | 4 | 37.822476267814636 | 3.3693445920944214 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| KNN | ml_tabular | funding_raised_usd | 6 | 0.12725472450256348 | 17.56334638595581 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| KNN | ml_tabular | investors_count | 6 | 0.18726003170013428 | 10.160885453224182 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| KNN | ml_tabular | is_funded | 2 | 0.20918941497802734 | 10.123677372932434 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| LagLlama | foundation | funding_raised_usd | 12 | 2.8139020204544067 | 1.025435447692871 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| LagLlama | foundation | investors_count | 12 | 11.33693814277649 | 0.573447585105896 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| LagLlama | foundation | is_funded | 3 | 9.252422094345093 | 0.5808987617492676 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| Lasso | ml_tabular | funding_raised_usd | 6 | 133.05757677555084 | 0.1844921112060547 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| Lasso | ml_tabular | investors_count | 6 | 853.938165307045 | 0.13397037982940674 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| LightGBM | ml_tabular | funding_raised_usd | 6 | 17.580730319023132 | 4.563787221908569 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| LightGBM | ml_tabular | investors_count | 6 | 29.652069687843323 | 1.1707526445388794 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| LightGBM | ml_tabular | is_funded | 2 | 20.16628396511078 | 1.1526070833206177 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| LightGBMTweedie | ml_tabular | funding_raised_usd | 6 | 24.520169258117676 | 5.560119032859802 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| LightGBMTweedie | ml_tabular | investors_count | 6 | 29.932886004447937 | 1.1422538757324219 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| LightGBMTweedie | ml_tabular | is_funded | 2 | 11.709624648094177 | 0.3916134834289551 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| LogisticRegression | ml_tabular | is_funded | 2 | 139.96766197681427 | 0.13291478157043457 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| MOMENT | foundation | funding_raised_usd | 12 | 6.425565600395203 | 1.1947851181030273 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| MOMENT | foundation | investors_count | 12 | 14.903549671173096 | 0.8789814710617065 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| MOMENT | foundation | is_funded | 3 | 12.609259605407715 | 0.9343504905700684 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| MSTL | statistical | funding_raised_usd | 28 | 119.60196721553802 | 1.5446263551712036 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| MSTL | statistical | investors_count | 23 | 1967.6096425056458 | 3.844602108001709 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| MSTL | statistical | is_funded | 8 | 1967.0548470020294 | 4.056684255599976 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| MeanPredictor | ml_tabular | funding_raised_usd | 6 | 0.0009340047836303711 | 0.00046896934509277344 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| MeanPredictor | ml_tabular | investors_count | 6 | 0.004581332206726074 | 0.00025451183319091797 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| MeanPredictor | ml_tabular | is_funded | 2 | 0.003905773162841797 | 0.00025725364685058594 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| Moirai | foundation | funding_raised_usd | 12 | 3.1177494525909424 | 1.0221457481384277 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| Moirai | foundation | investors_count | 12 | 11.619640946388245 | 0.5777878761291504 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| Moirai | foundation | is_funded | 4 | 9.494076371192932 | 0.545499324798584 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| Moirai2 | foundation | funding_raised_usd | 68 | 3.4939440488815308 | 1.0472036600112915 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| Moirai2 | foundation | investors_count | 68 | 11.405064463615417 | 0.5570920705795288 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| Moirai2 | foundation | is_funded | 32 | 8.999884724617004 | 0.5308980941772461 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| MoiraiLarge | foundation | funding_raised_usd | 12 | 4.716952085494995 | 1.068938970565796 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| MoiraiLarge | foundation | investors_count | 12 | 13.259610414505005 | 0.5932414531707764 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| MoiraiLarge | foundation | is_funded | 4 | 11.113070964813232 | 0.5750147104263306 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| NBEATS | deep_classical | funding_raised_usd | 96 | 14.398378252983093 | 2.0238821506500244 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/deep_classical/full/metrics.json |
| NBEATS | deep_classical | investors_count | 96 | 28.637384057044983 | 2.480656147003174 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/deep_classical/full/metrics.json |
| NBEATS | deep_classical | is_funded | 32 | 28.51807737350464 | 3.0372174978256226 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/deep_refs/task1_outcome/deep_classical/core_edgar/metrics.json |
| NBEATSx | transformer_sota | funding_raised_usd | 12 | 14.503607630729675 | 2.0816025733947754 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| NBEATSx | transformer_sota | investors_count | 12 | 30.3129825592041 | 2.5948516130447388 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| NBEATSx | transformer_sota | is_funded | 4 | 30.570395469665527 | 3.136952757835388 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| NHITS | deep_classical | funding_raised_usd | 96 | 16.428391456604004 | 2.0386730432510376 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/deep_classical/full/metrics.json |
| NHITS | deep_classical | investors_count | 96 | 30.52108633518219 | 2.487233281135559 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/deep_classical/full/metrics.json |
| NHITS | deep_classical | is_funded | 32 | 30.76546049118042 | 3.0541274547576904 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/deep_refs/task1_outcome/deep_classical/core_edgar/metrics.json |
| NLinear | transformer_sota | funding_raised_usd | 12 | 12.769177317619324 | 2.011831760406494 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| NLinear | transformer_sota | investors_count | 12 | 26.092832565307617 | 2.461946725845337 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| NLinear | transformer_sota | is_funded | 4 | 26.32720708847046 | 3.0011101961135864 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| PatchTST | transformer_sota | funding_raised_usd | 100 | 57.551882147789 | 2.2391997575759888 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| PatchTST | transformer_sota | investors_count | 100 | 79.22305715084076 | 2.9372174739837646 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| PatchTST | transformer_sota | is_funded | 36 | 87.1597912311554 | 3.6333025693893433 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| QuantileRegressor | ml_tabular | investors_count | 6 | 115.21589350700378 | 0.10009253025054932 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| RMoK | transformer_sota | funding_raised_usd | 12 | 232.63541448116302 | 1.9749020338058472 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| RMoK | transformer_sota | investors_count | 12 | 283.75363647937775 | 1.5650311708450317 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| RMoK | transformer_sota | is_funded | 4 | 233.6176337003708 | 1.5262548923492432 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| RandomForest | ml_tabular | funding_raised_usd | 6 | 188.95346128940582 | 2.0059149265289307 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| RandomForest | ml_tabular | investors_count | 6 | 649.4170596599579 | 0.8713734149932861 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| RandomForest | ml_tabular | is_funded | 2 | 470.0100508928299 | 0.6653556823730469 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| Ridge | ml_tabular | funding_raised_usd | 6 | 0.6517119407653809 | 0.1943683624267578 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| Ridge | ml_tabular | investors_count | 6 | 22.414336919784546 | 0.09550583362579346 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| SF_SeasonalNaive | statistical | funding_raised_usd | 28 | 0.4417538642883301 | 0.14089655876159668 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| SF_SeasonalNaive | statistical | investors_count | 23 | 2.0345520973205566 | 0.18816280364990234 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| SF_SeasonalNaive | statistical | is_funded | 8 | 1.8112632036209106 | 0.20863783359527588 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/statistical/core_edgar/metrics.json |
| SOFTS | transformer_sota | funding_raised_usd | 12 | 395.5474020242691 | 1.9834668636322021 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| SOFTS | transformer_sota | investors_count | 12 | 504.36146318912506 | 1.5781733989715576 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| SOFTS | transformer_sota | is_funded | 4 | 474.6627700328827 | 1.5615735054016113 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| SVR | ml_tabular | funding_raised_usd | 6 | 7.523138880729675 | 423.6171886920929 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| SVR | ml_tabular | investors_count | 6 | 7.102011799812317 | 249.8881858587265 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| SeasonalNaive | ml_tabular | funding_raised_usd | 6 | 0.0010380744934082031 | 0.0008004903793334961 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| SeasonalNaive | ml_tabular | investors_count | 6 | 0.004719138145446777 | 0.0004330873489379883 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| SeasonalNaive | ml_tabular | is_funded | 2 | 0.004002571105957031 | 0.00038313865661621094 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| StemGNN | transformer_sota | funding_raised_usd | 12 | 5.374015808105469 | 0.0010941028594970703 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| StemGNN | transformer_sota | investors_count | 12 | 13.583812475204468 | 0.0028036832809448242 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| StemGNN | transformer_sota | is_funded | 4 | 11.185238599777222 | 0.0028715133666992188 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TFT | deep_classical | funding_raised_usd | 96 | 77.43468463420868 | 2.275864005088806 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/deep_classical/full/metrics.json |
| TFT | deep_classical | investors_count | 96 | 97.96240270137787 | 2.9943827390670776 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task2_forecast/deep_classical/full/metrics.json |
| TFT | deep_classical | is_funded | 32 | 104.18697917461395 | 3.6934967041015625 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205/pilot/deep_refs/task1_outcome/deep_classical/core_edgar/metrics.json |
| TSMixer | transformer_sota | funding_raised_usd | 100 | 127.66831338405609 | 1.9642997980117798 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TSMixer | transformer_sota | investors_count | 100 | 173.63575434684753 | 1.5635442733764648 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TSMixer | transformer_sota | is_funded | 36 | 168.23214757442474 | 1.5184396505355835 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TSMixerx | transformer_sota | funding_raised_usd | 12 | 130.30370354652405 | 2.2613412141799927 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TSMixerx | transformer_sota | investors_count | 12 | 141.76150035858154 | 1.5543566942214966 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TSMixerx | transformer_sota | is_funded | 4 | 169.66550731658936 | 1.529691219329834 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TiDE | transformer_sota | funding_raised_usd | 12 | 23.948777556419373 | 2.1617116928100586 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TiDE | transformer_sota | investors_count | 12 | 43.479450941085815 | 2.7324429750442505 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TiDE | transformer_sota | is_funded | 4 | 46.461273312568665 | 3.314643621444702 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TimeMixer | transformer_sota | funding_raised_usd | 12 | 146.07332038879395 | 2.0895837545394897 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TimeMixer | transformer_sota | investors_count | 12 | 193.72503113746643 | 1.566767692565918 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TimeMixer | transformer_sota | is_funded | 4 | 190.29247760772705 | 1.52839195728302 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TimeMoE | foundation | funding_raised_usd | 12 | 3.653159260749817 | 5.047734022140503 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| TimeMoE | foundation | investors_count | 12 | 12.104049324989319 | 8.668771982192993 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| TimeMoE | foundation | is_funded | 3 | 10.054765701293945 | 11.132622957229614 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| TimeXer | transformer_sota | funding_raised_usd | 12 | 9.709867596626282 | 0.0010259151458740234 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TimeXer | transformer_sota | investors_count | 12 | 17.51503849029541 | 0.002809762954711914 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TimeXer | transformer_sota | is_funded | 4 | 14.295098900794983 | 0.0024584531784057617 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| Timer | foundation | funding_raised_usd | 12 | 3.612670421600342 | 3.449040651321411 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| Timer | foundation | investors_count | 12 | 12.073709964752197 | 6.501463055610657 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| Timer | foundation | is_funded | 3 | 10.0346519947052 | 6.777311563491821 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| TimesFM | foundation | funding_raised_usd | 68 | 2.4957350492477417 | 1.0131382942199707 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| TimesFM | foundation | investors_count | 68 | 10.465166211128235 | 0.5388302803039551 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| TimesFM | foundation | is_funded | 31 | 7.617969036102295 | 0.5248475074768066 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/foundation/full/metrics.json |
| TimesNet | transformer_sota | funding_raised_usd | 100 | 165.77568817138672 | 4.911632537841797 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TimesNet | transformer_sota | investors_count | 100 | 290.47784900665283 | 8.69097888469696 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| TimesNet | transformer_sota | is_funded | 36 | 347.1296001672745 | 11.358771562576294 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| VanillaTransformer | transformer_sota | funding_raised_usd | 12 | 52.46015751361847 | 2.296930193901062 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| VanillaTransformer | transformer_sota | investors_count | 12 | 66.60667157173157 | 2.9729864597320557 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| VanillaTransformer | transformer_sota | is_funded | 4 | 163.58066022396088 | 3.671358823776245 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| XGBoost | ml_tabular | funding_raised_usd | 6 | 28.950583815574646 | 0.8267769813537598 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| XGBoost | ml_tabular | investors_count | 6 | 138.0520099401474 | 0.4946625232696533 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| XGBoost | ml_tabular | is_funded | 2 | 67.57751870155334 | 0.4930543899536133 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| XGBoostPoisson | ml_tabular | funding_raised_usd | 6 | 34.17888867855072 | 0.8310798406600952 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| XGBoostPoisson | ml_tabular | investors_count | 6 | 155.06927728652954 | 0.49250471591949463 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| XGBoostPoisson | ml_tabular | is_funded | 2 | 93.13045859336853 | 0.47519755363464355 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/ml_tabular/core_edgar/metrics.json |
| iTransformer | transformer_sota | funding_raised_usd | 100 | 1572.957323551178 | 1.9896613359451294 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| iTransformer | transformer_sota | investors_count | 100 | 1489.628781914711 | 1.5695315599441528 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
| iTransformer | transformer_sota | is_funded | 36 | 1489.6344392299652 | 1.531008243560791 | /mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7/task1_outcome/transformer_sota/full/metrics.json |
<!-- END AUTO:COMPUTE_COST_REPORT -->

## V7.2 Pilot Gate Report

<!-- BEGIN AUTO:V72_PILOT_GATE_REPORT -->
| section | key | value | evidence_path |
|---|---|---|---|
| summary | generated_at_utc | 2026-02-22T00:56:16.742599+00:00 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| summary | overall_pass | false | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| counts | overlap_keys_v7_v72_non_autofit | 24 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| counts | rows_strict | 5109 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| counts | rows_total | 14491 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| counts | v72_rows_raw | 24 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| metrics | global_normalized_mae_improvement_pct | 1.490925 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| metrics | global_normalized_mae_v7 | 1.102229 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| metrics | global_normalized_mae_v72 | 1.085796 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| metrics | investors_count_gap_reduction_pct | -2.917353 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| metrics | investors_count_median_gap_v7 | 4.976659 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| metrics | investors_count_median_gap_v72 | 5.121846 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| metrics | v72_vs_v7_win_rate_pct | 50.000000 | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| checks | fairness_pass_100 | true | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| checks | global_normalized_mae_improvement_ge_8pct | false | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
| checks | investors_count_gap_reduction_ge_50pct | false | docs/benchmarks/block3_truth_pack/v72_pilot_gate_report.json |
<!-- END AUTO:V72_PILOT_GATE_REPORT -->

## Historical Full-Scale Experiment Ledger

<!-- BEGIN AUTO:HISTORICAL_FULL_SCALE_EXPERIMENT_LEDGER -->
### Run Ledger

| run_name | run_stage | raw_records | strict_records | legacy_records | strict_ratio | models | categories | condition_coverage_strict | condition_coverage_legacy | best_model_by_target_json | key_failures | evidence_path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block3_20260203_225620_iris | iris_initial | 631 | 0 | 631 | 0.000000 | 28 | 5 | 0.000000 | 0.403846 | {} | {} | runs/benchmarks/block3_20260203_225620_iris |
| block3_20260203_225620_iris_phase3 | iris_phase3 | 1750 | 0 | 1750 | 0.000000 | 49 | 7 | 0.000000 | 0.538462 | {} | {} | runs/benchmarks/block3_20260203_225620_iris_phase3 |
| block3_20260203_225620_iris_full | iris_full_baseline | 2646 | 0 | 2646 | 0.000000 | 49 | 7 | 0.000000 | 0.538462 | {} | {} | runs/benchmarks/block3_20260203_225620_iris_full |
| block3_20260203_225620_iris_phase7 | iris_phase7_partial | 1060 | 0 | 1060 | 0.000000 | 56 | 6 | 0.000000 | 0.230769 | {} | {} | runs/benchmarks/block3_20260203_225620_iris_phase7 |
| block3_20260203_225620_phase7 | phase7_canonical | 5159 | 1864 | 3295 | 0.361310 | 68 | 7 | 0.615385 | 1.000000 | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.35741397936, "model": "NHITS"}, "investors_count": {"category": "transformer_sota", "mae": 44.79163160827051, "model": "NBEATSx"}, "is_funded": {"category": "transformer_sota", "mae": 0.032280800947848576, "model": "PatchTST"}} | {"autofit_gap_gt_100pct": 16} | runs/benchmarks/block3_20260203_225620_phase7 |
| block3_20260203_225620_phase7_v71extreme_20260214_032205 | v71_pilot | 1643 | 1643 | 0 | 1.000000 | 18 | 5 | 1.000000 | 0.000000 | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.35741397936, "model": "NHITS"}, "investors_count": {"category": "deep_classical", "mae": 44.726689230844215, "model": "NBEATS"}, "is_funded": {"category": "deep_classical", "mae": 0.03232223233914305, "model": "NHITS"}} | {"autofit_gap_gt_100pct": 28, "v71_count_explosion": 8} | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205 |
| block3_20260203_225620_phase7_v71extreme_20260214_130737 | v71_full | 1112 | 1112 | 0 | 1.000000 | 12 | 4 | 1.000000 | 0.000000 | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.3574093943, "model": "NHITS"}, "investors_count": {"category": "deep_classical", "mae": 44.726689230844215, "model": "NBEATS"}, "is_funded": {"category": "deep_classical", "mae": 0.03224843655460842, "model": "NHITS"}} | {} | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737 |
| block3_20260203_225620_dual3090_phase7_20260215_172651 | other | 0 | 0 | 0 | 0.000000 | 0 | 0 | 0.000000 | 0.000000 | {} | {} | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_172651 |
| block3_20260203_225620_dual3090_phase7_20260215_173453 | other | 0 | 0 | 0 | 0.000000 | 0 | 0 | 0.000000 | 0.000000 | {} | {} | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_173453 |
| block3_20260203_225620_dual3090_phase7_20260215_182303 | other | 0 | 0 | 0 | 0.000000 | 0 | 0 | 0.000000 | 0.000000 | {} | {} | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_182303 |
| block3_20260203_225620_dual3090_phase7_20260215_185618 | other | 0 | 0 | 0 | 0.000000 | 0 | 0 | 0.000000 | 0.000000 | {} | {} | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_185618 |
| block3_20260203_225620_dual3090_phase7_20260215_191536 | other | 580 | 580 | 0 | 1.000000 | 45 | 5 | 0.230769 | 0.000000 | {"funding_raised_usd": {"category": "deep_classical", "mae": 380745.9910913987, "model": "NBEATS"}, "investors_count": {"category": "transformer_sota", "mae": 44.69275549359792, "model": "KAN"}, "is_funded": {"category": "deep_classical", "mae": 0.03295886751735992, "model": "NHITS"}} | {"autofit_gap_gt_100pct": 16} | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_191536 |
| block3_20260203_225620_phase7_v72_4090_20260219_173137 | other | 29 | 29 | 0 | 1.000000 | 3 | 1 | 0.192308 | 0.000000 | {"funding_raised_usd": {"category": "autofit", "mae": 396381.2919287242, "model": "AutoFitV71"}, "investors_count": {"category": "autofit", "mae": 257.7900048566551, "model": "AutoFitV72"}, "is_funded": {"category": "autofit", "mae": 0.09253663529720278, "model": "AutoFitV72"}} | {} | runs/benchmarks/block3_20260203_225620_phase7_v72_4090_20260219_173137 |
| block3_20260203_225620_phase7_v72_failure_pool_rerun_heavy_20260219_v72fix_sched | other | 8 | 8 | 0 | 1.000000 | 2 | 1 | 0.038462 | 0.000000 | {"investors_count": {"category": "autofit", "mae": 274.4063475010302, "model": "AutoFitV72"}} | {} | runs/benchmarks/block3_20260203_225620_phase7_v72_failure_pool_rerun_heavy_20260219_v72fix_sched |

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
| block3_20260203_225620_phase7 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.6154 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_phase7 | strict_ratio | Share of strict-comparable records among all valid records. | 0.3613 | runs/benchmarks/block3_20260203_225620_phase7 |
| block3_20260203_225620_phase7 | target_winners | Best strict model per target for this run. | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.35741397936, "model": "NHITS"}, "investors_count": {"category": "transformer_sota", "mae": 44.79163160827051, "model": "NBEATSx"}, "is_funded": {"category": "transformer_sota", "mae": 0.032280800947848576, "model": "PatchTST"}} | runs/benchmarks/block3_20260203_225620_phase7 |
| block3_20260203_225620_phase7_v71extreme_20260214_032205 | failure_tags | Failure taxonomy tags found in this run. | {"autofit_gap_gt_100pct": 28, "v71_count_explosion": 8} | docs/benchmarks/block3_truth_pack/failure_taxonomy.csv |
| block3_20260203_225620_phase7_v71extreme_20260214_032205 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 1.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_phase7_v71extreme_20260214_032205 | strict_ratio | Share of strict-comparable records among all valid records. | 1.0000 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205 |
| block3_20260203_225620_phase7_v71extreme_20260214_032205 | target_winners | Best strict model per target for this run. | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.35741397936, "model": "NHITS"}, "investors_count": {"category": "deep_classical", "mae": 44.726689230844215, "model": "NBEATS"}, "is_funded": {"category": "deep_classical", "mae": 0.03232223233914305, "model": "NHITS"}} | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_032205 |
| block3_20260203_225620_phase7_v71extreme_20260214_130737 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 1.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_phase7_v71extreme_20260214_130737 | strict_ratio | Share of strict-comparable records among all valid records. | 1.0000 | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737 |
| block3_20260203_225620_phase7_v71extreme_20260214_130737 | target_winners | Best strict model per target for this run. | {"funding_raised_usd": {"category": "deep_classical", "mae": 374432.3574093943, "model": "NHITS"}, "investors_count": {"category": "deep_classical", "mae": 44.726689230844215, "model": "NBEATS"}, "is_funded": {"category": "deep_classical", "mae": 0.03224843655460842, "model": "NHITS"}} | runs/benchmarks/block3_20260203_225620_phase7_v71extreme_20260214_130737 |
| block3_20260203_225620_dual3090_phase7_20260215_172651 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_dual3090_phase7_20260215_172651 | strict_ratio | Share of strict-comparable records among all valid records. | 0.0000 | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_172651 |
| block3_20260203_225620_dual3090_phase7_20260215_173453 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_dual3090_phase7_20260215_173453 | strict_ratio | Share of strict-comparable records among all valid records. | 0.0000 | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_173453 |
| block3_20260203_225620_dual3090_phase7_20260215_182303 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_dual3090_phase7_20260215_182303 | strict_ratio | Share of strict-comparable records among all valid records. | 0.0000 | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_182303 |
| block3_20260203_225620_dual3090_phase7_20260215_185618 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.0000 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_dual3090_phase7_20260215_185618 | strict_ratio | Share of strict-comparable records among all valid records. | 0.0000 | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_185618 |
| block3_20260203_225620_dual3090_phase7_20260215_191536 | failure_tags | Failure taxonomy tags found in this run. | {"autofit_gap_gt_100pct": 16} | docs/benchmarks/block3_truth_pack/failure_taxonomy.csv |
| block3_20260203_225620_dual3090_phase7_20260215_191536 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.2308 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_dual3090_phase7_20260215_191536 | strict_ratio | Share of strict-comparable records among all valid records. | 1.0000 | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_191536 |
| block3_20260203_225620_dual3090_phase7_20260215_191536 | target_winners | Best strict model per target for this run. | {"funding_raised_usd": {"category": "deep_classical", "mae": 380745.9910913987, "model": "NBEATS"}, "investors_count": {"category": "transformer_sota", "mae": 44.69275549359792, "model": "KAN"}, "is_funded": {"category": "deep_classical", "mae": 0.03295886751735992, "model": "NHITS"}} | runs/benchmarks/block3_20260203_225620_dual3090_phase7_20260215_191536 |
| block3_20260203_225620_phase7_v72_4090_20260219_173137 | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.1923 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_phase7_v72_4090_20260219_173137 | strict_ratio | Share of strict-comparable records among all valid records. | 1.0000 | runs/benchmarks/block3_20260203_225620_phase7_v72_4090_20260219_173137 |
| block3_20260203_225620_phase7_v72_4090_20260219_173137 | target_winners | Best strict model per target for this run. | {"funding_raised_usd": {"category": "autofit", "mae": 396381.2919287242, "model": "AutoFitV71"}, "investors_count": {"category": "autofit", "mae": 257.7900048566551, "model": "AutoFitV72"}, "is_funded": {"category": "autofit", "mae": 0.09253663529720278, "model": "AutoFitV72"}} | runs/benchmarks/block3_20260203_225620_phase7_v72_4090_20260219_173137 |
| block3_20260203_225620_phase7_v72_failure_pool_rerun_heavy_20260219_v72fix_sched | strict_condition_coverage | Coverage ratio of expected 104 keys under strict comparability. | 0.0385 | docs/benchmarks/block3_truth_pack/condition_inventory_full.csv |
| block3_20260203_225620_phase7_v72_failure_pool_rerun_heavy_20260219_v72fix_sched | strict_ratio | Share of strict-comparable records among all valid records. | 1.0000 | runs/benchmarks/block3_20260203_225620_phase7_v72_failure_pool_rerun_heavy_20260219_v72fix_sched |
| block3_20260203_225620_phase7_v72_failure_pool_rerun_heavy_20260219_v72fix_sched | target_winners | Best strict model per target for this run. | {"investors_count": {"category": "autofit", "mae": 274.4063475010302, "model": "AutoFitV72"}} | runs/benchmarks/block3_20260203_225620_phase7_v72_failure_pool_rerun_heavy_20260219_v72fix_sched |
<!-- END AUTO:HISTORICAL_FULL_SCALE_EXPERIMENT_LEDGER -->

## AutoFit Version Ladder (V1->V7.2)

<!-- BEGIN AUTO:AUTOFIT_VERSION_LADDER -->
### Version Ladder

| version | commit_hint | core_changes | inspiration_source | measured_targets | median_mae_by_target_json | median_gap_vs_best_non_autofit_json | primary_failure_mode | evidence_path |
|---|---|---|---|---|---|---|---|---|
| AutoFitV1 | baseline_pre_phase2 | Data-driven best-single selection with residual correction. | Pragmatic stacked residual correction from tabular ensemble practice. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 416746.80273314926, "investors_count": 125.4686315584109, "is_funded": 0.09864059969980168} | {"funding_raised_usd": 11.13926909363364, "investors_count": 180.07029688985838, "is_funded": 205.44912458859642} | high_gap_vs_best_non_autofit | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV2 | 87baa13 | Top-K weighted ensemble by inverse validation MAE. | Classical weighted model averaging. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 429035.75390673813, "investors_count": 223.18594436160748, "is_funded": 0.08939422251350707} | {"funding_raised_usd": 14.544737825059162, "investors_count": 398.07183212243416, "is_funded": 176.81692014372283} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV2E | 87baa13 | Top-K stacking with LightGBM meta-learner. | Stacked generalization. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 429035.56678631355, "investors_count": 223.17549131231723, "is_funded": 0.10312868071692563} | {"funding_raised_usd": 14.54296227693379, "investors_count": 398.0295202146263, "is_funded": 219.3468545490314} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV3 | 320c314 | Temporal CV + greedy/exhaustive subset selection with diversity preference. | Temporal OOF stacking with subset search. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 395955.94231643423, "investors_count": 166.5901593917879, "is_funded": 0.10890193145407928} | {"funding_raised_usd": 5.7122780068785906, "investors_count": 271.8469491563099, "is_funded": 237.22422339168654} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV3E | 320c314 | Top-K stacking variant under V3 temporal CV framework. | OOF-based stacking simplification. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 430407.32656598045, "investors_count": 204.14996378872286, "is_funded": 0.1058452036562632} | {"funding_raised_usd": 14.908285029875124, "investors_count": 355.7022760046878, "is_funded": 227.7588021271154} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV3Max | 320c314 | Exhaustive V3 search with bounded candidate budget. | Combinatorial ensemble search. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 396360.34859371965, "investors_count": 137.25314473071458, "is_funded": 0.10890193145407928} | {"funding_raised_usd": 5.784237218695543, "investors_count": 206.3042367132195, "is_funded": 237.22422339168654} | high_gap_vs_best_non_autofit | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV4 | c53abf6 | Target transforms + full OOF stacking + diversity-aware selection. | NCL and transform-aware robust stacking. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 428265.89420248836, "investors_count": 196.73858138376124, "is_funded": 0.09019217803206035} | {"funding_raised_usd": 14.299559151635744, "investors_count": 339.1945823307518, "is_funded": 179.157310539888} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV5 | dce0ff9 | Regime-aware tiered evaluation and collapse detection guard. | Cost-aware routing with monotonic fallback guard. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 396360.34859371965, "investors_count": 252.99398560055954, "is_funded": 0.09947219709664773} | {"funding_raised_usd": 5.784237218695543, "investors_count": 464.57128211048683, "is_funded": 208.02423765207965} | no_critical_failure_observed | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV6 | ad07032 | Caruana-style greedy ensembling + two-layer stacking + conformal weighting. | AutoGluon-style weighted ensemble with robust transform. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 396360.34859371965, "investors_count": 300.71177668475053, "is_funded": 0.09818964911714113} | {"funding_raised_usd": 5.680372003291545, "investors_count": 571.0564004342739, "is_funded": 203.91059367313343} | high_gap_vs_best_non_autofit | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV7 | faafdcf | Data-adapted robust ensemble with missingness/ratio features and repeated temporal CV. | SOTA tabular feature engineering + robust ensemble selection. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 397006.3727232514, "investors_count": 267.726448471727, "is_funded": 0.09068313538611147} | {"funding_raised_usd": 5.956653985432814, "investors_count": 497.61730862455914, "is_funded": 181.20164905530132} | high_gap_vs_best_non_autofit | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV71 | phase7_v71_extreme_branch | Lane-adaptive routing, dynamic thresholds, count-safe mode, anchor and policy logging. | Target-family specialization with fairness-first guards. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 396418.30570865615, "investors_count": 299.6087412433037, "is_funded": 0.08941793631829838} | {"funding_raised_usd": 5.68595606194453, "investors_count": 570.3742875872167, "is_funded": 173.9675552972115} | v71_count_explosion | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |
| AutoFitV72 | planned_not_materialized | Count-safe hardening + champion-anchor + offline policy layer (planned). | Evidence-driven v7.2 design gates. | funding_raised_usd,investors_count,is_funded | {"funding_raised_usd": 398693.79909221013, "investors_count": 274.40665472125744, "is_funded": 0.09253663529720282} | {"funding_raised_usd": 5.4683181461985475, "investors_count": 512.1846192283012, "is_funded": 180.69282978374306} | planned_not_materialized | docs/benchmarks/block3_truth_pack/autofit_lineage.csv |

### Step Deltas

| from_version | to_version | target | overlap_keys | median_mae_delta_pct | median_gap_delta_pct | evidence_path |
|---|---|---|---|---|---|---|
| AutoFitV1 | AutoFitV2 | funding_raised_usd | 24 | 3.201127 | 3.283085 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV1 | AutoFitV2 | investors_count | 17 | 77.881867 | 218.062056 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV1 | AutoFitV2 | is_funded | 3 | -9.373805 | -28.632204 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2 | AutoFitV2E | funding_raised_usd | 24 | -0.000000 | -0.000000 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2 | AutoFitV2E | investors_count | 17 | 0.000000 | 0.000000 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2 | AutoFitV2E | is_funded | 3 | 15.363921 | 42.510054 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2E | AutoFitV3 | funding_raised_usd | 24 | -7.710224 | -8.830684 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2E | AutoFitV3 | investors_count | 17 | -25.342449 | -126.053120 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV2E | AutoFitV3 | is_funded | 3 | 5.598104 | 17.877369 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3 | AutoFitV3E | funding_raised_usd | 24 | 8.700787 | 9.196811 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3 | AutoFitV3E | investors_count | 17 | 22.541211 | 83.826466 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3 | AutoFitV3E | is_funded | 3 | -2.806863 | -9.465421 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3E | AutoFitV3Max | funding_raised_usd | 20 | -8.025776 | -9.231988 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3E | AutoFitV3Max | investors_count | 13 | -32.787017 | -149.415682 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3E | AutoFitV3Max | is_funded | 1 | 2.887923 | 9.465421 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3Max | AutoFitV4 | funding_raised_usd | 20 | 8.049631 | 8.516542 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3Max | AutoFitV4 | investors_count | 20 | 43.330811 | 132.774319 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV3Max | AutoFitV4 | is_funded | 3 | -17.180369 | -57.935938 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV4 | AutoFitV5 | funding_raised_usd | 21 | -7.449938 | -8.515322 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV4 | AutoFitV5 | investors_count | 20 | 28.593987 | 125.583328 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV4 | AutoFitV5 | is_funded | 3 | 10.289162 | 28.735952 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV5 | AutoFitV6 | funding_raised_usd | 21 | 0.000000 | 0.000000 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV5 | AutoFitV6 | investors_count | 20 | 18.861235 | 106.515356 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV5 | AutoFitV6 | is_funded | 3 | -1.289353 | -3.971520 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV6 | AutoFitV7 | funding_raised_usd | 44 | 0.162989 | 0.172332 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV6 | AutoFitV7 | investors_count | 44 | -4.299484 | -28.135965 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV6 | AutoFitV7 | is_funded | 16 | -2.661028 | -8.216656 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV7 | AutoFitV71 | funding_raised_usd | 44 | -0.131639 | -0.139524 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV7 | AutoFitV71 | investors_count | 44 | 2.500045 | 14.941679 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV7 | AutoFitV71 | is_funded | 16 | -8.265286 | -24.403283 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV71 | AutoFitV72 | funding_raised_usd | 8 | 0.335957 | 0.350670 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV71 | AutoFitV72 | investors_count | 12 | -0.004231 | -0.025918 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
| AutoFitV71 | AutoFitV72 | is_funded | 4 | 7.087291 | 18.576917 | docs/benchmarks/block3_truth_pack/autofit_step_deltas.csv |
<!-- END AUTO:AUTOFIT_VERSION_LADDER -->

## High-Value SOTA Components For This Freeze

<!-- BEGIN AUTO:HIGH_VALUE_SOTA_COMPONENTS -->
| feature_component | winner_evidence | affected_subtasks | why_effective | integration_priority | risk | verification_test | evidence_path |
|---|---|---|---|---|---|---|---|
| Deep sequence inductive bias | investors_count winners by category: deep_classical=34, transformer_sota=10 | target_family:count; task2_forecast investors_count | Deep temporal models preserve count trajectory structure better than current AutoFit count lane. | high | Over-parameterized deep models may become unstable on sparse slices. | Track median gap reduction on investors_count under strict comparability. | docs/benchmarks/block3_truth_pack/condition_leaderboard.csv |
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
| snapshot_ts | 2026-02-22T21:40:33.693738+00:00 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_total | 8 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_total | 30 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_by_partition | {"batch": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_by_partition | {"batch": 22, "gpu": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

### Pending Reasons

| reason | count | evidence_path |
|---|---|---|
| (QOSMaxJobsPerUserLimit) | 22 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| (QOSGrpNodeLimit) | 8 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

### Prefix Status (p7/p7r/p7x/p7xF)

| prefix | source | state_counts | evidence_path |
|---|---|---|---|
| p7 | squeue | {"PENDING": 1} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | squeue | {"PENDING": 3, "RUNNING": 1} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | squeue | {"RUNNING": 5} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | squeue | {"PENDING": 17, "RUNNING": 2} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7 | sacct | {"CANCELLED": 66, "COMPLETED": 177, "FAILED": 12, "OUT_OF_ME": 4, "PENDING": 1} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | sacct | {"CANCELLED": 4, "COMPLETED": 6, "PENDING": 3, "RUNNING": 1} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | sacct | {"CANCELLED": 33, "COMPLETED": 83, "RUNNING": 5} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | sacct | {"CANCELLED": 11, "COMPLETED": 47, "PENDING": 17, "RUNNING": 2} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

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
