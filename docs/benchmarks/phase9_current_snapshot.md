# Phase 9 / V739 Current Snapshot

> Generated: 2026-03-23T09:46:59.887681+00:00
> Canonical benchmark: `runs/benchmarks/block3_phase9_fair`

## Verified Current Facts

| metric | value | evidence |
| --- | --- | --- |
| metrics_files | 137 | raw metrics scan |
| raw_records | 15888 | raw metrics scan |
| raw_models | 137 | raw metrics scan |
| raw_complete_models | 86 | raw metrics scan |
| raw_partial_models | 51 | raw metrics scan |
| filtered_records | 12044 | `all_results.csv` |
| filtered_models | 107 | `all_results.csv` |
| filtered_complete_models | 67 | `all_results.csv` |
| filtered_partial_models | 40 | `all_results.csv` |
| v739_conditions_landed | 126 | raw metrics scan |
| v739_jobs_live | 0 | `squeue -u npin,cfisch` |
| v739_canonical_phase9_scripts | 13 | V739 script scan |
| v739_legacy_phase10_scripts | 32 | V739 script scan |
| text_embeddings_artifacts_complete | True | `runs/text_embeddings/` |

## Live Queue Snapshot

| metric | value |
| --- | --- |
| jobs_total | 58 |
| running | 27 |
| pending | 31 |
| npin_pending | 31 |
| cfisch_pending | 0 |
| v739_pending | 0 |
| v739_running | 0 |

### Pending Reasons

| reason | count |
| --- | --- |
| (Priority) | 30 |
| (Resources) | 1 |

## Text Embedding Artifacts

| field | value |
| --- | --- |
| directory_exists | True |
| artifacts_complete | True |
| parquet_path | runs/text_embeddings/text_embeddings.parquet |
| pca_model_path | runs/text_embeddings/pca_model.pkl |
| n_total_rows | 5774931 |
| n_unique_texts | 69697 |
| n_entities | 22569 |
| pca_dim | 64 |

## V739 Submission Surface

| metric | value |
| --- | --- |
| canonical_phase9_fair_scripts | 13 |
| legacy_phase10_scripts | 32 |

### V739 Script Families

#### phase10_fast_npin

| script | target_class | output_dir |
| --- | --- | --- |
| .slurm_scripts/phase10_fast/v739/v739f_t1_ce.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_edgar |
| .slurm_scripts/phase10_fast/v739/v739f_t1_co.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_only |
| .slurm_scripts/phase10_fast/v739/v739f_t1_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_text |
| .slurm_scripts/phase10_fast/v739/v739f_t1_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/full |
| .slurm_scripts/phase10_fast/v739/v739f_t2_ce.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_edgar |
| .slurm_scripts/phase10_fast/v739/v739f_t2_co.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_only |
| .slurm_scripts/phase10_fast/v739/v739f_t2_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_text |
| .slurm_scripts/phase10_fast/v739/v739f_t2_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/full |
| .slurm_scripts/phase10_fast/v739/v739f_t3_ce.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task3_risk_adjust/autofit/core_edgar |
| .slurm_scripts/phase10_fast/v739/v739f_t3_co.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task3_risk_adjust/autofit/core_only |
| .slurm_scripts/phase10_fast/v739/v739f_t3_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task3_risk_adjust/autofit/full |
| .slurm_scripts/phase10_fast/v739/v739r2_t1_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739/v739r2_t2_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739/v739r_t1_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_text |
| .slurm_scripts/phase10_fast/v739/v739r_t1_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/full |
| .slurm_scripts/phase10_fast/v739/v739r_t2_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_text |
| .slurm_scripts/phase10_fast/v739/v739r_t2_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/full |
| .slurm_scripts/phase10_fast/v739/v739r_t3_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task3_risk_adjust/autofit/full |

#### phase10_fast_cfisch

| script | target_class | output_dir |
| --- | --- | --- |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739g_t1_ce.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_edgar |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739g_t1_co.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_only |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739g_t2_ce.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_edgar |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739g_t2_co.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_only |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r2_t1_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r2_t1_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r2_t2_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r2_t2_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r2_t2_fu_hopper.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739 |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r_t1_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/core_text |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r_t1_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task1_outcome/autofit/full |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r_t2_ct.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/core_text |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r_t2_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task2_forecast/autofit/full |
| .slurm_scripts/phase10_fast/v739_cf/cf_v739r_t3_fu.sh | legacy_phase10 | runs/benchmarks/block3_phase10/v739/task3_risk_adjust/autofit/full |

#### phase12_l40s_cfisch

| script | target_class | output_dir |
| --- | --- | --- |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t1_ce.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/core_edgar |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t1_co.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/core_only |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t2_ce.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task2_forecast/autofit/core_edgar |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t2_co.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task2_forecast/autofit/core_only |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t3_ce.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task3_risk_adjust/autofit/core_edgar |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t3_co.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task3_risk_adjust/autofit/core_only |
| .slurm_scripts/phase12/v739_l40s/cf_v739_l40s_t3_fu.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task3_risk_adjust/autofit/full |

#### phase12_rerun_autofit

| script | target_class | output_dir |
| --- | --- | --- |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t1_ct.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/core_text |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t1_fu.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task1_outcome/autofit/full |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t2_ct.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task2_forecast/autofit/core_text |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t2_fu.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task2_forecast/autofit/full |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t3_ct.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task3_risk_adjust/autofit/core_text |
| .slurm_scripts/phase12/rerun/cf_p12_af39_t3_fu.sh | canonical_phase9_fair | runs/benchmarks/block3_phase9_fair/task3_risk_adjust/autofit/full |

## Partial Raw Models

| model_name | conditions | records |
| --- | --- | --- |
| ETSformer | 93 | 113 |
| LightTS | 93 | 113 |
| Pyraformer | 93 | 113 |
| Reformer | 93 | 113 |
| CARD | 75 | 75 |
| DUET | 75 | 75 |
| FiLM | 75 | 75 |
| FilterTS | 75 | 75 |
| FreTS | 75 | 75 |
| Fredformer | 75 | 75 |
| ModernTCN | 75 | 75 |
| NonstationaryTransformer | 75 | 75 |
| PDF | 75 | 75 |
| PIR | 75 | 75 |
| SCINet | 75 | 75 |
| SRSNet | 75 | 75 |
| SegRNN | 75 | 75 |
| TimeRecipe | 75 | 75 |
| xPatch | 75 | 75 |
| CFPT | 69 | 69 |
| DeformableTST | 69 | 69 |
| MICN | 69 | 69 |
| PathFormer | 69 | 69 |
| SEMPO | 69 | 69 |
| SparseTSF | 69 | 69 |
| TimeBridge | 69 | 69 |
| TimePerceiver | 69 | 69 |
| NegativeBinomialGLM | 21 | 21 |
| AutoFitV1 | 14 | 14 |
| AutoFitV2 | 14 | 14 |
| AutoFitV2E | 14 | 14 |
| AutoFitV3 | 14 | 14 |
| AutoFitV3E | 14 | 14 |
| AutoFitV3Max | 14 | 14 |
| AutoFitV4 | 14 | 14 |
| AutoFitV5 | 14 | 14 |
| AutoFitV6 | 14 | 14 |
| AutoFitV7 | 14 | 14 |
| AutoFitV71 | 14 | 14 |
| AutoFitV72 | 14 | 14 |
| AutoFitV73 | 14 | 14 |
| AutoFitV731 | 14 | 14 |
| AutoFitV732 | 14 | 14 |
| AutoFitV733 | 14 | 14 |
| AutoFitV734 | 14 | 14 |
| AutoFitV735 | 14 | 14 |
| AutoFitV736 | 14 | 14 |
| AutoFitV737 | 14 | 14 |
| AutoFitV738 | 14 | 14 |
| FusedChampion | 14 | 14 |
| NFAdaptiveChampion | 14 | 14 |
