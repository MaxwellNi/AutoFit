# Phase 9 / V739 Current Snapshot

> Generated: 2026-03-24T16:33:32.655258+00:00
> Canonical benchmark: `runs/benchmarks/block3_phase9_fair`

## Verified Current Facts

| metric | value | evidence |
| --- | --- | --- |
| metrics_files | 137 | raw metrics scan |
| raw_records | 16023 | raw metrics scan |
| raw_models | 137 | raw metrics scan |
| raw_complete_models | 86 | raw metrics scan |
| raw_partial_models | 51 | raw metrics scan |
| filtered_records | 12178 | `all_results.csv` |
| filtered_models | 107 | `all_results.csv` |
| filtered_complete_models | 67 | `all_results.csv` |
| filtered_partial_models | 40 | `all_results.csv` |
| v739_conditions_landed | 131 | raw metrics scan |
| v739_jobs_live | 0 | `squeue -u npin,cfisch` |
| v739_canonical_phase9_scripts | 13 | V739 script scan |
| v739_legacy_phase10_scripts | 32 | V739 script scan |
| text_embeddings_artifacts_complete | True | `runs/text_embeddings/` |

## Live Queue Snapshot

| metric | value |
| --- | --- |
| jobs_total | 57 |
| running | 11 |
| pending | 46 |
| npin_pending | 46 |
| cfisch_pending | 0 |
| v739_pending | 0 |
| v739_running | 0 |

### Pending Reasons

| reason | count |
| --- | --- |
| (Priority) | 45 |
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
| CARD | 76 | 76 |
| DUET | 76 | 76 |
| FiLM | 76 | 76 |
| FilterTS | 76 | 76 |
| FreTS | 76 | 76 |
| Fredformer | 76 | 76 |
| ModernTCN | 76 | 76 |
| NonstationaryTransformer | 76 | 76 |
| PDF | 76 | 76 |
| PIR | 76 | 76 |
| SCINet | 76 | 76 |
| SRSNet | 76 | 76 |
| SegRNN | 76 | 76 |
| TimeRecipe | 76 | 76 |
| xPatch | 76 | 76 |
| CFPT | 69 | 69 |
| DeformableTST | 69 | 69 |
| MICN | 69 | 69 |
| PathFormer | 69 | 69 |
| SEMPO | 69 | 69 |
| SparseTSF | 69 | 69 |
| TimeBridge | 69 | 69 |
| TimePerceiver | 69 | 69 |
| NegativeBinomialGLM | 21 | 21 |
| AutoFitV1 | 19 | 19 |
| AutoFitV2 | 19 | 19 |
| AutoFitV2E | 19 | 19 |
| AutoFitV3 | 19 | 19 |
| AutoFitV3E | 19 | 19 |
| AutoFitV3Max | 19 | 19 |
| AutoFitV4 | 19 | 19 |
| AutoFitV5 | 19 | 19 |
| AutoFitV6 | 19 | 19 |
| AutoFitV7 | 19 | 19 |
| AutoFitV71 | 19 | 19 |
| AutoFitV72 | 19 | 19 |
| AutoFitV73 | 19 | 19 |
| AutoFitV731 | 19 | 19 |
| AutoFitV732 | 19 | 19 |
| AutoFitV733 | 19 | 19 |
| AutoFitV734 | 19 | 19 |
| AutoFitV735 | 19 | 19 |
| AutoFitV736 | 19 | 19 |
| AutoFitV737 | 19 | 19 |
| AutoFitV738 | 19 | 19 |
| FusedChampion | 19 | 19 |
| NFAdaptiveChampion | 19 | 19 |
