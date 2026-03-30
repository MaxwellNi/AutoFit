# Phase 9 / V739 Current Snapshot

> Generated: 2026-03-30T15:27:45.471279+00:00
> Canonical benchmark: `runs/benchmarks/block3_phase9_fair`

## Verified Current Facts

| metric | value | evidence |
| --- | --- | --- |
| metrics_files | 137 | raw metrics scan |
| raw_records | 16152 | raw metrics scan |
| raw_models | 137 | raw metrics scan |
| raw_nonretired_models | 116 | raw metrics scan minus retired AutoFit legacy lines |
| raw_retired_autofit_models | 21 | raw metrics scan |
| raw_complete_models | 75 | raw metrics scan |
| raw_partial_models | 62 | raw metrics scan |
| filtered_records | 12300 | `all_results.csv` |
| filtered_models | 107 | `all_results.csv` (includes retired AutoFit legacy lines if they pass filters) |
| filtered_nonretired_models | 86 | `all_results.csv` minus retired AutoFit legacy lines |
| filtered_retired_autofit_models | 21 | `all_results.csv` |
| filtered_complete_models | 55 | `all_results.csv` |
| filtered_partial_models | 52 | `all_results.csv` |
| v739_conditions_landed | 132 | raw metrics scan |
| v739_jobs_live | 5 | `squeue -u npin,cfisch` |
| v739_canonical_phase9_scripts | 13 | V739 script scan |
| v739_legacy_phase10_scripts | 32 | V739 script scan |
| text_embeddings_artifacts_complete | True | `runs/text_embeddings/` |

## Live Queue Snapshot

| metric | value |
| --- | --- |
| jobs_total | 42 |
| running | 13 |
| pending | 29 |
| npin_pending | 29 |
| cfisch_pending | 0 |
| v739_pending | 0 |
| v739_running | 5 |

### Pending Reasons

| reason | count |
| --- | --- |
| (Priority) | 28 |
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
| XGBoost | 159 | 159 |
| XGBoostPoisson | 157 | 157 |
| AutoFitV739 | 132 | 132 |
| Chronos2 | 114 | 160 |
| TTM | 114 | 160 |
| Crossformer | 105 | 109 |
| MSGNet | 105 | 109 |
| MambaSimple | 105 | 109 |
| MultiPatchFormer | 105 | 109 |
| PAttn | 105 | 109 |
| TimeFilter | 105 | 109 |
| ETSformer | 93 | 113 |
| LightTS | 93 | 113 |
| Pyraformer | 93 | 113 |
| Reformer | 93 | 113 |
| CARD | 83 | 83 |
| DUET | 83 | 83 |
| FiLM | 83 | 83 |
| FilterTS | 83 | 83 |
| FreTS | 83 | 83 |
| Fredformer | 83 | 83 |
| ModernTCN | 83 | 83 |
| NonstationaryTransformer | 83 | 83 |
| PDF | 83 | 83 |
| PIR | 83 | 83 |
| SCINet | 83 | 83 |
| SRSNet | 83 | 83 |
| SegRNN | 83 | 83 |
| TimeRecipe | 83 | 83 |
| xPatch | 83 | 83 |
| CFPT | 69 | 69 |
| DeformableTST | 69 | 69 |
| MICN | 69 | 69 |
| PathFormer | 69 | 69 |
| SEMPO | 69 | 69 |
| SparseTSF | 69 | 69 |
| TimeBridge | 69 | 69 |
| TimePerceiver | 69 | 69 |
| NegativeBinomialGLM | 21 | 21 |
| AutoFitV1 | 20 | 20 |
| AutoFitV2 | 20 | 20 |
| AutoFitV2E | 20 | 20 |
| AutoFitV3 | 20 | 20 |
| AutoFitV3E | 20 | 20 |
| AutoFitV3Max | 20 | 20 |
| AutoFitV4 | 20 | 20 |
| AutoFitV5 | 20 | 20 |
| AutoFitV6 | 20 | 20 |
| AutoFitV7 | 20 | 20 |
| AutoFitV71 | 20 | 20 |
| AutoFitV72 | 20 | 20 |
| AutoFitV73 | 20 | 20 |
| AutoFitV731 | 20 | 20 |
| AutoFitV732 | 20 | 20 |
| AutoFitV733 | 20 | 20 |
| AutoFitV734 | 20 | 20 |
| AutoFitV735 | 20 | 20 |
| AutoFitV736 | 20 | 20 |
| AutoFitV737 | 20 | 20 |
| AutoFitV738 | 20 | 20 |
| FusedChampion | 20 | 20 |
| NFAdaptiveChampion | 20 | 20 |
