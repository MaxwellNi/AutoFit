# Block 3 Model Benchmark Status

> Last updated: 2026-03-21 12:00 CET
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: direct scan of `runs/benchmarks/block3_phase9_fair/`

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw records | 15515 | direct scan 2026-03-21 12:00 |
| raw models (all) | 137 | direct scan (116 real + 21 retired AutoFit@1) |
| audit-excluded models | 17 | AUDIT_EXCLUDED_MODELS in aggregate_block3_results.py |
| active (leaderboard) models | 99 | 116 raw - 17 excluded |
| raw complete @160 | 77 | direct scan (includes 13 excluded@160) |
| active complete @160 | 64 | 77 - 13 excluded@160 |
| incomplete active models | 35 | 99 - 64 |
| unfixable gaps | 2 | XGBoost@159, XGBoostPoisson@157 (structural OOM) |
| per-ablation | co=2803, s2=2128, ce=2780, e2=2600, ct=2617, fu=2587 | direct scan |
| conditions per model | 160 | t1(72) + t2(48) + t3(40) |
| live jobs | npin 25R + 26PD = 51 | squeue 2026-03-21 12:00 |
| Phase 12 text reruns | 48/48 COMPLETED | all categories |
| Phase 15 new models | 23 submitted, covered by ALL33 accel | `.slurm_scripts/phase15/` |

## V739 Status

| Fact | Value | Evidence |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | Root `AGENTS.md` |
| landed conditions | `119/160` | co=28, ce=28, ct=28, fu=28, cos2=3, e2=4 |
| s2/e2 gap-fill | 4 RUNNING + af739_t1_e2 resubmitted | af739_t{1,2}_s2/t2_e2/t3_e2 R, t1_e2 resubmitted(5268390) |
| quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| mean rank (56 universal conditions) | **#13/80** (top 16%) | computed across 56 universal conditions shared by all 80 models |
| V734-V738 | RETIRED | oracle test-set leakage |

## Incomplete Active Models (35 models)

| Model | Records | Status | Notes |
| --- | ---: | --- | --- |
| XGBoost | 159/160 | ❌ structural OOM | Missing t1/full/is_funded. UNFIXABLE. |
| XGBoostPoisson | 157/160 | ❌ structural OOM | Missing t1/full/is_funded h{7,14,30}. UNFIXABLE. |
| AutoFitV739 | 119/160 | ⏳ af739 4R+1PD @2d | Missing s2+e2; af739_t1_e2 resubmitted 5268390 |
| ETSformer | 113/160 | ⏳ ALL33 accel RUNNING | Missing scattered s2/e2/ct/fu |
| LightTS | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Pyraformer | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Reformer | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Crossformer | 109/160 | ⏳ ALL33 accel RUNNING | Missing scattered conditions |
| MSGNet | 109/160 | ⏳ ALL33 accel RUNNING | same |
| MambaSimple | 109/160 | ⏳ ALL33 accel RUNNING | same |
| PAttn | 109/160 | ⏳ ALL33 accel RUNNING | same |
| DUET | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| DeformableTST | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11, slow (CUDA OOM warnings) |
| FilterTS | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| ModernTCN | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11, large model (12.5M params) |
| PDF | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| PIR | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| PathFormer | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| SEMPO | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| SparseTSF | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| TimeRecipe | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| xPatch | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| CARD | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| CFPT | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| FiLM | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| FreTS | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| Fredformer | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| NonstationaryTransformer | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| SCINet | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| SRSNet | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| SegRNN | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| TimeBridge | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| TimePerceiver | 64/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |

## Live Queue Reality (2026-03-21 12:00 CET)

| Queue slice | Value | Evidence |
| --- | ---: | --- |
| npin gpu RUNNING | 19 | gpu_ac_{co,ce,ct,fu}×12, gpu_fix11×3, af739×4 |
| npin gpu PENDING | 2 | af739_t1_e2(5268390), gpu_cos2_t2(5268391) just submitted |
| npin l40s RUNNING | 5 | l40_ac_t{1,3}_co(120G,9c) + t{1,2,3}_fu(200G,14c) |
| npin l40s PENDING | 10 | l40_ac remaining (5268070-79) |
| npin hopper RUNNING | 1 | hp_ac_t1_fu(5268095) on iris-197 |
| npin hopper PENDING | 14 | hp_ac remaining (5268096-109, besteffort) |
| **total** | **51** (25R + 26PD) | squeue 2026-03-21 12:00 |

**Events since 2026-03-20 12:00**:
- TIMEOUT@2d wave (~06:00 Mar 21): gpu_ac_t{1,2,3}_e2, gpu_cos2_t{1,2}, p15_t{1,2,3}_cos2, p15_new_t{1,2}_co, p15_new_t1_ce, cf_p15_t1_ces2 = 12 jobs
- af739_t1_e2(5266705) TIMEOUT@1d → RESUBMITTED as 5268390 @2d
- gpu_cos2_t2 RESUBMITTED as 5268391 (27 models missing t2 seed2)
- hp_ac_t1_fu first hopper job to start — running on iris-197 with 3 jgoncalves jobs
- L40S: both nodes 64/64 CPUs utilized (5 npin + 1 zkittel running)

## Phase 15: New TSLib Model Expansion (23 models)

**Submitted**: 2026-03-16 | **Status (2026-03-19)**: ALL33 accel GPU — 12 accel + 3 fix11 covering all 23 models
**Code commits**: `e177f6f` (encoder-only), `c4d214e` (6 bugs), `1185617` (n_vars), `0373037` (fix11), `a9162c2` (seed2)
**Migration**: 3-partition parallel strategy (2026-03-20): gpu+l40s+hopper all running identical ALL33 scripts, harness skip prevents duplicates

### Config Audit & Bug Fixes (2026-03-16 13:53 CET, updated 14:24 CET)

Comprehensive vendor source audit identified and fixed 7 bugs:

| Bug | Model(s) | Root Cause | Fix |
| --- | --- | --- | --- |
| `Invalid filter type` | FilterTS | `filter_type="freq"` not valid; valid: "all","predefined","cross_variable" | Changed to `"all"`, embedding to `"fourier_interpolate"` |
| `No module named 'timm'` | DeformableTST | `timm` not in insider env | Installed `timm==1.0.25` |
| `no attribute 'n_vars'` | DeformableTST | Missing `configs.n_vars` in base config | Added `n_vars=enc_in` to base config |
| `no attribute 'noisy_gating'` | DUET | Missing MoE config attrs | Added `noisy_gating=True, num_experts=4, k=2` |
| `configs.gpu` AttributeError | PathFormer | PathFormer init does `torch.device('cuda:{}'.format(configs.gpu))` | Added `gpu=0` to config |
| `num_nodes=1` shape mismatch | PathFormer | RevIN expects `num_features=num_nodes=enc_in`, was hardcoded to 1 | Removed from config; added `num_nodes=enc_in` to base config |
| SEMPO tuple return crash | SEMPO | Returns `(pretrain_heads_list, prediction)` not `(prediction, attn)` | `out[-1] if isinstance(out[0], list) else out[0]` at all 4 call sites |

**Impact**: The 3 currently running npin jobs (5253903/4/5) will fail on DeformableTST/DUET/FilterTS (3/23 models each). PathFormer and SEMPO not yet reached but will also fail on old code. The 9 PENDING jobs (3 npin + 6 cfisch) will pick up the fixed code. Targeted rerun scripts created for 5 errored models × 3 conditions: `.slurm_scripts/phase15/p15_rerun_errors_*.sh`

### Models (23)
CARD, CFPT, DeformableTST, DUET, FiLM, FilterTS, FreTS, Fredformer, MICN,
ModernTCN, NonstationaryTransformer, PDF, PIR, PathFormer, SCINet, SEMPO,
SRSNet, SegRNN, SparseTSF, TimeBridge, TimePerceiver, TimeRecipe, xPatch

### Forward Compatibility Fix
8 encoder-only models (forward(x) instead of standard 4-arg):
DeformableTST, Fredformer, ModernTCN, PDF, PathFormer, SparseTSF, TimeRecipe, xPatch
→ Fixed via `_ENCODER_ONLY_MODELS` frozenset + `_forward_model()` dispatcher

### Excluded Models (5)
| Model | Reason |
| --- | --- |
| Koopa | NaN divergence (§16) |
| CycleNet | Needs `cycle_index` tensor (structural) |
| TQNet | Needs `cycle_index` tensor (structural) |
| Mamba | Needs `mamba_ssm` (MambaSimple used instead) |
| TiRex | Needs NX-AI `tirex` package (not on PyPI; PyPI "tirex" is SIREX/CUME statistical tool) |

### Job Distribution (current: ALL33 3-partition acceleration)
| Partition | Scripts | Mem | CPUs | Scope |
| --- | --- | --- | --- | --- |
| gpu | gpu_t{1,2,3}_{co,ce,ct,fu} (12 scripts) | 150-200G | 7-8 | ALL 33 TSLib models (10 old + 23 new) |
| gpu | gpu_fix11_t{1,2,3} (3 scripts) | 150G | 7 | 11 fix11 models (n_vars/bug-fix) |
| gpu | gpu_fnd_cos2_t{1,2,3}, gpu_fnd_e2_t{1,2} (5 scripts) | 189G | 8 | Chronos2+TTM seed2+edgar |
| l40s | l40_t{1,2,3}_{co,ce,ct,e2,fu} (15 scripts) | 120-200G | 8-14 | ALL 33 TSLib models (iris-snt QOS) |
| hopper | hp_t{1,2,3}_{co,ce,ct,e2,fu} (15 scripts) | 150-200G | 9-12 | ALL 33 TSLib models (besteffort QOS) |

## Text Embeddings

| Fact | Value | Evidence |
| --- | --- | --- |
| artifacts complete | `true` | `docs/benchmarks/phase9_current_snapshot.json` |
| total rows | `5774931` | `runs/text_embeddings/embedding_metadata.json` |
| unique texts | `69697` | `runs/text_embeddings/embedding_metadata.json` |
| entities | `22569` | `runs/text_embeddings/embedding_metadata.json` |
| PCA dimension | `64` | `runs/text_embeddings/embedding_metadata.json` |

## Interpretation

1. V739 partially landed: 119/160 (↑3). 41 missing (s2+e2) covered by 4 af739 RUNNING + 1 resubmitted.
2. Of 99 active models, 64 at 160/160. 35 incomplete covered by 51 SLURM jobs.
3. 2 models have unfixable structural OOM gaps: XGBoost@159, XGBoostPoisson@157.
4. NegativeBinomialGLM: audit-excluded (21 records, structural failure).
5. Phase 12 text reruns: 48/48 COMPLETED. core_text+full: 91/91 models.
6. Phase 15 new TSLib models: 23 submitted, @64/160 (↑3 from 61), ALL33 3-partition accel RUNNING.
7. s2 (core_only_seed2) gap: 27 models missing t2 s2. gpu_cos2_t2 resubmitted (5268391).
8. Hopper partition: 1 RUNNING (hp_ac_t1_fu) + 14 PENDING. First hopper job started!
9. ETSformer: device error → constant predictions. Already in AUDIT_EXCLUDED.
10. Top-5 by mean rank: PatchTST(4.28), NHITS(4.38), NBEATS(5.01), NBEATSx(5.81), ChronosBolt(7.42).

## Immediate Next Actions

1. ~~Land V739 original 112 conditions.~~ ✅ DONE.
2. ~~Phase 12 text reruns.~~ ✅ DONE (48/48 COMPLETED).
3. ~~Phase 15 new models submitted.~~ ✅ DONE (ALL33 accel covering all 23 models).
4. ~~Fix HPC admin complaints~~ ✅ DONE → Admin approved l40s/hopper usage (2026-03-20).
5. Wait for 51 active jobs to complete (25R + 26PD across gpu+l40s+hopper).
6. V739 s2/e2 gap-fill: 4 af739 RUNNING + af739_t1_e2 resubmitted. When complete → V739@160.
7. s2 gap-fill: gpu_cos2_t2 resubmitted for 27 models missing t2 seed2.
8. Chronos2+TTM: ✅ COMPLETE @160 (s2=20/20).
9. Only after all completions → rebuild final leaderboard → start V740+ iteration.
