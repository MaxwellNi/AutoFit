# Block 3 Model Benchmark Status

> Last updated: 2026-03-20 10:00 CET
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: direct scan of `runs/benchmarks/block3_phase9_fair/`

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw metrics files | 132+ | direct scan 2026-03-20 |
| raw records | 15211 | direct scan 2026-03-20 10:00 |
| raw models (all) | 137 | direct scan (116 real + 21 retired AutoFit@1) |
| audit-excluded models | 17 | AUDIT_EXCLUDED_MODELS in aggregate_block3_results.py |
| active (leaderboard) models | 99 | 116 raw - 17 excluded |
| raw complete @160 | 77 | direct scan (includes 13 excluded@160) |
| active complete @160 | 64 | 77 - 13 excluded@160 |
| incomplete active models | 35 | 99 - 64 |
| unfixable gaps | 2 | XGBoost@159, XGBoostPoisson@157 (structural OOM) |
| total unique records | 15015 | deduped (task, ablation, horizon, target) |
| max possible records | ~18080 | ~99 active × ~160 each + excluded |
| overall completion | 83.0% | 15015 / ~18080 |
| conditions per model | 160 | t1(72) + t2(48) + t3(40) |
| live jobs | npin 27R + cfisch 1R + 4 af739 PD | squeue 2026-03-20 |
| Phase 12 text reruns | 48/48 COMPLETED | all categories |
| Phase 15 new models | 23 submitted, covered by ALL33 accel | `.slurm_scripts/phase15/` |

## V739 Status

| Fact | Value | Evidence |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | Root `AGENTS.md` |
| landed conditions | `116/160` | co=28, ce=28, ct=28, fu=28, cos2=1, e2=3; missing 44 s2/e2 |
| s2/e2 gap-fill | 1 RUNNING + 4 PENDING @2d | af739_t1_e2(5266705) R, 4 resubmitted(5267972-75) PD |
| quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| mean rank (56 universal conditions) | **#13/80** (top 16%) | computed across 56 universal conditions shared by all 80 models |
| V734-V738 | RETIRED | oracle test-set leakage |

## Incomplete Active Models (35 models)

| Model | Records | Status | Notes |
| --- | ---: | --- | --- |
| XGBoost | 159/160 | ❌ structural OOM | Missing t1/full/is_funded. UNFIXABLE. |
| XGBoostPoisson | 157/160 | ❌ structural OOM | Missing t1/full/is_funded h{7,14,30}. UNFIXABLE. |
| AutoFitV739 | 116/160 | ⏳ af739 1R+4PD @2d | Missing cos2+e2; 5267972-75+5266705 |
| ETSformer | 113/160 | ⏳ ALL33 accel RUNNING | Missing scattered s2/e2/ct/fu |
| LightTS | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Pyraformer | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Reformer | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Crossformer | 109/160 | ⏳ ALL33 accel RUNNING | Missing scattered conditions |
| MSGNet | 109/160 | ⏳ ALL33 accel RUNNING | same |
| MambaSimple | 109/160 | ⏳ ALL33 accel RUNNING | same |
| PAttn | 109/160 | ⏳ ALL33 accel RUNNING | same |
| DUET | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| DeformableTST | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11, slow (CUDA OOM warnings) |
| FilterTS | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| ModernTCN | 52/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11, large model (12.5M params) |
| PDF | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| PIR | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| PathFormer | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| SEMPO | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| SparseTSF | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| TimeRecipe | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| xPatch | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 fix11 model |
| CARD | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| CFPT | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| FiLM | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| FreTS | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| Fredformer | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| NonstationaryTransformer | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| SCINet | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| SRSNet | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| SegRNN | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| TimeBridge | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |
| TimePerceiver | 54/160 | ⏳ ALL33 accel RUNNING | Phase 15 new model |

## Live Queue Reality (2026-03-20 10:00 CET)

| Queue slice | Value | Evidence |
| --- | ---: | --- |
| npin gpu RUNNING | 27 | af739_t1_e2(1), ALL33 co/ce/ct/fu/e2(15), fix11(3), cos2(2), p15 cos2(3), p15 co/ce(3) |
| npin l40s RUNNING | 0 | All l40s cancelled (admin complaint + redundancy) |
| npin gpu PENDING | 4 | af739 s2/e2 (5267972-75) |
| cfisch gpu RUNNING | 1 | cf_p15_t1_ces2 |
| **total** | **32** (28R + 4PD) | squeue 2026-03-20 10:00 |

**Actions taken 2026-03-20 10:00**:
- l40_cos2_t2 (5262822): CANCELLED — admin complained 14 CPUs / 1 CPU utilized / 88G used. Redundant with gpu_cos2_t2 RUNNING.
- l40_ac_t3_ce (5264037): CANCELLED — redundant with gpu_ac_t3_ce RUNNING.
- af739_t1_s2, af739_t2_s2, af739_t2_e2, af739_t3_e2: All TIMEOUT@1d → scripts bumped to 2d → resubmitted (5267972-5267975)
- af739_t1_e2 (5266705): still RUNNING (6h, 1d limit, couldn't extend via scontrol)
- gpu_fnd_cos2_t1/t2, gpu_fnd_e2_t1/t2/t3: 5 COMPLETED (7-26 min each) → Chronos2+TTM now @160 ✅

## Phase 15: New TSLib Model Expansion (23 models)

**Submitted**: 2026-03-16 | **Status (2026-03-19)**: ALL33 accel GPU — 12 accel + 3 fix11 covering all 23 models
**Code commits**: `e177f6f` (encoder-only), `c4d214e` (6 bugs), `1185617` (n_vars), `0373037` (fix11), `a9162c2` (seed2)
**Migration**: All P15 hopper/l40s jobs cancelled 2026-03-19, resubmitted on gpu at 7-8 CPUs (admin compliance)

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

### Job Distribution (current: ALL33 acceleration)
| Partition | Scripts | Mem | CPUs | Scope |
| --- | --- | --- | --- | --- |
| gpu | gpu_t{1,2,3}_{co,ce,ct,fu} (12 scripts) | 150-200G | 7-8 | ALL 33 TSLib models (10 old + 23 new) |
| gpu | gpu_fix11_t{1,2,3} (3 scripts) | 150G | 7 | 11 fix11 models (n_vars/bug-fix) |
| gpu | gpu_fnd_cos2_t{1,2,3}, gpu_fnd_e2_t{1,2} (5 scripts) | 189G | 8 | Chronos2+TTM seed2+edgar |

## Text Embeddings

| Fact | Value | Evidence |
| --- | --- | --- |
| artifacts complete | `true` | `docs/benchmarks/phase9_current_snapshot.json` |
| total rows | `5774931` | `runs/text_embeddings/embedding_metadata.json` |
| unique texts | `69697` | `runs/text_embeddings/embedding_metadata.json` |
| entities | `22569` | `runs/text_embeddings/embedding_metadata.json` |
| PCA dimension | `64` | `runs/text_embeddings/embedding_metadata.json` |

## Interpretation

1. V739 partially landed: 116/160. 44 missing (s2+e2) covered by 5 af739 scripts (1 RUNNING + 4 PENDING @2d).
2. Of 99 active models, 64 at 160/160 (↑2: Chronos2+TTM COMPLETE). 35 incomplete covered by 32 SLURM jobs.
3. 2 models have unfixable structural OOM gaps: XGBoost@159, XGBoostPoisson@157.
4. NegativeBinomialGLM: audit-excluded (21 records, structural failure).
5. Phase 12 text reruns: 48/48 COMPLETED. core_text+full: 91/91 models.
6. Phase 15 new TSLib models: 23 submitted, @54/160, ALL33 accel RUNNING.
7. l40s partition: ZERO jobs remaining. NEVER submit new l40s jobs.
8. HPC policy: GPU only, 7 CPUs for co/ce/ct, 8 for fu/e2, max 200G.
9. Top-5 by mean rank: PatchTST(4.28), NHITS(4.38), NBEATS(5.01), NBEATSx(5.81), ChronosBolt(7.42).
10. NBEATS dominant champion: 65/160 wins (40.6%). Text embeddings HURT (core_text wins 11.7%).

## Immediate Next Actions

1. ~~Land V739 original 112 conditions.~~ ✅ DONE.
2. ~~Phase 12 text reruns.~~ ✅ DONE (48/48 COMPLETED).
3. ~~Phase 15 new models submitted.~~ ✅ DONE (ALL33 accel covering all 23 models).
4. ~~Fix HPC admin complaints~~ ✅ DONE (all hopper+l40s cancelled, migrated to gpu @ 7-8 CPUs).
5. Wait for 44 active jobs to complete (24R + 20PD on gpu/l40s).
6. V739 s2/e2 gap-fill: 5 af739 scripts RUNNING. When complete → V739@160.
7. Chronos2+TTM seed2/e2: 5 gpu_fnd scripts PENDING. When complete → both@160.
8. Only after all completions → rebuild final leaderboard → start V740+ iteration.
