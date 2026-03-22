# Block 3 Model Benchmark Status

> Last updated: 2026-03-22 02:30 CET
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: direct scan of `runs/benchmarks/block3_phase9_fair/`

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw records | 15608 | direct scan 2026-03-22 02:10 |
| raw models (all) | 137 | direct scan (116 real + 21 retired AutoFit@1) |
| audit-excluded models | 24 | AUDIT_EXCLUDED_MODELS in aggregate_block3_results.py (17 old + 7 new Finding H) |
| active (leaderboard) models | 92 | 116 raw - 24 excluded |
| raw complete @160 | 77 | direct scan (includes 13 excluded@160) |
| active complete @160 | 64 | 77 - 13 excluded@160 |
| incomplete active models | 28 | 92 - 64 |
| unfixable gaps | 2 | XGBoost@159, XGBoostPoisson@157 (structural OOM) |
| per-ablation | co=2803, s2=2004, ce=2780, e2=2490, ct=2764, fu=2767 | direct scan |
| conditions per model | 160 | t1(72) + t2(48) + t3(40) |
| live jobs | npin 33R + 34PD = 67 | squeue 2026-03-22 02:30 |
| Phase 12 text reruns | 48/48 COMPLETED | all categories |
| Phase 15 new models | 23 submitted (15 valid + 8 broken excluded), accel_v2 scripts | `.slurm_scripts/phase15/accel_v2/` |

## V739 Status

| Fact | Value | Evidence |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | Root `AGENTS.md` |
| landed conditions | `120/160` | co=28, ce=28, ct=28, fu=28, cos2=3, e2=5 |
| s2/e2 gap-fill | 5 RUNNING @2d | af739_t{1,2}_s2/t{1,2,3}_e2 RUNNING, very slow (~3 conds/2d) |
| quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| mean rank (56 universal conditions) | **#13/80** (top 16%) | computed across 56 universal conditions shared by all 80 models |
| V734-V738 | RETIRED | oracle test-set leakage |

## Incomplete Active Models (35 models)

| Model | Records | Status | Notes |
| --- | ---: | --- | --- |
| XGBoost | 159/160 | ❌ structural OOM | Missing t1/full/is_funded. UNFIXABLE. |
| XGBoostPoisson | 157/160 | ❌ structural OOM | Missing t1/full/is_funded h{7,14,30}. UNFIXABLE. |
| AutoFitV739 | 120/160 | ⏳ af739 5R @2d | Missing s2(25)+e2(23); very slow, ~3 conds per 2d job |
| ETSformer | 113/160 | ⏳ ALL33 accel RUNNING | Missing scattered s2/e2/ct/fu |
| LightTS | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Pyraformer | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Reformer | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Crossformer | 109/160 | ⏳ ALL33 accel RUNNING | Missing scattered conditions |
| MSGNet | 109/160 | ⏳ ALL33 accel RUNNING | same |
| MambaSimple | 109/160 | ⏳ ALL33 accel RUNNING | same |
| PAttn | 109/160 | ⏳ ALL33 accel RUNNING | same |
| DUET | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| FilterTS | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| ModernTCN | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11, large model (12.5M params) |
| PDF | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| PIR | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| TimeRecipe | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| xPatch | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| CARD | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| FiLM | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| FreTS | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| Fredformer | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| NonstationaryTransformer | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| SCINet | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| SRSNet | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| SegRNN | 67/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |

## Live Queue Reality (2026-03-22 02:30 CET)

| Queue slice | Value | Evidence |
| --- | ---: | --- |
| npin gpu RUNNING | 22 | 17 g2_ac_v2 + af739_t{1,2}_{s2,e2} + gpu_cos2_t2 |
| npin gpu PENDING | 0 | — |
| npin l40s RUNNING | 5 | l40_ac_t{1,3}_co + t{1,2,3}_fu (old v1 scripts) |
| npin l40s PENDING | 17 | l2_ac_v2 all ablations (5269748-5269764) |
| npin hopper RUNNING | 3 | hp_ac_t{1,2,3}_fu (old v1 scripts) |
| npin hopper PENDING | 17 | h2_ac_v2 all ablations (5269765-5269781) |
| npin old gpu | 3 | gpu_ac_t{2,3}_fu (will timeout ~04:00), af739_t3_e2 |
| **total** | **67** (33R + 34PD) | squeue 2026-03-22 02:30 |

**Events since 2026-03-21 12:00**:
- TIMEOUT@2d wave: gpu_ac_t{1,2,3}_{co,ce,ct} + gpu_fix11_t{1,2}_co + gpu_fix11_t1_ce + gpu_ac_t1_fu = 13 jobs
- af739 slow: only ~3 conditions per 2-day allocation, V739 at 120/160
- **Finding H**: 8 P15 models produce 100% constant predictions (0/67 fairness pass): CFPT, DeformableTST, PathFormer, SEMPO, SparseTSF, TimeBridge, TimePerceiver, MICN(already excluded)
- **Optimization**: cancelled 22 pending l40s/hopper v1 jobs, created accel_v2 scripts with 23 models (removed 10 broken/excluded models)
- **accel_v2 submitted**: 17 GPU (all RUNNING immediately) + 17 L40S + 17 Hopper = 51 new optimized jobs
- DeformableTST blocked EVERY running job (CUDA OOM on V100, dimension mismatch on H100)
- PathFormer also produces constant predictions on all conditions tested

## Phase 15: New TSLib Model Expansion (23 models)

**Submitted**: 2026-03-16 | **Status (2026-03-22 02:30)**: 8 models EXCLUDED (Finding H: 100% constant predictions), 15 valid models + 8 old TSLib = 23 models in accel_v2 scripts
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

### Models (23 submitted → 15 valid + 8 excluded)
**Valid (15)**: CARD, DUET, FiLM, FilterTS, FreTS, Fredformer, ModernTCN, NonstationaryTransformer, PDF, PIR, SCINet, SRSNet, SegRNN, TimeRecipe, xPatch
**Excluded Finding H (7 new + MICN already excluded = 8)**: CFPT, DeformableTST, MICN, PathFormer, SEMPO, SparseTSF, TimeBridge, TimePerceiver — all 100% constant predictions, 0% fairness pass

### Forward Compatibility Fix
8 encoder-only models (forward(x) instead of standard 4-arg):
DeformableTST, Fredformer, ModernTCN, PDF, PathFormer, SparseTSF, TimeRecipe, xPatch
→ Fixed via `_ENCODER_ONLY_MODELS` frozenset + `_forward_model()` dispatcher

### Excluded Models (5 structural + 8 Finding H = 13)
| Model | Reason |
| --- | --- |
| Koopa | NaN divergence (§16) |
| CycleNet | Needs `cycle_index` tensor (structural) |
| TQNet | Needs `cycle_index` tensor (structural) |
| Mamba | Needs `mamba_ssm` (MambaSimple used instead) |
| TiRex | Needs NX-AI `tirex` package (not on PyPI) |
| CFPT | 100% constant predictions, 0/67 fairness pass (Finding H) |
| DeformableTST | CUDA OOM all GPUs + dimension mismatch + constant predictions (Finding H) |
| PathFormer | 100% constant predictions, 0/92 fairness pass (Finding H) |
| SEMPO | 100% constant predictions, 0/91 fairness pass (Finding H) |
| SparseTSF | 100% constant predictions, 0/91 fairness pass (Finding H) |
| TimeBridge | 100% constant predictions, 0/85 fairness pass (Finding H) |
| TimePerceiver | 100% constant predictions, 0/85 fairness pass (Finding H) |
| MICN | 100% constant predictions, 0/105 fairness pass (already excluded Finding G) |

### Job Distribution (current: accel_v2 3-partition, 23 models)
| Partition | Scripts | Mem | CPUs | Scope |
| --- | --- | --- | --- | --- |
| gpu | g2_ac_t{1,2,3}_{co,s2,ce,e2,ct,fu} (17 scripts) | 150-200G | 7-8 | 23 working TSLib (8 old + 15 valid P15) |
| l40s | l2_ac_t{1,2,3}_{co,s2,ce,e2,ct,fu} (17 scripts) | 120-200G | 8-14 | 23 working TSLib (iris-snt QOS) |
| hopper | h2_ac_t{1,2,3}_{co,s2,ce,e2,ct,fu} (17 scripts) | 150-200G | 9-12 | 23 working TSLib (besteffort QOS) |
| gpu (old) | af739_t{1,2}_{s2,e2}, af739_t3_e2, gpu_cos2_t2 | 189-200G | 8 | V739 + seed2 gap |

**accel_v2 improvement**: Removed 10 broken/excluded models (CFPT, DeformableTST, MICN, MultiPatchFormer, PathFormer, SEMPO, SparseTSF, TimeBridge, TimeFilter, TimePerceiver) → ~30% faster per job. Scripts in `.slurm_scripts/phase15/accel_v2/`

## Text Embeddings

| Fact | Value | Evidence |
| --- | --- | --- |
| artifacts complete | `true` | `docs/benchmarks/phase9_current_snapshot.json` |
| total rows | `5774931` | `runs/text_embeddings/embedding_metadata.json` |
| unique texts | `69697` | `runs/text_embeddings/embedding_metadata.json` |
| entities | `22569` | `runs/text_embeddings/embedding_metadata.json` |
| PCA dimension | `64` | `runs/text_embeddings/embedding_metadata.json` |

## Interpretation

1. V739 partially landed: 120/160 (↑1). 40 missing (s2+e2) covered by 5 af739 RUNNING (~3 conds per 2d job).
2. Of 92 active models, 64 at 160/160. 28 incomplete covered by 67 SLURM jobs.
3. 2 models have unfixable structural OOM gaps: XGBoost@159, XGBoostPoisson@157.
4. NegativeBinomialGLM: audit-excluded (21 records, structural failure).
5. Phase 12 text reruns: 48/48 COMPLETED. core_text+full: 91/91 models.
6. Phase 15: 15 valid + 8 excluded (Finding H), @67/160 each, accel_v2 RUNNING.
7. **Finding H**: 8 P15 models produce 100% constant predictions. Added to AUDIT_EXCLUDED_MODELS.
8. accel_v2 scripts: 23 models × 3 partitions × 17 scripts each = 51 new jobs (30% faster).
9. s2 (core_only_seed2) gap: gpu_cos2_t2 RUNNING (5268391). Key priority.
10. e2 (core_edgar_seed2) gap: ETSformer/LightTS/Pyraformer/Reformer at 0/28 e2. accel_v2 covers this.
11. Top-5 by mean rank: PatchTST(4.28), NHITS(4.38), NBEATS(5.01), NBEATSx(5.81), ChronosBolt(7.42).

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
