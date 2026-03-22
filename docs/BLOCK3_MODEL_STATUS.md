# Block 3 Model Benchmark Status

> Last updated: 2026-03-22 11:50 CET
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: direct scan of `runs/benchmarks/block3_phase9_fair/`

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw records | **15750** | direct scan 2026-03-22 11:40 (+142 from 15608) |
| raw models (all) | 137 | direct scan (116 real + 21 retired AutoFit@1) |
| audit-excluded models | **24** | AUDIT_EXCLUDED_MODELS in aggregate_block3_results.py (Finding A-H + Structural) |
| active (leaderboard) models | **92** | 116 raw - 24 excluded |
| raw complete @160 | 77 | direct scan (includes 13 excluded@160) |
| active complete @160 | 64 | 77 - 13 excluded@160 |
| incomplete active models | 28 | 92 - 64 |
| unfixable gaps | 2 | XGBoost@159, XGBoostPoisson@157 (structural OOM) |
| conditions per model | 160 | t1(72) + t2(48) + t3(40) |
| live jobs | npin **25R + 29PD = 54** | squeue 2026-03-22 11:40 |
| Phase 12 text reruns | 48/48 COMPLETED | all categories |
| Phase 15 new models | 23 submitted (15 valid + 8 broken excluded), accel_v2 scripts | `.slurm_scripts/phase15/accel_v2/` |

## V739 Status

| Fact | Value | Evidence |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | Root `AGENTS.md` |
| landed conditions | `124/160` | co=28, ce=28, ct=28, fu=28, cos2=3+?, e2=5+? (+4 in 9h) |
| s2/e2 gap-fill | 2R + 3 TIMEOUT | af739_t1_e2(23h) + t3_e2(39h) RUNNING; t1_s2/t2_s2/t2_e2 TIMEOUT |
| quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| mean rank (56 universal conditions) | **#13/80** (top 16%) | computed across 56 universal conditions shared by all 80 models |
| V734-V738 | RETIRED | oracle test-set leakage |

## Incomplete Active Models (35 models)

| Model | Records | Status | Notes |
| --- | ---: | --- | --- |
| XGBoost | 159/160 | ❌ structural OOM | Missing t1/full/is_funded. UNFIXABLE. |
| XGBoostPoisson | 157/160 | ❌ structural OOM | Missing t1/full/is_funded h{7,14,30}. UNFIXABLE. |
| AutoFitV739 | **124**/160 | ⏳ af739 2R @2d | Missing s2(~22)+e2(~14); 3 TIMEOUT today, very slow ~3 conds/2d |
| ETSformer | 113/160 | ⏳ ALL33 accel RUNNING | Missing scattered s2/e2/ct/fu |
| LightTS | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Pyraformer | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Reformer | 113/160 | ⏳ ALL33 accel RUNNING | same |
| Crossformer | 109/160 | ⏳ ALL33 accel RUNNING | Missing scattered conditions |
| MSGNet | 109/160 | ⏳ ALL33 accel RUNNING | same |
| MambaSimple | 109/160 | ⏳ ALL33 accel RUNNING | same |
| PAttn | 109/160 | ⏳ ALL33 accel RUNNING | same |
| DUET | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| FilterTS | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| ModernTCN | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11, large model (12.5M params) |
| PDF | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| PIR | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| TimeRecipe | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| xPatch | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 fix11 model |
| CARD | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 new model (speed bottleneck: ~17-50min/epoch) |
| FiLM | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| FreTS | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| Fredformer | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| NonstationaryTransformer | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| SCINet | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| SRSNet | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |
| SegRNN | **69**/160 | ⏳ accel_v2 RUNNING | Phase 15 new model |

## Live Queue Reality (2026-03-22 11:50 CET)

| Queue slice | Value | Evidence |
| --- | ---: | --- |
| npin gpu RUNNING | 20 | 17 g2_ac_v2 (9h in) + af739_t1_e2(23h) + af739_t3_e2(39h) + gpu_cos2_t2(23h) |
| npin gpu PENDING | 0 | — |
| npin l40s RUNNING | 4 | l2_ac_t1_ce/co/ct/e2 (just started after old v1 cancel) |
| npin l40s PENDING | 13 | l2_ac_v2 remaining (Priority/Resources) |
| npin hopper RUNNING | 1 | h2_ac_t1_ce (5min in, besteffort priority=1) |
| npin hopper PENDING | 16 | h2_ac_v2 remaining (Priority) |
| **total** | **54** (25R + 29PD) | squeue 2026-03-22 11:40 |

**Events since 2026-03-22 02:30**:
- **7 TIMEOUT@2d**: gpu_fix11_t1_ce, gpu_ac_t{1,2,3}_fu, af739_t{1,2}_s2, af739_t2_e2
- **8 cancelled** (old v1 with broken models): l40_ac_t{1,3}_co + t{1,2,3}_fu (5 l40s jobs at 47h), hp_ac_t{1,2,3}_fu (3 hopper PENDING)
- **L40S v2 started**: 4 l2_ac jobs immediately launched after v1 cancellation
- **Hopper first v2 slot**: h2_ac_t1_ce started on iris-197
- **V739 +4**: from 120→124 conditions (af739_t{1,2}_s2+ af739_t2_e2 contributed before timeout)
- **P15 +2**: all 23 P15 models from 67→69 conditions (g2_ac v2 contributing)
- **NegBinGLM added to AUDIT_EXCLUDED_MODELS**: code now has 24 models (was 23)

**Partition constraints (sinfo verified)**:
- L40S: 2 nodes (iris-198/199), 32c/503G each. CPU bottleneck: 14c/job max 4 jobs.
- Hopper: 1 node (iris-197), 112c/201G. RAM bottleneck: 189G/job = max 1 job.
- Hopper QOS: besteffort only (priority=1). No iris-hopper access.

**g2_ac v2 projections (9h in, ~39h remaining in 48h timelimit)**:
- e2 (3 jobs): 10/23 → WILL COMPLETE ✓
- ct (3 jobs): 5/23 → SHOULD COMPLETE ✓
- co (3 jobs): 3/23 → needs requeue (auto via --signal=USR1@120)
- s2 (3 jobs): 3/23 → needs requeue
- ce (3 jobs): 1/23 (stuck on CARD) → needs 2+ requeues
- fu (3 jobs): 1/23 (stuck on CARD) → needs 2+ requeues

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

1. V739 partially landed: 124/160 (+4 today). 36 missing (s2+e2) covered by 2 af739 RUNNING (~3 conds per 2d job).
2. Of 92 active models, 64 at 160/160. 28 incomplete covered by 54 SLURM jobs (25R + 29PD).
3. P15 new models at 69/160 each. e2/ct will complete in first g2_ac_v2 run. co/s2/ce/fu need auto-requeue.
4. Partition limits verified: L40S max 4 jobs (CPU), Hopper max 1 job (RAM), Hopper besteffort only.
5. NegBinGLM now included in AUDIT_EXCLUDED_MODELS (24 total excluded).
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
