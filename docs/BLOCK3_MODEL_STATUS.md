# Block 3 Model Benchmark Status

> Last updated: 2026-03-19 17:30 CET
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: direct scan of `runs/benchmarks/block3_phase9_fair/`

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw metrics files | 132 | direct scan 2026-03-19 |
| raw records | 14785 | direct scan 2026-03-19 17:30 |
| raw models (all) | 137 | direct scan (114 real + 23 retired AutoFit@1) |
| audit-excluded models | 17 | AUDIT_EXCLUDED_MODELS in aggregate_block3_results.py |
| active (leaderboard) models | 97 | 114 raw - 17 excluded |
| raw complete @160 | 75 | direct scan (includes 13 excluded@160) |
| active complete @160 | 62 | 75 - 13 excluded@160 |
| incomplete active models | 35 | 97 - 62 |
| unfixable gaps | 2 | XGBoost@159, XGBoostPoisson@157 (structural OOM) |
| total unique records | 14464 | deduped (task, ablation, horizon, target) |
| max possible records | 18080 | 113 models × ~160 each |
| overall completion | 80.0% | 14464 / 18080 |
| conditions per model | 160 | t1(72) + t2(48) + t3(40) |
| live jobs | npin 22R+20PD, cfisch 2R | squeue 2026-03-19 |
| Phase 12 text reruns | 48/48 COMPLETED | all categories |
| Phase 15 new models | 23 submitted, covered by ALL33 accel | `.slurm_scripts/phase15/` |

## V739 Status

| Fact | Value | Evidence |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | Root `AGENTS.md` |
| landed conditions | `112/160` | co=28, ce=28, ct=28, fu=28; missing: cos2=20, ces2(e2)=28 |
| s2/e2 gap-fill | RUNNING | af739_t{1,2}_s2, af739_t{1,2,3}_e2 on gpu |
| quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| mean rank (56 universal conditions) | **#13/80** (top 16%) | computed across 56 universal conditions shared by all 80 models |
| V734-V738 | RETIRED | oracle test-set leakage |

## Incomplete Active Models (38 models)

| Model | Records | Status | Notes |
| --- | ---: | --- | --- |
| XGBoost | 159/160 | ❌ structural OOM | Missing t1/full/is_funded. UNFIXABLE. |
| XGBoostPoisson | 157/160 | ❌ structural OOM | Missing t1/full/is_funded h{7,14,30}. UNFIXABLE. |
| Chronos2 | 114/160 | ⏳ gpu_fnd scripts PENDING | Missing cos2(18) + e2(28) |
| TTM | 114/160 | ⏳ gpu_fnd scripts PENDING | Missing cos2(18) + e2(28) |
| AutoFitV739 | 113/160 | ⏳ af739 scripts RUNNING | Missing cos2+e2; af739_t1_e2 resubmitted (5266705) |
| Crossformer | 104/160 | ⏳ ALL33 accel R+PD | Missing scattered co/ce/ct/fu/cos2/e2 |
| MSGNet | 104/160 | ⏳ ALL33 accel R+PD | same |
| MambaSimple | 104/160 | ⏳ ALL33 accel R+PD | same |
| MultiPatchFormer | 104/160 | ⏳ ALL33 accel R+PD | (audit-excluded: constant predictions) |
| PAttn | 104/160 | ⏳ ALL33 accel R+PD | same |
| TimeFilter | 104/160 | ⏳ ALL33 accel R+PD | (audit-excluded: constant predictions) |
| ETSformer | 92/160 | ⏳ ALL33 accel R+PD | Missing co/ce/ct/fu/e2 partially |
| LightTS | 92/160 | ⏳ ALL33 accel R+PD | same |
| Pyraformer | 92/160 | ⏳ ALL33 accel R+PD | same |
| Reformer | 92/160 | ⏳ ALL33 accel R+PD | same |
| DUET | 36/160 | ⏳ ALL33 accel + fix11 | Phase 15 fix11 model |
| DeformableTST | 36/160 | ⏳ ALL33 accel + fix11 | Phase 15 fix11 model |
| FilterTS | 36/160 | ⏳ ALL33 accel + fix11 | Phase 15 fix11 model |
| ModernTCN | 36/160 | ⏳ ALL33 accel + fix11 | Phase 15 fix11 model, was OOM@128G |
| PDF | 36/160 | ⏳ ALL33 accel + fix11 | Phase 15 fix11 model |
| PIR | 36/160 | ⏳ ALL33 accel + fix11 | Phase 15 fix11 model |
| PathFormer | 36/160 | ⏳ ALL33 accel + fix11 | Phase 15 fix11 model |
| SEMPO | 36/160 | ⏳ ALL33 accel + fix11 | Phase 15 fix11 model |
| SparseTSF | 36/160 | ⏳ ALL33 accel + fix11 | Phase 15 fix11 model |
| TimeRecipe | 36/160 | ⏳ ALL33 accel + fix11 | Phase 15 fix11 model |
| xPatch | 36/160 | ⏳ ALL33 accel + fix11 | Phase 15 fix11 model |
| CARD | 35/160 | ⏳ ALL33 accel PD | Phase 15 new model |
| CFPT | 35/160 | ⏳ ALL33 accel PD | Phase 15 new model |
| FiLM | 35/160 | ⏳ ALL33 accel PD | Phase 15 new model |
| FreTS | 35/160 | ⏳ ALL33 accel PD | Phase 15 new model |
| Fredformer | 35/160 | ⏳ ALL33 accel PD | Phase 15 new model |
| MICN | 35/160 | ⏳ ALL33 accel PD | (audit-excluded: constant predictions) |
| NonstationaryTransformer | 35/160 | ⏳ ALL33 accel PD | Phase 15 new model |
| SCINet | 35/160 | ⏳ ALL33 accel PD | Phase 15 new model |
| SRSNet | 35/160 | ⏳ ALL33 accel PD | Phase 15 new model |
| SegRNN | 35/160 | ⏳ ALL33 accel PD | Phase 15 new model |
| TimeBridge | 35/160 | ⏳ ALL33 accel PD | Phase 15 new model |
| TimePerceiver | 35/160 | ⏳ ALL33 accel PD | Phase 15 new model |

## Live Queue Reality (2026-03-19 17:30 CET)

| Queue slice | Value | Evidence |
| --- | ---: | --- |
| npin gpu RUNNING | 23 | af739 s2/e2(4), ALL33 co/ce/ct/e2(11), cos2(2), p15 cos2(3), p15 co/ce(2), fnd(1) |
| npin l40s RUNNING | 2 | l40_ac_t3_ce(14h), l40_cos2_t2(25h) — legacy 14c, finishing naturally |
| npin gpu PENDING | 13 | af739_t1_e2(resubmit), ALL33 fu(3)+ct(1), fix11(3), fnd cos2/e2(5) |
| cfisch gpu RUNNING | 1 | cf_p15_t1_ces2 |
| cfisch l40s RUNNING | 1 | l40cf_ac_t1_fu (legacy 14c, ~16h left) |
| **total** | **40** (27R + 13PD) | squeue 2026-03-19 17:34 |

**Actions taken 2026-03-19 17:34**:
- af739_t1_e2: TIMEOUT@1d → resubmitted as job 5266705
- l40_cos2_t1: cancelled (PD on l40s, redundant with gpu_cos2_t1 RUNNING)

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

1. V739 is partially landed: 112/160 conditions complete. The 48 missing conditions (seed2 + edgar_seed2) are covered by 5 af739 scripts RUNNING on gpu.
2. Of 97 active models, 62 are at 160/160. The remaining 35 active incomplete models are covered by 44 active SLURM jobs.
3. 2 models have unfixable structural OOM gaps: XGBoost@159, XGBoostPoisson@157 (also active, counted in the 35).
4. NegativeBinomialGLM is audit-excluded (21 records, structural convergence failure).
5. Phase 12 text reruns: 48/48 COMPLETED. core_text: 91/91 models. full: 91/91 models.
6. Phase 15 new TSLib models: 23 models submitted, all covered by ALL33 accel GPU scripts.
7. l40s partition: UNUSABLE for new jobs (MaxMemPerCPU=15G, our workload needs 115G+). 3+1 legacy l40s jobs finishing.
8. HPC resource policy: GPU partition ONLY, 7 CPUs for co/ce/ct, 8 CPUs for fu/e2, max 200G.
9. Top-5 by mean rank: PatchTST (4.28), NHITS (4.38), NBEATS (5.01), NBEATSx (5.81), ChronosBolt (7.42).
10. NBEATS is dominant champion: 65/160 conditions won (40.6%). Text embeddings HURT (core_text wins 11.7%).

## Immediate Next Actions

1. ~~Land V739 original 112 conditions.~~ ✅ DONE.
2. ~~Phase 12 text reruns.~~ ✅ DONE (48/48 COMPLETED).
3. ~~Phase 15 new models submitted.~~ ✅ DONE (ALL33 accel covering all 23 models).
4. ~~Fix HPC admin complaints~~ ✅ DONE (all hopper+l40s cancelled, migrated to gpu @ 7-8 CPUs).
5. Wait for 44 active jobs to complete (24R + 20PD on gpu/l40s).
6. V739 s2/e2 gap-fill: 5 af739 scripts RUNNING. When complete → V739@160.
7. Chronos2+TTM seed2/e2: 5 gpu_fnd scripts PENDING. When complete → both@160.
8. Only after all completions → rebuild final leaderboard → start V740+ iteration.
