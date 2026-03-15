# Block 3 Model Benchmark Status

> Last updated: 2026-03-15
> Current authority: `docs/CURRENT_SOURCE_OF_TRUTH.md`
> Evidence: direct scan of `runs/benchmarks/block3_phase9_fair/`

## Snapshot

| Metric | Value | Evidence |
| --- | ---: | --- |
| raw metrics files | 91 | direct scan 2026-03-15 |
| raw records | 10213 | direct scan 2026-03-15 (Phase 12 landing) |
| raw models | 91 | direct scan 2026-03-15 |
| raw complete models (≥104) | 80 | direct scan 2026-03-15 |
| raw partial models | 11 | direct scan 2026-03-15 |
| live jobs running | 29 | squeue 2026-03-15 |
| live jobs pending | 19 | squeue 2026-03-15 |
| Phase 12 text rerun jobs | 20R + 14PD | `.slurm_scripts/phase12/rerun/` |

## V739 Status

| Fact | Value | Evidence |
| --- | --- | --- |
| current valid AutoFit line | `AutoFitV739` | Root `AGENTS.md` |
| landed conditions | `112/112` (ALL COMPLETE) | direct scan: 12 metrics.json under `*/autofit/*` |
| ablation breakdown | co=28, ce=28, ct=28, fu=28 | direct scan |
| quality | 0 NaN/Inf, 0 fallback, 100% fairness pass | direct scan |
| mean rank (56 universal conditions) | **#13/80** (top 16%) | computed across 56 universal conditions shared by all 80 models |
| mean rank score | 14.38 | lower is better |
| conditions won (champion) | 3/56 (1 per task) | best MAE in that condition |
| all 12 V739 jobs | COMPLETED | sacct 2026-03-15 |

## Partial Models (11)

| Model | Records | Status | Notes |
| --- | ---: | --- | --- |
| NegativeBinomialGLM | 20/104 | ❌ structural failure | NaN/inf in weights, only is_funded works |
| TimeFilter | 57/104 | ⏳ gap-fill running | TSLib jobs on gpu+l40s |
| MultiPatchFormer | 57/104 | ⏳ gap-fill running | constant predictions (fairness_pass=False) |
| MSGNet | 57/104 | ⏳ gap-fill running | very slow (~3.5 min/epoch) |
| PAttn | 57/104 | ⏳ gap-fill running | TSLib jobs on gpu+l40s |
| MambaSimple | 57/104 | ⏳ gap-fill running | TSLib jobs on gpu+l40s |
| Crossformer | 57/104 | ⏳ gap-fill running | TSLib jobs on gpu+l40s |
| ETSformer | 80/104 | ⏳ gap-fill running | TSLib jobs on gpu |
| LightTS | 80/104 | ⏳ gap-fill running | TSLib jobs on gpu |
| Pyraformer | 80/104 | ⏳ gap-fill running | TSLib jobs on gpu |
| Reformer | 80/104 | ⏳ gap-fill running | TSLib jobs on gpu |

## Live Queue Reality

| Queue slice | Value | Evidence |
| --- | ---: | --- |
| npin l40s RUNNING | 4 | squeue 2026-03-15 (TSLib t1+t3 co/ce gap-fill) |
| npin gpu RUNNING | 15 | squeue 2026-03-15 (Phase 12: deep+foun+irre ct/fu) |
| npin bigmem RUNNING | 5 | squeue 2026-03-15 (Phase 12: stat ct/fu) |
| cfisch gpu RUNNING | 9 | squeue 2026-03-15 (8 TSLib gap-fill + 1 FM seed2) |
| cfisch gpu PENDING | 14 | squeue 2026-03-15 (Phase 12: af39+tran+tsli ct/fu) |
| cfisch bigmem PENDING | 5 | squeue 2026-03-15 (Phase 12: ml_t ct/fu) |
| **total** | **48** (29R + 19PD) | squeue 2026-03-15 |

## Text Embeddings

| Fact | Value | Evidence |
| --- | --- | --- |
| artifacts complete | `true` | `docs/benchmarks/phase9_current_snapshot.json` |
| total rows | `5774931` | `runs/text_embeddings/embedding_metadata.json` |
| unique texts | `69697` | `runs/text_embeddings/embedding_metadata.json` |
| entities | `22569` | `runs/text_embeddings/embedding_metadata.json` |
| PCA dimension | `64` | `runs/text_embeddings/embedding_metadata.json` |

## Interpretation

1. V739 is fully landed and benchmarked. It ranks **#16/78** by mean rank across all conditions.
2. The remaining 12 partial models have gap-fill jobs running (17 SLURM jobs total).
3. NegativeBinomialGLM (20/104) has a structural numerical overflow failure and cannot complete.
4. The current physical Phase 9 results still represent the seed-replication reinterpretation of `core_text` / `full`.
5. V739 uses {core_only, core_edgar, core_text, full}; other models use {core_only, core_edgar, core_only_seed2, core_edgar_seed2}. Prior to Phase 12, text ablation was dead code (raw text strings stripped by `select_dtypes`). With PCA embeddings (float32), text ablation NOW WORKS. Phase 12 reruns are in progress.
6. Common misstatements about current status are documented in `docs/PHASE9_V739_FACT_ALIGNMENT.md`.
7. Top-5 models by mean rank: NHITS (4.21), PatchTST (4.36), NBEATS (4.77), NBEATSx (5.84), ChronosBolt (7.11).
8. NBEATS is the dominant champion model: 24/56 conditions won (43%).
9. Per-task champion distribution: NBEATS(8), NHITS(5+), KAN(5), DeepNPTS(4), Chronos(2+), GRU(3), V739(3), PatchTST(2).

## Immediate Next Actions

1. ~~Land the first canonical fair-line V739 results.~~ ✅ DONE (112/112)
2. Close the 12 remaining partial models (gap-fill jobs running — 13 jobs active).
3. ~~Submit real text/full reruns.~~ ✅ DONE (40 Phase 12 jobs submitted 2026-03-15, 20 running + 14 pending).
4. Wait for Phase 12 text reruns to land, then rebuild leaderboard with core_text/full results.
5. Only then start V740+ iteration.
