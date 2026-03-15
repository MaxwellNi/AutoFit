# Phase 9 / V739 Fact Alignment

> Last verified: 2026-03-14
> Purpose: prevent future contributors from mixing current Phase 9 truth with legacy exploratory evidence.

This document records the corrected, auditable statements that should govern all future discussion of the current benchmark line.

## Current Fact vs. Common Misstatement

| Common misstatement | Correct current fact | Evidence |
| --- | --- | --- |
| "V739 already has benchmark rankings." | No. Canonical fair-line V739 landed coverage is still `0/104`. | `docs/benchmarks/phase9_current_snapshot.json`, `docs/V739_CURRENT_RUN_MONITOR.md` |
| "The 8 live V739 jobs are already the fair-line run." | No. The observed live V739 jobs still target legacy `runs/benchmarks/block3_phase10/v739/`. | `scontrol show job`, `docs/V739_CURRENT_RUN_MONITOR.md` |
| "104 conditions are fake." | No. `104/104` remains the official benchmark completion unit. Repeated analytical surfaces do not invalidate the official unit. | Root `AGENTS.md`, `docs/CURRENT_SOURCE_OF_TRUTH.md` |
| "NF models completely ignore exogenous features." | Too broad. The codebase contains static-exog support paths, even if the current V739 candidate pool does not use them. | `src/narrative/block3/models/deep_models.py` |
| "We know which candidate model V739 selected in phase10 metrics." | No. Current artifacts do not expose a verified `selected_model` field for those legacy V739 outputs. | `runs/benchmarks/block3_phase10/v739/*/metrics.json` |
| "Text embeddings are still missing." | No. Text embedding artifacts now exist in `runs/text_embeddings/`. | `runs/text_embeddings/embedding_metadata.json`, `docs/benchmarks/phase9_current_snapshot.json` |
| "Embedding collapse root cause is fully proven." | No. Severe collapse is empirically observable, but root-cause attribution remains unverified. | `runs/text_embeddings/text_embeddings.parquet`, `scripts/generate_text_embeddings.py` |
| "V739 trains with less data than other models in the fair benchmark." | No. All models fit on the same train split; V739 additionally uses `val_raw` for candidate selection. | `scripts/run_block3_benchmark_shard.py`, `src/narrative/block3/models/nf_adaptive_champion.py` |
| "We should start V740 now." | No. The current work order is: land V739, close partial models, land real text/full reruns, then consider V740+. | Root `AGENTS.md`, `.local_mandatory_preexec.md` |

## Practical Interpretation Rules

1. Treat `runs/benchmarks/block3_phase9_fair/` as the only canonical benchmark root.
2. Treat `runs/benchmarks/block3_phase10/v739/` as legacy exploratory diagnostics only.
3. Treat `docs/_legacy_repo/` as archive material, not current operational truth.
4. Treat `docs/references/` as useful background knowledge, not current status truth.
5. If a statement depends on a current result, point to a current artifact.
6. If a statement depends on an inferred explanation, label it explicitly as a hypothesis.

## Minimal Audit Routine Before Making Claims

1. Refresh the current fact snapshot:

```bash
python3 scripts/build_phase9_current_snapshot.py
```

2. Inspect the current filtered leaderboard:

```bash
/mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/python3 scripts/aggregate_block3_results.py
```

3. Check live V739 queue targets directly:

```bash
squeue -u npin,cfisch | egrep 'JOBID|v739'
for jid in $(squeue -u npin,cfisch -h -o '%i %j' | awk '$2 ~ /v739/ {print $1}'); do
  scontrol show job "$jid" | egrep 'JobId=|JobName=|Command=|WorkDir=|StdOut=|StdErr='
done
```

4. Only then write a current-state conclusion.
