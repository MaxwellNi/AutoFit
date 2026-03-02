# Block 3 Model Benchmark Status

> Last Updated: 2026-03-03
> Full benchmark table: `docs/BLOCK3_FULL_SOTA_BENCHMARK.md`
> V7.3 execution spec: `docs/BLOCK3_V73_RESEARCH_EXECUTION_SPEC_20260225.md`

## Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| strict_condition_completion | 104/104 | all 104 task x ablation x target x horizon keys covered |
| total_models | 71 | across 7 categories |
| total_metric_records | 5586 | materialized from Phase 7 benchmark |
| champion_models | NBEATS(41), Chronos(22), NHITS(15), KAN(10), DeepNPTS(8), PatchTST(4), NBEATSx(3), DLinear(1) | `docs/BLOCK3_FULL_SOTA_BENCHMARK.md` |

## Notes

1. V7.3 benchmark jobs are running on Iris HPC (SLURM).
2. V7.3.1 variant jobs submitted from secondary account for parallel evaluation.
