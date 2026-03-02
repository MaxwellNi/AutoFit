# Block 3 Benchmark Results

> Last Updated: 2026-03-03
> Full benchmark table: `docs/BLOCK3_FULL_SOTA_BENCHMARK.md`
> Live summary: `docs/BLOCK3_LIVE_SUMMARY.md`
> V7.3 execution spec: `docs/BLOCK3_V73_RESEARCH_EXECUTION_SPEC_20260225.md`

## Strict Snapshot

| Metric | Value | Evidence |
|---|---:|---|
| strict_records | 5586 | materialized metrics across 71 models |
| strict_condition_completion | 104/104 | all task x ablation x target x horizon covered |
| categories | 7 | autofit, deep_classical, foundation, irregular, ml_tabular, statistical, transformer_sota |
| champion_families | deep_classical=56, transformer_sota=26, foundation=22 | `docs/BLOCK3_FULL_SOTA_BENCHMARK.md` |

## Notes

1. V7.3 benchmark is in progress on Iris HPC (SLURM jobs running).
2. Full 104-condition champion table and per-model rankings are maintained in the dedicated documents above.
