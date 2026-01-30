# Benchmark Truthfulness Audit Anchor (20260129_073037)

Public audit anchor for benchmark matrix truthfulness: paper tables reflect only actually executed runs. Host names: **3090**, **4090** only. No local absolute paths.

## Scope

- **Real runs:** Detected from run artifacts produced by `scripts/run_full_benchmark.py`: metrics.json, config yaml, STATUS_RUN.json. Each completed run has exp_name, backbone, fusion_type, module_variant, use_edgar, seed, and metrics (rmse/mae/mse/r2, train_time_sec, max_cuda_mem_mb).
- **Phantom rows:** Rows in main_results that have no backing run (combination of exp_name, backbone, fusion_type, module_variant, use_edgar). Marked and removed from final tables.
- **Source of truth:** `benchmark_truthfulness.json` (real_runs list). Paper tables are built ONLY from this via `scripts/make_paper_tables_v2.py`.

## Metric extraction mapping

- `rmse`: metrics.json .results[].rmse
- `mae`: metrics.json .results[].mae
- `mse`: metrics.json .results[].mse
- `r2`: metrics.json .results[].r2
- `train_time_sec`: metrics.json .results[].train_time_sec (or from timing log if available); if missing, note "timing_not_logged"
- `max_cuda_mem_mb`: metrics.json .results[].max_cuda_mem_mb

## Rules

- One row in main_results must correspond to >=1 real completed run_dir with a real metrics artifact.
- If a combination appears in tables but no backing run exists => mark as phantom and remove from final tables.
- No NaN in train_time_sec when a timing log exists; otherwise explicitly mark as missing with reason.
- main_results.csv and efficiency.csv are built ONLY from benchmark_truthfulness.json (make_paper_tables_v2.py). Aggregation across seeds: mean/std when multiple seeds; std blank when single seed.

## Reproducibility

**Audit (3090 or 4090):**

```bash
HOST_TAG=3090 python scripts/audit_benchmark_matrix_truthfulness.py \
  --bench_list runs/orchestrator/20260129_073037/bench_dirs_all.txt \
  --output_dir runs/orchestrator/20260129_073037/analysis \
  --paper_tables_dir runs/orchestrator/20260129_073037/paper_tables \
  2>&1 | tee runs/orchestrator/20260129_073037/analysis/logs/benchmark_truthfulness_3090.log
```

**Regenerate paper tables from truthfulness:**

```bash
python scripts/make_paper_tables_v2.py \
  --benchmark_truthfulness_json runs/orchestrator/20260129_073037/analysis/benchmark_truthfulness.json \
  --output_dir runs/orchestrator/20260129_073037/paper_tables
```

Outputs (local, untracked): benchmark_truthfulness.json, benchmark_truthfulness.md, logs/benchmark_truthfulness_{HOST_TAG}.log under analysis dir; main_results.csv, efficiency.csv under paper_tables.
