# AutoFit-TS

AutoFit-TS is a research-grade framework for learning and auditing temporal
foundation models on irregular, long-horizon longitudinal data. It is designed
for high-integrity scientific evaluation: reproducible benchmarks, strict
no-leakage validation, and concept-level explanations that support auditability
in downstream analyses.

Key features
- Automatic composition of temporal architectures to match dataset characteristics
  (irregular sampling, missingness, non-stationarity, multiscale rhythms, and
  exogenous influence) under a fixed compute/budget constraint.
- Multi-stream temporal backbone with modular exogenous conditioning (cross-
  attention, FiLM, bridge-token fusion) and interchangeable foundation wrappers.
- Concept Bottleneck layer that produces interpretable narrative indices that
  enable additive concept models and structured natural-language explanation
  reports.
- Parquet-native data ingestion and diagnostics, end-to-end benchmark pipelines,
  and automated generation of paper-ready evaluation tables (main, ablation,
  faithfulness, efficiency).

Repository layout
- `src/narrative/` — core pipeline, model components, diagnostics, and explainability
- `scripts/` — utilities for data preparation, benchmarking, auditing, and report generation
- `configs/` — canonical experiment/dataset configurations
- `docs/` — methodological notes, API references, and evaluation descriptors
- `runs/` — runtime artifacts and generated reports (not tracked in the canonical release)

Getting started (developer quickstart)
1. Create a Python environment (recommended: Python 3.9+).
2. Install runtime dependencies from `pyproject.toml`.
3. Run unit tests:

```bash
PYTHONPATH=src pytest -q
```

Example: small-scale smoke benchmark
```bash
# build a small offers parquet (implementation-specific)
# run a minimal benchmark with limit rows and a short horizon
PYTHONPATH=src python scripts/run_full_benchmark.py --offers_path data/raw/offers --limit_rows 5000 --label_horizon 3 --models dlinear patchtst
```

Citation
If you use this code in research, please cite the AutoFit-TS project and include a
reference to the accompanying paper (when available).

License and contribution
This repository is intended as a public research artifact. See `LICENSE` for terms.
Contributions are welcome via issues and pull requests; please follow the
repository's contribution guidelines.
