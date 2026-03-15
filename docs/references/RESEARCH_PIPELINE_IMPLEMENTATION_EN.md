# Research Pipeline Implementation

This document summarizes the design, modules, and reproducible evaluation
practices for the AutoFit-TS research pipeline. It is intended for a public
research audience and emphasizes methodology, interfaces, and developer
guidance.

## Objectives

- Robust temporal representation learning for irregular, long-horizon panel data
  with heterogeneous exogenous signals.
- Automated, budget-aware composition of candidate architectures tailored to
  measured dataset characteristics (non-stationarity, long memory, multiscale
  periodicity, irregular sampling, exogenous strength).
- Auditable, concept-level explanations via an explicit concept bottleneck and
  additive concept models.

## High-level pipeline

1. Parquet-native ingestion: partition-aware readers that load only required
   columns and avoid CSV fallbacks.
2. Schema profiling: automated detection of candidate time columns, join keys,
   and a reproducible schema profile artifact.
3. Timeline normalization: build irregular event indices, compute delta-times,
   and produce truncation/cutoff masks for fixed-length inputs.
4. Feature stores: compact aggregations and joins for EDGAR and other external
   datasets (last, mean, EMA).
5. Candidate composition: assemble modular pipelines
   (IrregularPatch → Encoder → Fusion → Head) with optional exogenous
   conditioning and concept bottleneck.
6. Budget-aware search: successive halving with early stopping; export
   checkpoints and best_config.yaml for final training.
7. Explainability: concept bottleneck, attribution exporter (concept/time/exogenous),
   and faithfulness tests (deletion/insertion, counterfactual ablations).
8. Evaluation and reporting: parquet-native aggregation that emits four paper
   tables (main, ablation, faithfulness, efficiency).

## Core modules (map)

- `src/narrative/data_preprocessing/` — ingestion, schema profiling, timeline utilities
- `src/narrative/models/` — embeddings, encoders, fusion, and foundation wrappers
- `src/narrative/auto_fit/` — candidate composition, budget search, dataset diagnostics
- `src/narrative/explainability/` — attribution exporter and faithfulness tests

## Reproducibility & auditing

- Experiments record resolved configurations and artifacts under `runs/<exp>/`.
- Benchmarks produce manifests and paper-table outputs (parquet/CSV) to facilitate
  reproducible downstream analysis.
- The codebase enforces strict no-leakage rules for time-splitting and label
  generation; see evaluation modules for validation gate implementations.

## Developer quickstart

1. Create a Python 3.9+ environment and install dependencies from `pyproject.toml`.
2. Run unit tests:

```bash
PYTHONPATH=src pytest -q
```

3. Run a small smoke benchmark:

```bash
PYTHONPATH=src python scripts/run_full_benchmark.py --offers_path data/raw/offers --limit_rows 5000 --label_horizon 3 --models dlinear patchtst
```

## Extensibility

- Candidate generators and budget policies are modular. Extend
  `src/narrative/auto_fit/compose_candidates.py` and
  `src/narrative/auto_fit/budget_search.py` to add new architectures or search
  strategies.
- The foundation wrapper layer allows plugging experimental backbone implementations
  with minimal interface changes.

For detailed API usage, consult docstrings in `src/narrative/` and the example
configs in `configs/`.

