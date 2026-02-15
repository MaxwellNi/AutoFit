# Block 3 Operational SOP (Mandatory)

Last updated: 2026-02-15
Scope: Block 3 benchmark, AutoFit V7/V7.1/V7.2 preparation

## Non-Negotiable Gate

Every code change that can affect benchmark behavior must pass the following gate **before** any pilot/full submission:

1. Freeze pointer verification
2. Phase-7 fix verification
3. Targeted V7.1 tests (no-leakage, coverage guard, objective switch, reproducibility)
4. Synthetic smoke fit/predict with AutoFitV71 and strict coverage/routing audit

Canonical command:

```bash
bash scripts/preflight_block3_v71_gate.sh --v71-variant=g02
```

Submission scripts enforce this gate by default:

- `scripts/submit_phase7_full_benchmark.sh`
- `scripts/submit_phase7_v71_extreme.sh`
- `scripts/run_phase7_dual3090_safe.sh`

Unsafe bypass is blocked unless explicitly acknowledged:

```bash
ALLOW_UNSAFE_SKIP_PREFLIGHT=1 ... --skip-preflight
```

## Environment Policy (Strict)

- Use only existing `insider` environments.
- 3090 local server: `conda activate insider`.
- Iris/Slurm jobs: `micromamba/conda activate insider` in job script preamble.
- Do not create new environments.
- Do not install runtime dependencies into `base`.

Known repo roots by host:

- 3090 server: `/home/pni/project/repo_root`
- 4090 server: `/home/pni/projects/repo_root`
- Iris: `/home/users/npin/repo_root`

If freeze assets are stored outside `repo_root/runs`, set:

```bash
export BLOCK3_RUNS_ROOT=/path/to/shared/runs
```

Preflight will auto-link `repo_root/runs -> $BLOCK3_RUNS_ROOT` when needed.

To fetch required freeze assets directly from Iris on local servers:

```bash
bash scripts/pull_block3_freeze_from_iris.sh --iris-login npin@iris
```

If your SSH alias is simply `iris`, you can omit `npin@`.

Dependency installer (optional, insider-only):

```bash
bash scripts/install_block3_deps_in_insider.sh
```

## Dual 3090 Safe Saturation Policy

Primary launcher:

```bash
bash scripts/run_phase7_dual3090_safe.sh --full --v71-variant=g02
```

The launcher auto-activates `insider` and auto-repairs missing/incompatible dependencies
before preflight and benchmark execution.

Scheduling design:

- `GPU0`: `deep_classical`, `transformer_sota_A`, `foundation_A`
- `GPU1`: `transformer_sota_B`, `foundation_B`, `irregular`
- `CPU-A`: `ml_tabular`, `autofit_af1`
- `CPU-B`: `statistical`, `autofit_af2` (contains `AutoFitV71` override)

OOM guards:

- Host memory watermark before each shard launch (`MemAvailable` guard)
- GPU free-memory watermark before each GPU shard launch
- Thread caps per worker (`OMP/MKL/OPENBLAS`) to avoid CPU oversubscription
- Per-worker isolated logs and failure accounting

## Fairness and Leakage Policy

- Temporal split + embargo only
- No test-target feedback into training/routing/selection
- Coverage guard must pass (`prediction_coverage_ratio >= 0.98`)
- Failed fairness checks are invalid for leaderboard comparisons

## Release Checklist

1. `preflight_block3_v71_gate.sh` PASS
2. Critical tests PASS
3. Smoke PASS with fairness guard
4. Pilot PASS gates
5. Full run submission
6. Aggregate with comparability filter enabled
