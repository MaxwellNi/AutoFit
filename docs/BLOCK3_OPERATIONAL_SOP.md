# Block 3 Operational SOP (Mandatory)

Last updated: 2026-03-05
Scope: Block 3 benchmark, AutoFit V7/V7.1/V7.2/V7.3 preparation

## Mandatory Contract

Before every submit/run path, execute:

```bash
python3 scripts/assert_block3_execution_contract.py --entrypoint <script-path>
```

Canonical contract file:

- `docs/BLOCK3_EXECUTION_CONTRACT.md`

If the assertion fails, stop immediately and fix blockers before scheduling.

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
- Iris (npin): `/home/users/npin/repo_root`
- Iris (cfisch): shared GPFS at `/mnt/aiongpfs/projects/eint/repo_root`

## Secondary Account: cfisch (Iris HPC)

The project uses a secondary account (`cfisch`) for parallel job scheduling on
Iris HPC. Both accounts share the same GPFS workspace and conda environment.

### Access

```bash
ssh iris-cf    # SSH alias for cfisch@iris-cluster
```

The SSH alias `iris-cf` must be configured in `~/.ssh/config` on the access node.

### Prerequisites (one-time setup, already completed)

1. **Git safe directory** — the repo is owned by `npin`; `cfisch` must mark it
   safe to allow git operations inside SLURM jobs:

   ```bash
   ssh iris-cf 'git config --global --add safe.directory /mnt/aiongpfs/projects/eint/repo_root'
   ```

2. **Conda environment permissions** — the `insider` conda env was created by
   `npin` with `0755` permissions on Python binaries. Group execute bits must be
   set for the `eint` group so cfisch can run them:

   ```bash
   chmod -R g+rX /mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/bin/
   chmod -R g+rX /mnt/aiongpfs/projects/eint/envs/.micromamba/envs/insider/lib/
   ```

3. **Output directory permissions** — benchmark result directories must be
   group-writable. The `runs/benchmarks/` tree uses `rwxrwsrwx` (setgid),
   which propagates group ownership automatically. The contract audit output
   file also needs group-write:

   ```bash
   chmod -R g+w docs/benchmarks/block3_truth_pack/
   ```

### Job submission from cfisch

cfisch does not have SSH keys for the GitHub remote. The `git pull` step in
SLURM scripts will fail, but scripts use `|| echo "WARN: git pull failed"` to
prevent job abort. The shared GPFS means any code pushed by npin is immediately
visible to cfisch without pulling.

To submit jobs as cfisch:

```bash
ssh iris-cf "cd /mnt/aiongpfs/projects/eint/repo_root/.slurm_scripts/cfisch_batch2 && sbatch --parsable <script>.sh"
```

Script directories for cfisch:
- `.slurm_scripts/cfisch_batch2/` — GPU-partition jobs (baselines + V735 co/ct)
- `.slurm_scripts/cfisch_bigmem/` — bigmem-partition jobs (ML tabular + statistical ce/fu)

### QOS limits (shared across ALL users in each QOS)

| QOS | GrpNodes | MaxJobsPU | RAM/Node | GPU | Notes |
|---|---:|---:|---:|---|---|
| iris-gpu-long | 6 | 4 | 756G | 4x V100 32GB | GrpNodes is QOS-wide, not per-user |
| iris-bigmem-long | 2 | 4 | 3024G | none | For CPU-only ML/statistical tasks |
| iris-batch-long | 24 | 8 | 112G | none | Too small for most benchmark tasks |

`GrpNodes` is the total node count available to the entire QOS across all users,
not a per-account limit. When all 6 GPU nodes are occupied (by any user in the
QOS), new jobs queue with reason `QOSGrpNodeLimit`. `MaxJobsPU` is per-user.

Because both `npin` and `cfisch` submit under `--account=christian.fisch`, their
jobs draw from the same QOS pool. This means the two accounts do not double the
available nodes — they share the same 6-node GPU cap, but each gets its own
4-job-per-user limit.

### Memory allocation by ablation (observed)

| Ablation | Peak RSS | Recommended `--mem` |
|---|---:|---:|
| core_only | ~113G | 200G |
| core_text | ~282G | 350G |
| core_edgar | ~163G | 200G |
| full | ~337G | 400G |

### Partitioning strategy

- **GPU partition**: deep_classical, foundation, transformer_sota, autofit, irregular
- **bigmem partition**: ml_tabular, statistical (CPU-only; no GPU needed)

Migrating ML/statistical tasks to bigmem frees GPU QOS slots for models that
need GPU acceleration.

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

## V7.2 Failure-Pool Queue Soft Bump

Use the soft-bump helper to accelerate the four fixed V7.2 failure-pool jobs
without changing benchmark semantics.

Stage t0 (priority bump):

```bash
bash scripts/soft_bump_v72_failure_pool_queue.sh --stage=t0 --apply
```

If `scontrol top` is denied on your cluster, use stage t2h fallback:

```bash
bash scripts/soft_bump_v72_failure_pool_queue.sh --stage=t2h --apply --hold-count=6
```

After at least one `p7r_v72_ic_ce_h*` job starts running, release held jobs:

```bash
bash scripts/soft_bump_v72_failure_pool_queue.sh --release-held --apply
```

Optional auto-release watcher (recommended during overnight queue wait):

```bash
nohup bash scripts/watch_v72_fasttrack_release.sh --interval=120 --max-wait-min=2880 \
  >/tmp/v72_fasttrack_release.log 2>&1 &
tail -f /tmp/v72_fasttrack_release.log
```

## Release Checklist

1. `preflight_block3_v71_gate.sh` PASS
2. Critical tests PASS
3. Smoke PASS with fairness guard
4. Pilot PASS gates
5. Full run submission
6. Aggregate with comparability filter enabled
7. Build truth pack and review V7.2 evidence master:
   - `micromamba run -n insider python scripts/build_block3_truth_pack.py --output-dir docs/benchmarks/block3_truth_pack`
   - `docs/AUTOFIT_V72_EVIDENCE_MASTER_20260217.md`
