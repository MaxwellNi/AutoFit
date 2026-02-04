# Agent Context

## Mission
Upgrade `TRAIN_WIDE_FINAL` to a true wide feature freeze with highest-tier audits, so future modeling does not require backfilling features. This is the last data freeze before Block 3 training.

## Hard Constraints
- Only commit/push: `scripts/`, `configs/`, `docs/audits/`, and `docs/audits/MANIFEST.json`.
- Never commit anything under `runs/`.
- Never create/track files whose **names** contain: `cursor`, `prompt`, `recovery`, `runbook`, `decision`, `transcript` (case-insensitive).
- All tracked files must be **English** only.
- Block 3 must read **only** `docs/audits/FULL_SCALE_POINTER.yaml`.

## Current State (as of 2026-02-04)
- `WIDE_STAMP=20260203_225620` has completed Steps 1-3 only.
- Step 3 output exists:
  - `runs/offers_core_full_snapshot_wide_20260203_225620/offers_core_snapshot.parquet`
  - `runs/offers_core_full_snapshot_wide_20260203_225620/MANIFEST.json`
- Step 4+ outputs are not complete yet (daily directory exists but no final artifacts).
- Active/pending queue jobs:
  - `5168794` (`wide_freeze_aion_resume`) `PENDING` reason `Priority`
  - `5168809` (`wide_freeze_aion_resume`) `PENDING` reason `Priority`
  - `5168800` (`monitor_wide_freeze_aion`) `PENDING` reason `Priority`
- Last observed pending rank in `batch` was approximately 389-391.

## Execution History Captured From This Thread
1. Full run `11029034` executed Steps 1-3 and then failed at Step 4 with OOM.
2. Resume run `11029184` failed before Step 4 due to `bash: No such file or directory` inside `srun`.
3. Resource mismatch fixed (`ntasks-per-node=1`, `-c 16`) after submission error: `CPU count per node can not be satisfied`.
4. Added `BASH_BIN` detection and hard-fail behavior in aion sbatch wrappers.
5. Added DuckDB controls for daily build:
   - `DUCKDB_MEMORY_LIMIT_GB`
   - `DUCKDB_THREADS`
6. Added dependency preflight in resume sbatch (`duckdb`, `deltalake`).
7. Added monitor loop and monitor sbatch for auto-check/auto-resubmit behavior:
   - `scripts/monitor_wide_freeze_resume.sh`
   - `scripts/slurm/monitor_wide_freeze_aion.sbatch`
8. Added 4090 non-SLURM adapters under `scripts/ssh_4090/`.
9. Added sync workflows:
   - Cluster push to Mac: `scripts/sync_outputs_to_mac.sh`
   - Mac push to 4090: `scripts/sync_outputs_to_4090.sh`
   - Mac pull from cluster (preferred): `scripts/pull_outputs_from_cluster.sh`
10. Connectivity findings:
   - Cluster host cannot directly reach 4090 from this environment (no VPN path).
   - Mac DNS/port issues were resolved in scripts by defaulting ULHPC login to `access-iris.uni.lu:8022`.
   - Current remaining blocker on Mac is SSH auth to ULHPC (`Permission denied (publickey)`), not routing.

## Key Scripts (Updated)
- Full pipeline: `scripts/run_wide_freeze_full.sh`
- Resume pipeline (Step 4-10): `scripts/run_wide_freeze_from_daily.sh`
- Aion full sbatch: `scripts/slurm/run_wide_freeze_aion.sbatch`
- Aion resume sbatch: `scripts/slurm/run_wide_freeze_aion_from_daily.sbatch`
- Monitor loop: `scripts/monitor_wide_freeze_resume.sh`
- Monitor sbatch: `scripts/slurm/monitor_wide_freeze_aion.sbatch`
- 4090 local runners: `scripts/ssh_4090/*.sh`
- Sync helpers:
  - `scripts/pull_outputs_from_cluster.sh`
  - `scripts/sync_outputs_to_mac.sh`
  - `scripts/sync_outputs_to_4090.sh`

## Commit Trail (Thread-Relevant)
- `47a8db3` Add 4090 adapters and wide-freeze updates.
- `1a15ea6` Add mac-hop sync scripts for 4090 workflow.
- `2ac1477` Add Mac-side pull script for cluster outputs.
- `299c216` Use ULHPC login host/port defaults for pull sync.

## Pending Work
1. Resolve Mac -> ULHPC SSH key auth (`id_ed25519_iris` expected).
2. Pull data from cluster to Mac with `scripts/pull_outputs_from_cluster.sh`.
3. Push pulled artifacts from Mac to 4090 with `scripts/sync_outputs_to_4090.sh`.
4. On 4090, run Step 4-10:
   - `WIDE_STAMP=20260203_225620 bash scripts/ssh_4090/run_wide_freeze_aion_from_daily_4090.sh`
5. Validate Step 4-10 artifacts and pointer before Block 3 handoff.
