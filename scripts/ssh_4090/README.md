# 4090 SSH Adapters

These scripts adapt the SLURM submission workflows to run directly on the SSH 4090 host.

## Environment
- Default repo root: `/home/pni/projects/repo_root`
- Uses `micromamba` if available, otherwise `conda`

## Wide freeze (resume from daily)
```bash
WIDE_STAMP=20260203_225620 bash scripts/ssh_4090/run_wide_freeze_aion_from_daily_4090.sh
```

## Wide freeze (full)
```bash
bash scripts/ssh_4090/run_wide_freeze_aion_4090.sh
```

## Monitor (local only)
```bash
WIDE_STAMP=20260203_225620 bash scripts/ssh_4090/monitor_wide_freeze_aion_4090.sh
```

## GPU workflows (single 4090)
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/ssh_4090/run_autofit_search_4090.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/ssh_4090/run_benchmark_matrix_4090.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/ssh_4090/run_final_training_4090.sh
```

## Matrix entry (no SLURM array)
```bash
INDEX=8 bash scripts/ssh_4090/launch_iris_4090.sh
```

## Required inputs on 4090
- `data/raw/offers`
- `data/raw/edgar/accessions`
- `runs/offers_core_full_snapshot_wide_${WIDE_STAMP}`
- `runs/orchestrator/20260129_073037/analysis/wide_${WIDE_STAMP}`
- `runs/offers_text_v1_20260129_073037_full` (if referenced in audits)
- `runs/selections/b11_v2_canonical/sampled_entities.json`

## Data sync (via Mac)
From the cluster host, sync outputs to your Mac:
```bash
MAC_HOST=<mac-host-or-ip> MAC_USER=<mac-user> \
WIDE_STAMP=20260203_225620 bash scripts/sync_outputs_to_mac.sh
```

From your Mac, sync to 4090:
```bash
REMOTE_HOST=4090 REMOTE_USER=pni \
WIDE_STAMP=20260203_225620 bash scripts/sync_outputs_to_4090.sh
```
