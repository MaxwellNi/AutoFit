# Block3 Slurm Snapshot

## Snapshot Summary

| metric | value | evidence_path |
|---|---|---|
| snapshot_ts | 2026-02-23T12:43:06.589093+00:00 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_total | 8 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_total | 14 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_by_partition | {"batch": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_by_partition | {"batch": 6, "gpu": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Pending Reason Top-K

| reason | count | evidence_path |
|---|---|---|
| (QOSGrpNodeLimit) | 8 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| (QOSMaxJobsPerUserLimit) | 6 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Prefix Status

| prefix | source | state_counts | evidence_path |
|---|---|---|---|
| p7 | squeue | {"RUNNING": 1} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | squeue | {"RUNNING": 3} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | squeue | {"RUNNING": 1} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | squeue | {"PENDING": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7 | sacct | {"CANCELLED": 66, "COMPLETED": 177, "FAILED": 12, "OUT_OF_ME": 4, "RUNNING": 1} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | sacct | {"CANCELLED": 4, "COMPLETED": 7, "RUNNING": 3} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | sacct | {"CANCELLED": 33, "COMPLETED": 87, "RUNNING": 1} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | sacct | {"CANCELLED": 11, "COMPLETED": 58, "PENDING": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Collection Commands

```bash
squeue -u $USER -h -o "%T %j %P %R"
sacct -u $USER -S 2026-02-12 -n -X -o JobName,State
sacctmgr show qos iris-batch-long,iris-gpu-long format=Name,MaxJobsPU,MaxWall,Priority -P -n
```
