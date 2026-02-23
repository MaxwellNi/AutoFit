# Block3 Slurm Snapshot

## Snapshot Summary

| metric | value | evidence_path |
|---|---|---|
| snapshot_ts | 2026-02-23T14:47:22.088014+00:00 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_total | 8 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_total | 12 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_by_partition | {"batch": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_by_partition | {"batch": 4, "gpu": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Pending Reason Top-K

| reason | count | evidence_path |
|---|---|---|
| (QOSGrpNodeLimit) | 8 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| (QOSMaxJobsPerUserLimit) | 4 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Prefix Status

| prefix | source | state_counts | evidence_path |
|---|---|---|---|
| p7 | squeue | {} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | squeue | {"RUNNING": 3} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | squeue | {} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | squeue | {"PENDING": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7 | sacct | {"CANCELLED": 66, "COMPLETED": 178, "FAILED": 12, "OUT_OF_ME": 4} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | sacct | {"CANCELLED": 4, "COMPLETED": 7, "RUNNING": 3} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | sacct | {"CANCELLED": 33, "COMPLETED": 88} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | sacct | {"CANCELLED": 11, "COMPLETED": 58, "PENDING": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Collection Commands

```bash
squeue -u $USER -h -o "%T %j %P %R"
sacct -u $USER -S 2026-02-12 -n -X -o JobName,State
sacctmgr show qos iris-batch-long,iris-gpu-long format=Name,MaxJobsPU,MaxWall,Priority -P -n
```
