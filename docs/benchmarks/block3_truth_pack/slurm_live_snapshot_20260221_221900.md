# Block3 Slurm Snapshot

## Snapshot Summary

| metric | value | evidence_path |
|---|---|---|
| snapshot_ts | 2026-02-21T22:19:00.241471+00:00 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_total | 8 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_total | 77 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_by_partition | {"batch": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_by_partition | {"batch": 69, "gpu": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Pending Reason Top-K

| reason | count | evidence_path |
|---|---|---|
| (QOSMaxJobsPerUserLimit) | 69 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| (QOSGrpNodeLimit) | 8 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Prefix Status

| prefix | source | state_counts | evidence_path |
|---|---|---|---|
| p7 | squeue | {"PENDING": 1} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | squeue | {"PENDING": 3, "RUNNING": 3} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | squeue | {"PENDING": 43, "RUNNING": 5} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | squeue | {"PENDING": 30} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7 | sacct | {"CANCELLED": 66, "COMPLETED": 177, "FAILED": 12, "OUT_OF_ME": 4, "PENDING": 1} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | sacct | {"CANCELLED": 4, "COMPLETED": 4, "PENDING": 3, "RUNNING": 3} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | sacct | {"COMPLETED": 73, "PENDING": 43, "RUNNING": 5} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | sacct | {"COMPLETED": 47, "PENDING": 30} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Collection Commands

```bash
squeue -u $USER -h -o "%T %j %P %R"
sacct -u $USER -S 2026-02-12 -n -X -o JobName,State
sacctmgr show qos iris-batch-long,iris-gpu-long format=Name,MaxJobsPU,MaxWall,Priority -P -n
```
