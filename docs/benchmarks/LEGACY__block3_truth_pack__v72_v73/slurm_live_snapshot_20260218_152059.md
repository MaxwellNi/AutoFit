# Block3 Slurm Snapshot

## Snapshot Summary

| metric | value | evidence_path |
|---|---|---|
| snapshot_ts | 2026-02-18T15:20:58.887178+00:00 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_total | 7 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_total | 103 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| running_by_partition | {"batch": 7} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| pending_by_partition | {"batch": 95, "gpu": 8} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Pending Reason Top-K

| reason | count | evidence_path |
|---|---|---|
| (Priority) | 95 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| (QOSGrpNodeLimit) | 8 | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Prefix Status

| prefix | source | state_counts | evidence_path |
|---|---|---|---|
| p7 | squeue | {"PENDING": 5, "RUNNING": 7} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | squeue | {"PENDING": 2} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | squeue | {"PENDING": 66} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | squeue | {"PENDING": 30} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7 | sacct | {"CANCELLED": 66, "COMPLETED": 170, "FAILED": 8, "OUT_OF_ME": 4, "PENDING": 5, "RUNNING": 7} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7r | sacct | {"PENDING": 2} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7x | sacct | {"COMPLETED": 55, "PENDING": 66} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |
| p7xF | sacct | {"COMPLETED": 47, "PENDING": 30} | docs/benchmarks/block3_truth_pack/slurm_snapshot.json |

## Collection Commands

```bash
squeue -u $USER -h -o "%T %j %P %R"
sacct -u $USER -S 2026-02-12 -n -X -o JobName,State
sacctmgr show qos iris-batch-long,iris-gpu-long format=Name,MaxJobsPU,MaxWall,Priority -P -n
```
