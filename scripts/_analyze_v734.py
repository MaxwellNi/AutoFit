#!/usr/bin/env python3
"""Analyze V734 results and current championship status."""
import json, pathlib, sys
from collections import defaultdict

RESULT_DIRS = [
    pathlib.Path('/mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_phase7'),
    pathlib.Path('/mnt/aiongpfs/projects/eint/runs/benchmarks/block3_20260203_225620_iris_full'),
]
TARGETS = ['funding_raised_usd', 'investors_count', 'is_funded']
HORIZONS = [1, 7, 14, 30]
ABLATIONS = ['core_only', 'core_text', 'core_edgar', 'full']

records = []
seen = set()
for d in RESULT_DIRS:
    if not d.exists(): continue
    for f in d.rglob('metrics.json'):
        k = str(f.resolve())
        if k in seen: continue
        seen.add(k)
        data = json.loads(f.read_text())
        if isinstance(data, list):
            records.extend(data)

groups = defaultdict(list)
for r in records:
    t, h, a = r.get('target',''), r.get('horizon'), r.get('ablation','')
    mae, model = r.get('mae'), r.get('model_name','')
    if t in TARGETS and h in HORIZONS and a in ABLATIONS and mae is not None:
        groups[(t, h, a)].append((float(mae), model))

# Unique models
all_models = set()
for entries in groups.values():
    for _, m in entries:
        all_models.add(m)

print(f"Total records: {len(records)}")
print(f"Unique models: {len(all_models)}")
print(f"Total conditions: {len(groups)}/48")
print(f"Models: {sorted(all_models)}")
print()

# V734 analysis
v734_wins = 0
v734_details = []
for key in sorted(groups):
    entries = sorted(groups[key], key=lambda x: (x[0], x[1]))
    v734_rank, v734_mae = None, None
    for i, (mae, m) in enumerate(entries):
        if 'AutoFitV734' in m:
            v734_mae, v734_rank = mae, i + 1
            break
    best_mae, best_model = entries[0]
    if v734_rank == 1: v734_wins += 1
    gap = ((v734_mae - best_mae) / best_mae * 100) if v734_mae and best_mae > 0 else 0
    v734_details.append((key, v734_rank, v734_mae, best_model, best_mae, gap, len(entries)))

print(f"V734 wins: {v734_wins}/{len(groups)} ({v734_wins/len(groups)*100:.1f}%)")
print()

by_abl = defaultdict(lambda: {'wins':0, 'total':0, 'ranks':[], 'gaps':[]})
for (t,h,a), rank, mae, best_m, best_mae, gap, n in v734_details:
    by_abl[a]['total'] += 1
    if rank: by_abl[a]['ranks'].append(rank)
    if rank == 1: by_abl[a]['wins'] += 1
    by_abl[a]['gaps'].append(gap)

print("=== V734 by ablation ===")
for a in ABLATIONS:
    d = by_abl[a]
    avg_rank = sum(d['ranks'])/len(d['ranks']) if d['ranks'] else 0
    avg_gap = sum(d['gaps'])/len(d['gaps']) if d['gaps'] else 0
    print(f"  {a:12s}: {d['wins']}/{d['total']} wins, avg_rank={avg_rank:.1f}, avg_gap={avg_gap:.2f}%")

print()
print("=== ALL conditions detail ===")
for (t,h,a), rank, mae, best_m, best_mae, gap, n in v734_details:
    status = "WIN" if rank == 1 else f"LOSS(rank={rank},gap={gap:.4f}%)" if rank else "MISSING"
    print(f"  ({t}, h={h:2d}, {a:12s}): {status:40s} winner={best_m:12s} mae={best_mae:.6f} | V734={'N/A' if mae is None else f'{mae:.6f}'} | n_models={n}")
