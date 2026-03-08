#!/usr/bin/env python3
"""
Phase 8 benchmark leaderboard — correct methodology.

Condition = (task, ablation, target, horizon)
Ranking = per-condition RMSE rank, then averaged across all conditions.
Filters: fairness_pass=True AND prediction_coverage_ratio >= 0.98
"""
import json, glob, os, sys
from collections import defaultdict

base = 'runs/benchmarks/block3_20260203_225620_phase7'

all_records = []
for f in sorted(glob.glob(base + '/**/metrics.json', recursive=True)):
    try:
        data = json.load(open(f))
        all_records.extend(data)
    except:
        continue

# Apply comparability filter
filtered = []
rejected_fairness = 0
rejected_coverage = 0
for r in all_records:
    if r.get('fairness_pass') is not True:
        rejected_fairness += 1
        continue
    try:
        cov = float(r.get('prediction_coverage_ratio', 0))
    except:
        cov = 0
    if cov < 0.98:
        rejected_coverage += 1
        continue
    filtered.append(r)

# Model category mapping
model_cat = {}
for r in filtered:
    model_cat[r.get('model_name', '?')] = r.get('category', '?')

# Group by condition = (task, ablation, target, horizon)
conditions = defaultdict(list)
for r in filtered:
    cond = (r.get('task'), r.get('ablation'), r.get('target'), r.get('horizon'))
    conditions[cond].append(r)

# Per-condition ranking
model_ranks = defaultdict(list)
model_champion = defaultdict(int)
model_rmse_values = defaultdict(list)

for cond, records in sorted(conditions.items()):
    model_best = {}
    for r in records:
        mn = r.get('model_name', '?')
        rmse = r.get('rmse')
        if rmse is None:
            continue
        try:
            rmse = float(rmse)
            if rmse != rmse or rmse > 1e15:
                continue
        except:
            continue
        if mn not in model_best or rmse < model_best[mn]:
            model_best[mn] = rmse

    sorted_models = sorted(model_best.items(), key=lambda x: x[1])
    for rank, (mn, val) in enumerate(sorted_models, 1):
        model_ranks[mn].append(rank)
        model_rmse_values[mn].append(val)
        if rank == 1:
            model_champion[mn] += 1

# Build leaderboard
leaderboard = []
for mn in model_ranks:
    ranks = model_ranks[mn]
    leaderboard.append({
        'model': mn,
        'category': model_cat.get(mn, '?'),
        'n_conds': len(ranks),
        'avg_rank': sum(ranks) / len(ranks),
        'champion': model_champion.get(mn, 0),
        'median_rmse': sorted(model_rmse_values[mn])[len(model_rmse_values[mn])//2],
    })
leaderboard.sort(key=lambda x: x['avg_rank'])

# Print summary stats
n_conds = len(conditions)
n_models = len(leaderboard)
print(f'Records: {len(all_records)} raw → {len(filtered)} filtered '
      f'(rejected: {rejected_fairness} fairness, {rejected_coverage} coverage)')
print(f'Conditions: {n_conds} | Models: {n_models}')
print()

# Category breakdown
cats = defaultdict(set)
for row in leaderboard:
    cats[row['category']].add(row['model'])
print('Models per category:')
for cat in sorted(cats):
    print(f'  {cat}: {len(cats[cat])} models')
print()

# Full leaderboard
print('=' * 100)
print(f'  LEADERBOARD — {n_models} models × {n_conds} conditions ({len(filtered)} filtered records)')
print('=' * 100)
print(f'{"#":>3}  {"Model":<26} {"Category":<18} {"Conds":>5} {"AvgRank":>8} {"Champ":>5}')
print('-' * 100)
for i, row in enumerate(leaderboard, 1):
    print(f'{i:>3}  {row["model"]:<26} {row["category"]:<18} {row["n_conds"]:>5} '
          f'{row["avg_rank"]:>8.2f} {row["champion"]:>5}')

# Champion distribution
print()
print('CHAMPION DISTRIBUTION:')
champs = [r for r in leaderboard if r['champion'] > 0]
for row in sorted(champs, key=lambda x: -x['champion']):
    pct = row['champion'] / n_conds * 100
    print(f'  {row["model"]:<26} {row["champion"]:>3} / {n_conds} ({pct:.1f}%)')

# Per-ablation leaderboard (top 15)
print()
print('=' * 100)
print('  PER-ABLATION LEADERBOARD (top 15)')
print('=' * 100)
for ablation in ['core_only', 'core_text', 'core_edgar', 'full']:
    abl_conds = {c: r for c, r in conditions.items() if c[1] == ablation}
    if not abl_conds:
        continue
    abl_ranks = defaultdict(list)
    abl_champ = defaultdict(int)
    for cond, records in sorted(abl_conds.items()):
        model_best = {}
        for r in records:
            mn = r.get('model_name', '?')
            rmse = r.get('rmse')
            if rmse is None: continue
            try:
                rmse = float(rmse)
                if rmse != rmse or rmse > 1e15: continue
            except: continue
            if mn not in model_best or rmse < model_best[mn]:
                model_best[mn] = rmse
        sorted_m = sorted(model_best.items(), key=lambda x: x[1])
        for rank, (mn, val) in enumerate(sorted_m, 1):
            abl_ranks[mn].append(rank)
            if rank == 1:
                abl_champ[mn] += 1

    abl_lb = []
    for mn in abl_ranks:
        r = abl_ranks[mn]
        abl_lb.append({'model': mn, 'n': len(r), 'avg': sum(r)/len(r), 'ch': abl_champ.get(mn, 0)})
    abl_lb.sort(key=lambda x: x['avg'])

    print(f'\n  [{ablation.upper()}] ({len(abl_conds)} conditions)')
    print(f'  {"#":>3}  {"Model":<26} {"Conds":>5} {"AvgRank":>8} {"Champ":>5}')
    print(f'  {"-"*65}')
    for i, row in enumerate(abl_lb[:15], 1):
        print(f'  {i:>3}  {row["model"]:<26} {row["n"]:>5} {row["avg"]:>8.2f} {row["ch"]:>5}')

# V735 / V736 detailed
print()
print('=' * 100)
print('  V735 / V736 DETAILED BREAKDOWN')
print('=' * 100)
for target_model in ['AutoFitV735', 'AutoFitV736']:
    idx = [i+1 for i, r in enumerate(leaderboard) if r['model'] == target_model]
    data = [r for r in leaderboard if r['model'] == target_model]
    if not data:
        print(f'\n  {target_model}: NOT FOUND IN DATA')
        continue
    d = data[0]
    print(f'\n  {target_model}: #{idx[0]}/{n_models} overall (avg rank {d["avg_rank"]:.2f}, '
          f'{d["n_conds"]} conditions, {d["champion"]} champion wins)')
    
    # Per-ablation
    for ablation in ['core_only', 'core_text', 'core_edgar', 'full']:
        abl_ranks = []
        for cond in sorted(conditions.keys()):
            if cond[1] != ablation: continue
            records = conditions[cond]
            model_best = {}
            for r in records:
                mn = r.get('model_name', '?')
                rmse = r.get('rmse')
                if rmse is None: continue
                try:
                    rmse = float(rmse)
                    if rmse != rmse or rmse > 1e15: continue
                except: continue
                if mn not in model_best or rmse < model_best[mn]:
                    model_best[mn] = rmse
            if target_model not in model_best:
                continue
            sorted_m = sorted(model_best.items(), key=lambda x: x[1])
            total = len(sorted_m)
            for rank, (mn, val) in enumerate(sorted_m, 1):
                if mn == target_model:
                    abl_ranks.append((rank, total, cond))
                    break
        if abl_ranks:
            avg = sum(r[0] for r in abl_ranks) / len(abl_ranks)
            best = min(r[0] for r in abl_ranks)
            worst = max(r[0] for r in abl_ranks)
            print(f'    {ablation:<14}: avg={avg:.1f}  range=[{best}-{worst}]  '
                  f'across {len(abl_ranks)} conditions')
        else:
            print(f'    {ablation:<14}: NO DATA')

# Per-task breakdown for V735/V736
    print(f'  Per-task:')
    for task in ['task1_outcome', 'task2_forecast', 'task3_risk_adjust']:
        task_ranks = []
        for cond in sorted(conditions.keys()):
            if cond[0] != task: continue
            records = conditions[cond]
            model_best = {}
            for r in records:
                mn = r.get('model_name', '?')
                rmse = r.get('rmse')
                if rmse is None: continue
                try:
                    rmse = float(rmse)
                    if rmse != rmse or rmse > 1e15: continue
                except: continue
                if mn not in model_best or rmse < model_best[mn]:
                    model_best[mn] = rmse
            if target_model not in model_best:
                continue
            sorted_m = sorted(model_best.items(), key=lambda x: x[1])
            total = len(sorted_m)
            for rank, (mn, val) in enumerate(sorted_m, 1):
                if mn == target_model:
                    task_ranks.append(rank)
                    break
        if task_ranks:
            avg = sum(task_ranks) / len(task_ranks)
            print(f'    {task:<22}: avg rank={avg:.1f} across {len(task_ranks)} conditions')
