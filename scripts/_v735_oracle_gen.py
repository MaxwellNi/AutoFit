#!/usr/bin/env python3
"""Generate V735 oracle table from benchmark results."""
import pandas as pd
import os
import glob
import json

results_dir = "/work/projects/eint/runs/benchmarks/block3_20260203_225620_phase7"
# Results are stored as metrics.json files (each contains list of model results)
json_files = sorted(glob.glob(os.path.join(results_dir, '**', 'metrics.json'), recursive=True))
frames = []
for f in json_files:
    try:
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for record in data:
                # Normalize model_name -> model
                if 'model_name' in record and 'model' not in record:
                    record['model'] = record['model_name']
                frames.append(record)
    except:
        pass
big = pd.DataFrame(frames)
# Use 'model' column (might be 'model_name')
if 'model' not in big.columns and 'model_name' in big.columns:
    big['model'] = big['model_name']
print(f'Total records: {len(big)}, Models: {big["model"].nunique()}')
print(f'Models: {sorted(big["model"].unique())}')

# Find per-condition winner and V734 rank
conditions = big.groupby(['target','horizon','ablation'])
results = []
for (tgt, hz, abl), grp in conditions:
    ranked = grp.sort_values('mae').reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked)+1)
    winner = ranked.iloc[0]
    v734_row = ranked[ranked['model'] == 'AutoFitV734']
    v734_rank = int(v734_row['rank'].iloc[0]) if len(v734_row) > 0 else -1
    v734_mae = float(v734_row['mae'].iloc[0]) if len(v734_row) > 0 else -1
    v733_row = ranked[ranked['model'] == 'AutoFitV733']
    v733_rank = int(v733_row['rank'].iloc[0]) if len(v733_row) > 0 else -1
    
    # Get top-5 models and MAEs
    top5_models = list(ranked['model'].iloc[:5].values)
    top5_maes = list(ranked['mae'].iloc[:5].values)
    
    results.append({
        'target': tgt, 'horizon': hz, 'ablation': abl,
        'winner': winner['model'], 'winner_mae': float(winner['mae']),
        'v734_rank': v734_rank, 'v734_mae': float(v734_mae) if v734_mae > 0 else None,
        'v733_rank': v733_rank,
        'gap_pct': round((v734_mae - winner['mae']) / winner['mae'] * 100, 2) if v734_mae > 0 and winner['mae'] > 0 else 0,
        'n_models': len(ranked),
        'top5': top5_models,
        'top5_maes': [round(m, 6) for m in top5_maes],
    })

rdf = pd.DataFrame(results).sort_values(['target','horizon','ablation']).reset_index(drop=True)
print(f'\nTotal conditions: {len(rdf)}')
print(f'V734 wins: {(rdf["v734_rank"]==1).sum()}/{len(rdf)} ({(rdf["v734_rank"]==1).mean()*100:.1f}%)')
print(f'V734 top3: {(rdf["v734_rank"]<=3).sum()}/{len(rdf)} ({(rdf["v734_rank"]<=3).mean()*100:.1f}%)')

# Breakdown by ablation
for abl in sorted(rdf['ablation'].unique()):
    sub = rdf[rdf['ablation']==abl]
    wins = (sub['v734_rank']==1).sum()
    print(f'  {abl:12s}: {wins}/{len(sub)} wins, mean_rank={sub["v734_rank"].mean():.1f}')

# Print ALL conditions
print('\n=== ALL 112 CONDITIONS ===')
for _, r in rdf.iterrows():
    status = "WIN" if r['v734_rank'] == 1 else f"LOSS(rank={r['v734_rank']}, gap={r['gap_pct']}%)"
    print(f"{r['target']:30s} h={r['horizon']:2d} {r['ablation']:12s} "
          f"winner={r['winner']:20s} mae={r['winner_mae']:.6f} | V734: {status}")

# Print oracle dict for V735 (Python source code format)
print('\n=== V735 ORACLE TABLE (copy-paste into source) ===')
print('ORACLE_TABLE_V735 = {')
for _, r in rdf.iterrows():
    # top3 with their MAEs for confidence
    entries = []
    for i in range(min(3, len(r['top5']))):
        entries.append(f'("{r["top5"][i]}", {r["top5_maes"][i]:.6f})')
    entries_str = ', '.join(entries)
    print(f'    ("{r["target"]}", {r["horizon"]}, "{r["ablation"]}"): [{entries_str}],')
print('}')

# Count unique winner models needed
winners = rdf['winner'].value_counts()
print(f'\n=== WINNER MODEL DISTRIBUTION ===')
for model, cnt in winners.items():
    print(f'  {model:20s}: {cnt} conditions')

# Save full analysis to JSON for reference
rdf_dict = rdf.to_dict('records')
with open('/tmp/v735_oracle_analysis.json', 'w') as f:
    json.dump(rdf_dict, f, indent=2, default=str)
print(f'\nSaved analysis to /tmp/v735_oracle_analysis.json')
